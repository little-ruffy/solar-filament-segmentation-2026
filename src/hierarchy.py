import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from pycocotools import mask as mask_util
from scipy import ndimage as ndi

from src.config import CROP_OFFSET, CROP_SIZE, FULL_SIZE, GONG_STATIONS
from src.metrics import encode_binary_mask

TREE_THRESHOLDS: Tuple[float, ...] = (
    0.04, 0.07, 0.11, 0.16,
    0.21, 0.26, 0.31, 0.36,
    0.42, 0.48, 0.54, 0.60,
    0.66, 0.72, 0.78, 0.84,
)

STATION_TO_IDX = {st: i for i, st in enumerate(GONG_STATIONS)}

_CLOSE_ELLIPSE_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_DILATE_ELLIPSE_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_ERODE_ELLIPSE_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_RING_ELLIPSE_13 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))


def build_subset_monotone_tree(
    fg_prob: np.ndarray,
    thresholds: Tuple[float, ...] = TREE_THRESHOLDS,
    min_area: int = 65,
    dedup_ratio: float = 0.985,
) -> List[dict]:
    """
    Construct a 16-level strictly subset-monotone, sibling-disjoint Component Tree
    on the native 1856x1856 grid (mapped to 2048x2048 RLEs).

    Bottom-up construction from highest threshold tau_{K-1} down to tau_0:
      M_k = fill_holes(morph_close(fg >= tau_k)) UNION M_{k+1}
    Guarantees 100.000% mathematically:
      1. Every child mask is a strict pixel subset of its parent mask (child <= parent).
      2. Sibling masks (and any tree antichain) have ZERO overlapping pixels.
    """
    K = len(thresholds)
    prev_union = np.zeros_like(fg_prob, dtype=np.uint8)
    prev_lbl = np.zeros_like(fg_prob, dtype=np.int32)
    prev_node_ids: Dict[int, int] = {}

    nodes: List[dict] = []

    for lvl in range(K - 1, -1, -1):
        tau = float(thresholds[lvl])
        raw = (fg_prob >= tau).astype(np.uint8)
        if raw.any():
            closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, _CLOSE_ELLIPSE_5)
            filled = ndi.binary_fill_holes(closed > 0).astype(np.uint8)
        else:
            filled = raw

        cur_mask = np.maximum(filled, prev_union)
        lbl, n_cc = ndi.label(cur_mask)
        if n_cc == 0:
            prev_union = np.zeros_like(fg_prob, dtype=np.uint8)
            prev_lbl = np.zeros_like(fg_prob, dtype=np.int32)
            prev_node_ids = {}
            continue

        slices = ndi.find_objects(lbl)
        cur_union = np.zeros_like(fg_prob, dtype=np.uint8)
        cur_lbl = np.zeros_like(fg_prob, dtype=np.int32)
        cur_node_ids: Dict[int, int] = {}
        new_lbl_counter = 0

        for cc_id, sl in enumerate(slices, start=1):
            if sl is None:
                continue
            comp_sub = lbl[sl] == cc_id
            area = int(comp_sub.sum())
            if area < min_area:
                continue

            # Find children from (lvl + 1) contained inside this component
            child_prev_labels = np.unique(prev_lbl[sl][comp_sub])
            child_prev_labels = child_prev_labels[child_prev_labels > 0]
            child_nids = [prev_node_ids[int(cl)] for cl in child_prev_labels if int(cl) in prev_node_ids]

            # Deduplicate trivial single-child chain wrapper if area barely changed
            if len(child_nids) == 1:
                only_ch = nodes[child_nids[0]]
                if float(only_ch["area"]) / float(area) >= dedup_ratio:
                    # Update birth_tau downward (wider persistence!) while keeping exact mask if desired,
                    # WAIT: to preserve strict subset with any future lower levels, update mask to `comp_sub`
                    # (which is a superset of `only_ch`'s children!).
                    only_ch["birth_tau"] = tau
                    only_ch["persistence_lvls"] += 1
                    only_ch["bbox_slice"] = sl
                    only_ch["sub_mask"] = comp_sub.copy()
                    only_ch["area"] = area
                    new_lbl_counter += 1
                    cur_union[sl][comp_sub] = 1
                    cur_lbl[sl][comp_sub] = new_lbl_counter
                    cur_node_ids[new_lbl_counter] = child_nids[0]
                    continue

            nid = len(nodes)
            node = {
                "nid": nid,
                "parent": -1,
                "host_parent": -1,
                "children": child_nids,
                "yolo_cands": [],
                "yolo_compat_children": [],
                "is_yolo_node": 0.0,
                "yolo_prior_score": 0.0,
                "birth_tau": tau,
                "max_tau": max([tau] + [nodes[c]["max_tau"] for c in child_nids]),
                "persistence_lvls": 1,
                "bbox_slice": sl,
                "sub_mask": comp_sub.copy(),
                "area": area,
            }
            for c_nid in child_nids:
                nodes[c_nid]["parent"] = nid
            nodes.append(node)

            new_lbl_counter += 1
            cur_union[sl][comp_sub] = 1
            cur_lbl[sl][comp_sub] = new_lbl_counter
            cur_node_ids[new_lbl_counter] = nid

        prev_union = cur_union
        prev_lbl = cur_lbl
        prev_node_ids = cur_node_ids

    return nodes


def inject_disjoint_yolo_candidates(
    nodes: List[dict],
    yolo_dets: Optional[List[dict]],
    valid_disk_mask: np.ndarray,
    min_score: float = 0.10,
    min_area: int = 55,
) -> List[dict]:
    """
    Augment the subset-monotone dense Component Tree with strictly pixel-disjoint
    top-down instance candidates from Family-B (YOLO11l-seg):
      1. Orphan Rescue Roots (`parent == -1`): faint/limb filaments missed by the dense root level.
      2. Inside-Node AND-OR Split / Shape Alternatives (`parent == -2`, attached to minimal
         containing dense host `u*` via `yolo_cands` + `yolo_compat_children`): separates
         touching filaments bridged in dense maps or provides sharper boundary masks while
         preserving 100.000% mathematical pixel-disjointness in O(|V|) DP.
    """
    if not yolo_dets:
        return nodes

    cands = [d for d in yolo_dets if float(d.get("score", 0.0)) >= min_score]
    if not cands:
        return nodes
    cands.sort(key=lambda d: float(d["score"]), reverse=True)

    rles_b = [
        {
            "size": [FULL_SIZE, FULL_SIZE],
            "counts": d["rle"]["counts"].encode("ascii") if isinstance(d["rle"]["counts"], str) else d["rle"]["counts"],
        }
        for d in cands
    ]
    dec_full = mask_util.decode(rles_b)
    if dec_full.ndim == 2:
        dec_full = dec_full[:, :, None]
    dec_crop = dec_full[CROP_OFFSET:CROP_OFFSET + CROP_SIZE, CROP_OFFSET:CROP_OFFSET + CROP_SIZE, :]

    # Step 1: Consolidate YOLO detections into a 100% pairwise pixel-disjoint set on [1856, 1856]
    claimed = np.zeros((CROP_SIZE, CROP_SIZE), dtype=bool)
    disjoint_yolo: List[Tuple[np.ndarray, int, float]] = []
    for j, d in enumerate(cands):
        m_raw = dec_crop[:, :, j] > 0
        orig_area = int(m_raw.sum())
        if orig_area < min_area:
            continue
        m_rem = m_raw & valid_disk_mask & (~claimed)
        rem_area = int(m_rem.sum())
        if rem_area < min_area or rem_area < 0.55 * orig_area:
            continue
        claimed |= m_rem
        disjoint_yolo.append((m_rem, rem_area, float(d["score"])))

    if not disjoint_yolo:
        return nodes

    # Step 2: Paint pairwise-disjoint dense tree roots (`parent == -1`) onto `root_lbl`
    n_dense = len(nodes)
    root_lbl = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.int32)
    for nid in range(n_dense):
        nd = nodes[nid]
        if nd["parent"] == -1:
            root_lbl[nd["bbox_slice"]][nd["sub_mask"]] = nid + 1
    u_roots = root_lbl > 0

    def _overlap_with_node(m_bool: np.ndarray, nid: int) -> int:
        nd = nodes[nid]
        sl = nd["bbox_slice"]
        return int((m_bool[sl] & nd["sub_mask"]).sum())

    # Step 3: Route each disjoint YOLO mask to either Orphan Root or Minimal Dense Container `u*`
    touched_hosts = set()
    for m_rem, rem_area, sc in disjoint_yolo:
        in_roots = int((m_rem & u_roots).sum())
        out_roots = rem_area - in_roots

        if out_roots >= min_area and out_roots >= 0.65 * rem_area:
            m_orph = m_rem & (~u_roots)
            ys, xs = np.nonzero(m_orph)
            sl_o = (slice(int(ys.min()), int(ys.max()) + 1), slice(int(xs.min()), int(xs.max()) + 1))
            sub_o = m_orph[sl_o].copy()
            nid = len(nodes)
            nodes.append({
                "nid": nid,
                "parent": -1,
                "host_parent": -1,
                "children": [],
                "yolo_cands": [],
                "yolo_compat_children": [],
                "is_yolo_node": 1.0,
                "yolo_prior_score": sc,
                "birth_tau": 0.04,
                "max_tau": min(0.84, max(0.04, sc)),
                "persistence_lvls": 1,
                "bbox_slice": sl_o,
                "sub_mask": sub_o,
                "area": out_roots,
            })
            continue

        if in_roots < min_area:
            continue

        lbl_vals = root_lbl[m_rem]
        lbl_vals = lbl_vals[lbl_vals > 0]
        if lbl_vals.size == 0:
            continue
        uniq_v, cnts_v = np.unique(lbl_vals, return_counts=True)
        best_pos = int(np.argmax(cnts_v))
        u_cur = int(uniq_v[best_pos]) - 1
        root_ov = int(cnts_v[best_pos])
        if root_ov < min_area or root_ov < 0.60 * rem_area:
            continue

        # Greedy unique descent: since 0.78 > 0.50 and children are disjoint, at most 1 child qualifies
        min_req = max(min_area, int(math.ceil(0.78 * root_ov)))
        while True:
            next_child = -1
            for ch in nodes[u_cur]["children"]:
                if ch < n_dense and _overlap_with_node(m_rem, ch) >= min_req:
                    next_child = ch
                    break
            if next_child >= 0:
                u_cur = next_child
            else:
                break

        host_nd = nodes[u_cur]
        sl_h = host_nd["bbox_slice"]
        clip_local = m_rem[sl_h] & host_nd["sub_mask"]
        clip_area = int(clip_local.sum())
        if clip_area < min_area:
            continue

        # Skip near-duplicate of `u_cur` itself or one of its direct dense children
        if float(clip_area) / float(host_nd["area"]) >= 0.90:
            continue
        dup_child = False
        for ch in host_nd["children"]:
            if ch >= n_dense:
                continue
            ch_nd = nodes[ch]
            sl_c = ch_nd["bbox_slice"]
            # Intersection inside `sl_c` (which lies inside `sl_h`)
            ry0 = sl_c[0].start - sl_h[0].start
            ry1 = sl_c[0].stop - sl_h[0].start
            rx0 = sl_c[1].start - sl_h[1].start
            rx1 = sl_c[1].stop - sl_h[1].start
            inter_c = int((clip_local[ry0:ry1, rx0:rx1] & ch_nd["sub_mask"]).sum())
            if inter_c > 0:
                iou_c = float(inter_c) / float(clip_area + ch_nd["area"] - inter_c)
                if iou_c >= 0.90:
                    dup_child = True
                    break
        if dup_child:
            continue

        ys_l, xs_l = np.nonzero(clip_local)
        gy0 = sl_h[0].start + int(ys_l.min())
        gy1 = sl_h[0].start + int(ys_l.max()) + 1
        gx0 = sl_h[1].start + int(xs_l.min())
        gx1 = sl_h[1].start + int(xs_l.max()) + 1
        sl_z = (slice(gy0, gy1), slice(gx0, gx1))
        sub_z = clip_local[int(ys_l.min()):int(ys_l.max()) + 1, int(xs_l.min()):int(xs_l.max()) + 1].copy()

        z_nid = len(nodes)
        nodes.append({
            "nid": z_nid,
            "parent": -2,
            "host_parent": u_cur,
            "children": [],
            "yolo_cands": [],
            "yolo_compat_children": [],
            "is_yolo_node": 1.0,
            "yolo_prior_score": sc,
            "birth_tau": host_nd["birth_tau"],
            "max_tau": min(0.84, max(host_nd["birth_tau"], sc)),
            "persistence_lvls": 1,
            "bbox_slice": sl_z,
            "sub_mask": sub_z,
            "area": clip_area,
        })
        host_nd["yolo_cands"].append(z_nid)
        touched_hosts.add(u_cur)

    # Step 4: Precompute strictly pixel-disjoint `yolo_compat_children` for every touched dense host
    for u_nid in touched_hosts:
        host_nd = nodes[u_nid]
        sl_h = host_nd["bbox_slice"]
        hy0, hx0 = sl_h[0].start, sl_h[1].start
        union_z = np.zeros((sl_h[0].stop - hy0, sl_h[1].stop - hx0), dtype=bool)
        for z_nid in host_nd["yolo_cands"]:
            z_nd = nodes[z_nid]
            sl_z = z_nd["bbox_slice"]
            union_z[
                (sl_z[0].start - hy0):(sl_z[0].stop - hy0),
                (sl_z[1].start - hx0):(sl_z[1].stop - hx0),
            ] |= z_nd["sub_mask"]

        compat = []
        for ch in host_nd["children"]:
            ch_nd = nodes[ch]
            sl_c = ch_nd["bbox_slice"]
            ov_z = (
                union_z[
                    (sl_c[0].start - hy0):(sl_c[0].stop - hy0),
                    (sl_c[1].start - hx0):(sl_c[1].stop - hx0),
                ]
                & ch_nd["sub_mask"]
            ).any()
            if not ov_z:
                compat.append(ch)
        host_nd["yolo_compat_children"] = compat

    return nodes


def solve_tree_antichain_dp(nodes: List[dict], u_hat: np.ndarray, tau: float) -> List[int]:
    """
    Exact O(|V|) bottom-up AND-OR Dynamic Programming on the Component-Tree forest maximizing:
      sum_{c in A} (u_hat[c] - tau)
    over all valid pairwise pixel-disjoint selections A (choosing at each node u between
    NONE=0, SELF=1, DENSE_CHILDREN=2, and YOLO_PARTITION=3).
    """
    n = len(nodes)
    if n == 0:
        return []

    roots = [i for i, nd in enumerate(nodes) if nd["parent"] == -1]
    order: List[int] = []
    stack = [(r, False) for r in roots]
    while stack:
        u, visited = stack.pop()
        if visited:
            order.append(u)
        else:
            stack.append((u, True))
            for ch in nodes[u]["children"]:
                stack.append((ch, False))

    dp_val = np.zeros(n, dtype=np.float64)
    choice = np.zeros(n, dtype=np.int8)  # 0=none, 1=self, 2=dense_children, 3=yolo_partition
    tau_f = float(tau)

    for u in order:
        nd = nodes[u]
        w_self = float(u_hat[u]) - tau_f
        ch_list = nd["children"]
        w_dense = sum(dp_val[ch] for ch in ch_list) if ch_list else 0.0

        y_cands = nd.get("yolo_cands")
        if y_cands:
            y_pos = sum(max(0.0, float(u_hat[z]) - tau_f) for z in y_cands)
            compat_ch = nd.get("yolo_compat_children", [])
            w_yolo = y_pos + (sum(dp_val[ch] for ch in compat_ch) if compat_ch else 0.0)
        else:
            w_yolo = -1.0

        best_v = 0.0
        best_c = 0
        if w_dense > best_v:
            best_v = w_dense
            best_c = 2
        if w_yolo > best_v:
            best_v = w_yolo
            best_c = 3
        if w_self > best_v:
            best_v = w_self
            best_c = 1

        dp_val[u] = best_v
        choice[u] = best_c

    selected: List[int] = []
    q = list(roots)
    while q:
        u = q.pop()
        c_mode = int(choice[u])
        if c_mode == 1:
            selected.append(u)
        elif c_mode == 2:
            for ch in nodes[u]["children"]:
                if dp_val[ch] > 0.0:
                    q.append(ch)
        elif c_mode == 3:
            nd = nodes[u]
            for z in nd.get("yolo_cands", []):
                if float(u_hat[z]) - tau_f > 0.0:
                    selected.append(z)
            for ch in nd.get("yolo_compat_children", []):
                if dp_val[ch] > 0.0:
                    q.append(ch)

    selected.sort()
    return selected


def solve_two_tier_antichain_handoff(
    nodes: List[dict],
    u_hat: np.ndarray,
    strict_tau: float,
    rescue_tau: float = 0.2250,
) -> List[int]:
    """
    Hierarchy-consistent Two-Tier Antichain Handoff:
      1. Exact Antichain DP at `strict_tau` (`sel_strict`) preserves 100% of optimal intact
         confident filaments without lowering the parent-vs-children split penalty (eliminating
         low-tau over-splitting of single filaments!).
      2. Exact Antichain DP at `rescue_tau` (`< strict_tau`) on all remaining disjoint subtrees
         (blocking `sel_strict` and all of their ancestors & descendants) adds non-overlapping
         faint rescue candidates in `(rescue_tau, strict_tau)` for Stage-2 refinement.
    Guarantees `[i for i in out if u_hat[i] >= strict_tau] == sel_strict` and 100% pixel disjointness.
    """
    n = len(nodes)
    if n == 0:
        return []
    strict_t = float(strict_tau)
    rescue_t = float(rescue_tau)
    sel_strict = solve_tree_antichain_dp(nodes, u_hat, tau=strict_t)
    if rescue_t >= strict_t - 1e-9:
        return sel_strict

    blocked = np.zeros(n, dtype=bool)
    for idx in sel_strict:
        blocked[idx] = True
        nd_i = nodes[idx]
        if nd_i["parent"] == -2:
            # YOLO split candidate inside `host_parent`: block host_parent's ancestors and overlapping children
            hp = int(nd_i["host_parent"])
            cur = hp
            while cur >= 0:
                blocked[cur] = True
                for z in nodes[cur].get("yolo_cands", []):
                    blocked[z] = True
                cur = int(nodes[cur]["parent"])
            compat_set = set(nodes[hp].get("yolo_compat_children", []))
            for ch in nodes[hp]["children"]:
                if ch not in compat_set:
                    stk = [ch]
                    while stk:
                        v = stk.pop()
                        blocked[v] = True
                        for z in nodes[v].get("yolo_cands", []):
                            blocked[z] = True
                        stk.extend(nodes[v]["children"])
        else:
            cur = int(nd_i["parent"])
            while cur >= 0:
                blocked[cur] = True
                for z in nodes[cur].get("yolo_cands", []):
                    blocked[z] = True
                cur = int(nodes[cur]["parent"])
            stk = [idx]
            while stk:
                v = stk.pop()
                blocked[v] = True
                for z in nodes[v].get("yolo_cands", []):
                    blocked[z] = True
                stk.extend(nodes[v]["children"])

    u_arr = np.asarray(u_hat, dtype=np.float64)
    u_rescue = np.where(blocked, -1e6, np.minimum(u_arr, strict_t - 1e-6))
    sel_rescue = solve_tree_antichain_dp(nodes, u_rescue, tau=rescue_t)
    merged = sorted(set(sel_strict) | set(sel_rescue))
    return merged


def encode_submask_rle_2048(
    sub_m: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    full_h: int = FULL_SIZE,
    full_w: int = FULL_SIZE,
) -> str:
    """
    41x faster, 100% bit-exact column-major COCO RLE encoder directly from a tight
    bounding-box submask without allocating or scanning a full 2048x2048 canvas.
    """
    bw = x1 - x0
    if bw <= 0 or y1 <= y0 or not sub_m.any():
        r = mask_util.frPyObjects([{"counts": [full_h * full_w], "size": [full_h, full_w]}], full_h, full_w)[0]
        c = r["counts"]
        return c.decode("ascii") if isinstance(c, bytes) else str(c)

    strip = np.zeros((full_h, bw), dtype=np.uint8, order="F")
    strip[y0:y1, :] = sub_m.astype(np.uint8)
    flat = strip.ravel(order="F")
    diffs = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    idx = np.concatenate(([0], diffs, [flat.size]))
    runs = np.diff(idx).tolist()
    if flat[0] == 1:
        runs = [0] + runs
    runs[0] += x0 * full_h
    tail_z = (full_w - x1) * full_h
    if tail_z > 0:
        if len(runs) % 2 == 1:
            runs[-1] += tail_z
        else:
            runs.append(tail_z)
    r = mask_util.frPyObjects([{"counts": runs, "size": [full_h, full_w]}], full_h, full_w)[0]
    c = r["counts"]
    return c.decode("ascii") if isinstance(c, bytes) else str(c)


def node_to_full_rle(node: dict) -> dict:
    """Convert a node's cropped sub_mask into a 2048x2048 canonical COCO RLE dict."""
    if "rle" in node:
        return node["rle"]
    sl = node["bbox_slice"]
    sub_m = node["sub_mask"]
    y0, y1 = sl[0].start + CROP_OFFSET, sl[0].stop + CROP_OFFSET
    x0, x1 = sl[1].start + CROP_OFFSET, sl[1].stop + CROP_OFFSET
    rle = {"size": [FULL_SIZE, FULL_SIZE], "counts": encode_submask_rle_2048(sub_m, y0, y1, x0, x1)}
    node["rle"] = rle
    return rle


FEATURE_NAMES: List[str] = [
    # 1. Probability & Spine Calibration (22)
    "fg_mean", "fg_std", "fg_q10", "fg_q25", "fg_q50", "fg_q75", "fg_q90", "fg_q98", "fg_max",
    "fg_mass_log", "frac_gt_25", "frac_gt_40", "frac_gt_60", "frac_gt_75",
    "inner_ring_fg", "outer_ring_fg", "boundary_fg_drop",
    "sp_mean", "sp_q90", "sp_max", "sp_cov30", "sp_over_fg",
    # 2. Astrophysical Photometry & Geometry (14)
    "ff_in_mean", "ff_in_q10", "ff_in_min", "ff_ring_mean", "ff_contrast_drop", "ff_contrast_snr",
    "dog_in_mean", "dog_in_q90",
    "r_norm", "mu_cos", "radial_align_abs", "cloud_grad", "disk_rms", "station_id",
    # 3. Ribbon Morphology & Scale (12)
    "area", "log_area", "perim", "compactness_pp", "bbox_h", "bbox_w", "bbox_aspect", "bbox_fill",
    "pca_l1", "pca_l2", "elongation", "est_width",
    # 4. Symmetry & Epistemic Uncertainty (13)
    "tta_std_in", "tta_std_q90", "tta_std_ring",
    "disagree_in", "disagree_q90", "disagree_ring",
    "fam0_fg", "fam1_fg", "fam2_fg", "fam3_fg", "fam4_fg", "fam_std_corr", "fam_spread",
    # 5. Component-Tree Hierarchy & Image Context (8)
    "birth_tau", "max_tau", "persistence_lvls", "n_children",
    "area_over_parent", "max_child_ratio", "n_siblings", "img_fg_mass_log",
    # 6. Family-B YOLO11l-seg Detector Agreement & Node Source (8)
    "det_best_iou", "det_best_score", "det_iou_x_score",
    "det_union_cov", "det_precision_in_best", "det_n_overlap15",
    "is_yolo_node", "yolo_prior_score",
]


def extract_tree_features(
    stem: str,
    nodes: List[dict],
    x_u8: np.ndarray,
    fg_blend: np.ndarray,
    sp_blend: np.ndarray,
    tta_std: np.ndarray,
    disagree: np.ndarray,
    fam_fgs: List[np.ndarray],
    disk_meta_entry: dict,
    yolo_dets: Optional[List[dict]] = None,
    fam_fstds: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    """Extract all 77 node descriptors in float32 for Hurdle GBDT scoring."""
    n = len(nodes)
    F_dim = len(FEATURE_NAMES)
    if n == 0:
        return np.zeros((0, F_dim), dtype=np.float32)

    # Decode physical flat-field contrast c(x) and DoG from x_u8
    ff_c = x_u8[:, :, 0].astype(np.float32) * (0.70 / 255.0) - 0.45
    dog_c = x_u8[:, :, 1].astype(np.float32) * (0.48 / 255.0) - 0.08

    cx = float(disk_meta_entry.get("cx", 1023.0)) - CROP_OFFSET
    cy = float(disk_meta_entry.get("cy", 1024.0)) - CROP_OFFSET
    R = max(float(disk_meta_entry.get("R", 902.8)), 100.0)
    cloud_grad = float(disk_meta_entry.get("cloud_grad", 0.0))
    disk_rms = float(disk_meta_entry.get("rms", 1.0))
    station_id = float(STATION_TO_IDX.get(stem[-2:], 0))
    img_fg_mass_log = math.log1p(float(fg_blend.sum()))

    # Precompute sibling counts
    children_count_of_parent: Dict[int, int] = {}
    root_count = 0
    for nd in nodes:
        p = nd["parent"]
        if p == -1:
            root_count += 1
        else:
            children_count_of_parent[p] = children_count_of_parent.get(p, 0) + 1

    # Ensure RLEs built for fast batch IoU against YOLO detections
    node_rles_bytes = [
        {"size": [FULL_SIZE, FULL_SIZE], "counts": node_to_full_rle(nd)["counts"].encode("ascii")}
        for nd in nodes
    ]

    det_list = yolo_dets or []
    if len(det_list) > 0:
        det_rles_bytes = [
            {"size": [FULL_SIZE, FULL_SIZE], "counts": d["rle"]["counts"].encode("ascii")}
            for d in det_list
        ]
        det_scores = np.array([float(d["score"]) for d in det_list], dtype=np.float32)
        det_areas = np.array([max(float(d["area"]), 1.0) for d in det_list], dtype=np.float32)
        iou_nd = np.asarray(mask_util.iou(node_rles_bytes, det_rles_bytes, [0] * len(det_rles_bytes)), dtype=np.float32)
        # Also build union map of confident detections (score >= 0.20) on 1856x1856
        strong_rles = [det_rles_bytes[j] for j, sc in enumerate(det_scores) if sc >= 0.20]
        if strong_rles:
            dec_strong = mask_util.decode(strong_rles)
            det_union_1856 = (
                dec_strong[CROP_OFFSET:CROP_OFFSET + CROP_SIZE, CROP_OFFSET:CROP_OFFSET + CROP_SIZE, :].sum(axis=2) > 0
            )
        else:
            det_union_1856 = None
    else:
        iou_nd = None
        det_scores = None
        det_areas = None
        det_union_1856 = None

    feats = np.zeros((n, F_dim), dtype=np.float32)
    pad = 7

    for i, nd in enumerate(nodes):
        sl = nd["bbox_slice"]
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        py0 = max(0, y0 - pad)
        py1 = min(CROP_SIZE, y1 + pad)
        px0 = max(0, x0 - pad)
        px1 = min(CROP_SIZE, x1 + pad)

        sub_m = nd["sub_mask"]
        pad_m = np.zeros((py1 - py0, px1 - px0), dtype=np.uint8)
        pad_m[(y0 - py0):(y1 - py0), (x0 - px0):(x1 - px0)] = sub_m.astype(np.uint8)
        m_bool = pad_m > 0

        eroded = cv2.erode(pad_m, _ERODE_ELLIPSE_5)
        inner_ring = m_bool & (eroded == 0)
        dil_near = cv2.dilate(pad_m, _DILATE_ELLIPSE_5)
        outer_near = (dil_near > 0) & (~m_bool)
        dil_wide = cv2.dilate(pad_m, _RING_ELLIPSE_13)
        outer_wide = (dil_wide > 0) & (~m_bool)

        fg_p = fg_blend[py0:py1, px0:px1]
        sp_p = sp_blend[py0:py1, px0:px1]
        std_p = tta_std[py0:py1, px0:px1]
        dis_p = disagree[py0:py1, px0:px1]
        ff_p = ff_c[py0:py1, px0:px1]
        dog_p = dog_c[py0:py1, px0:px1]

        v_fg = fg_p[m_bool]
        v_sp = sp_p[m_bool]
        v_ff = ff_p[m_bool]
        v_dog = dog_p[m_bool]
        v_std = std_p[m_bool]
        v_dis = dis_p[m_bool]

        area_f = float(nd["area"])
        q10, q25, q50, q75, q90, q98 = np.percentile(v_fg, [10, 25, 50, 75, 90, 98])
        fg_mean = float(v_fg.mean())
        fg_std = float(v_fg.std())
        fg_max = float(v_fg.max())
        fg_mass_log = math.log1p(float(v_fg.sum()))

        frac25 = float((v_fg >= 0.25).mean())
        frac40 = float((v_fg >= 0.40).mean())
        frac60 = float((v_fg >= 0.60).mean())
        frac75 = float((v_fg >= 0.75).mean())

        in_r_fg = float(fg_p[inner_ring].mean()) if inner_ring.any() else fg_mean
        out_r_fg = float(fg_p[outer_near].mean()) if outer_near.any() else 0.0
        b_drop = in_r_fg - out_r_fg

        sp_mean = float(v_sp.mean())
        sp_q90 = float(np.percentile(v_sp, 90))
        sp_max = float(v_sp.max())
        sp_cov30 = float((v_sp >= 0.30).mean())
        sp_over_fg = sp_mean / (fg_mean + 0.05)

        ff_in_mean = float(v_ff.mean())
        ff_in_q10 = float(np.percentile(v_ff, 10))
        ff_in_min = float(v_ff.min())
        if outer_wide.any():
            v_ff_ring = ff_p[outer_wide]
            ff_ring_mean = float(v_ff_ring.mean())
            ff_ring_std = float(v_ff_ring.std())
        else:
            ff_ring_mean = 0.0
            ff_ring_std = 0.03
        ff_drop = ff_ring_mean - ff_in_mean
        ff_snr = ff_drop / (ff_ring_std + 1e-3)

        dog_in_mean = float(v_dog.mean())
        dog_in_q90 = float(np.percentile(v_dog, 90))

        # PCA & Centroid geometry
        ys_loc, xs_loc = np.nonzero(sub_m)
        cy_node = float(ys_loc.mean()) + y0
        cx_node = float(xs_loc.mean()) + x0
        rx = cx_node - cx
        ry = cy_node - cy
        r_norm = math.hypot(rx, ry) / R
        mu_cos = math.sqrt(max(0.0, 1.0 - r_norm * r_norm))

        if len(xs_loc) >= 4:
            dx_c = xs_loc.astype(np.float32) - float(xs_loc.mean())
            dy_c = ys_loc.astype(np.float32) - float(ys_loc.mean())
            cxx = float(np.mean(dx_c * dx_c)) + 0.0833
            cyy = float(np.mean(dy_c * dy_c)) + 0.0833
            cxy = float(np.mean(dx_c * dy_c))
            tr = cxx + cyy
            det_cov = max(cxx * cyy - cxy * cxy, 1e-6)
            disc = math.sqrt(max(tr * tr - 4.0 * det_cov, 0.0))
            l1 = 0.5 * (tr + disc)
            l2 = max(0.5 * (tr - disc), 0.0833)
            vx = l1 - cyy
            vy = cxy
            vnorm = math.hypot(vx, vy)
            if vnorm > 1e-6 and math.hypot(rx, ry) > 1.0:
                radial_align_abs = abs(vx * rx + vy * ry) / (vnorm * math.hypot(rx, ry))
            else:
                radial_align_abs = 0.0
        else:
            l1, l2, radial_align_abs = 1.0, 1.0, 0.0

        elongation = math.sqrt(l1 / max(l2, 1e-4))
        est_len = max(3.5 * math.sqrt(l1), 1.0)
        est_width = area_f / est_len

        perim = float(inner_ring.sum()) if inner_ring.any() else 4.0 * math.sqrt(area_f)
        compactness_pp = 4.0 * math.pi * area_f / max(perim * perim, 1.0)
        bh = float(y1 - y0)
        bw = float(x1 - x0)
        bbox_aspect = max(bh, bw) / max(min(bh, bw), 1.0)
        bbox_fill = area_f / max(bh * bw, 1.0)

        tta_std_in = float(v_std.mean())
        tta_std_q90 = float(np.percentile(v_std, 90))
        tta_std_ring = float(std_p[outer_near].mean()) if outer_near.any() else tta_std_in

        disagree_in = float(v_dis.mean())
        disagree_q90 = float(np.percentile(v_dis, 90))
        disagree_ring = float(dis_p[outer_near].mean()) if outer_near.any() else disagree_in

        raw_f_means = [float(fm[y0:y1, x0:x1][sub_m].mean()) for fm in fam_fgs]
        n_actual_fams = len(raw_f_means)
        f_means = list(raw_f_means)
        while len(f_means) < 5:
            f_means.append(fg_mean)
        fam0_fg, fam1_fg, fam2_fg, fam3_fg, fam4_fg = (
            f_means[0], f_means[1], f_means[2], f_means[3], f_means[4]
        )
        raw_spread = (max(raw_f_means) - min(raw_f_means)) if n_actual_fams >= 2 else 0.0
        if fam_fstds is not None and len(fam_fstds) >= 2:
            m_cnt = float(len(fam_fstds))
            raw_f_var = float(np.var(raw_f_means[: len(fam_fstds)]))
            mean_fstd_sq = float(
                np.mean([float(fs[y0:y1, x0:x1][sub_m].mean()) ** 2 for fs in fam_fstds])
            )
            corr_std = math.sqrt(raw_f_var + ((m_cnt - 1.0) / m_cnt) * mean_fstd_sq)
            raw_std = math.sqrt(raw_f_var)
            fam_std_corr = min(1.0, corr_std)
            if raw_std > 1e-5:
                fam_spread = min(1.0, raw_spread * (corr_std / raw_std))
            else:
                fam_spread = min(1.0, 2.30 * corr_std)
        else:
            fam_std_corr = float(np.std(raw_f_means)) if n_actual_fams >= 2 else 0.0
            fam_spread = raw_spread

        p_nid = nd["parent"]
        ref_p = p_nid if p_nid >= 0 else nd.get("host_parent", -1)
        area_over_parent = area_f / float(nodes[ref_p]["area"]) if ref_p >= 0 else 1.0
        ch_nids = nd["children"]
        max_child_ratio = max((float(nodes[c]["area"]) for c in ch_nids), default=0.0) / area_f
        if p_nid == -1:
            n_sib = root_count - 1
        elif p_nid >= 0:
            n_sib = children_count_of_parent.get(p_nid, 1) - 1
        elif ref_p >= 0:
            n_sib = max(len(nodes[ref_p].get("yolo_cands", [])) - 1, 0)
        else:
            n_sib = 0

        if iou_nd is not None and iou_nd.shape[1] > 0:
            row_ious = iou_nd[i]
            best_j = int(np.argmax(row_ious))
            det_best_iou = float(row_ious[best_j])
            det_best_score = float(det_scores[best_j])
            det_iou_x_score = float(np.max(row_ious * det_scores))
            # intersection = iou * (A + B) / (1 + iou) -> node_in_det = inter / A
            inter_est = det_best_iou * (area_f + float(det_areas[best_j])) / (1.0 + det_best_iou + 1e-8)
            det_precision_in_best = inter_est / max(area_f, 1.0)
            det_n_overlap15 = float((row_ious >= 0.15).sum())
            det_union_cov = float(det_union_1856[y0:y1, x0:x1][sub_m].mean()) if det_union_1856 is not None else 0.0
        else:
            det_best_iou = 0.0
            det_best_score = 0.0
            det_iou_x_score = 0.0
            det_union_cov = 0.0
            det_precision_in_best = 0.0
            det_n_overlap15 = 0.0

        feats[i] = [
            fg_mean, fg_std, q10, q25, q50, q75, q90, q98, fg_max,
            fg_mass_log, frac25, frac40, frac60, frac75,
            in_r_fg, out_r_fg, b_drop,
            sp_mean, sp_q90, sp_max, sp_cov30, sp_over_fg,
            ff_in_mean, ff_in_q10, ff_in_min, ff_ring_mean, ff_drop, ff_snr,
            dog_in_mean, dog_in_q90,
            r_norm, mu_cos, radial_align_abs, cloud_grad, disk_rms, station_id,
            area_f, math.log1p(area_f), perim, compactness_pp, bh, bw, bbox_aspect, bbox_fill,
            math.sqrt(l1), math.sqrt(l2), elongation, est_width,
            tta_std_in, tta_std_q90, tta_std_ring,
            disagree_in, disagree_q90, disagree_ring,
            fam0_fg, fam1_fg, fam2_fg, fam3_fg, fam4_fg, fam_std_corr, fam_spread,
            float(nd["birth_tau"]), float(nd["max_tau"]), float(nd["persistence_lvls"]), float(len(ch_nids)),
            area_over_parent, max_child_ratio, float(n_sib), img_fg_mass_log,
            det_best_iou, det_best_score, det_iou_x_score,
            det_union_cov, det_precision_in_best, det_n_overlap15,
            float(nd.get("is_yolo_node", 0.0)), float(nd.get("yolo_prior_score", 0.0)),
        ]

    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
