import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.hierarchy import (
    FEATURE_NAMES,
    build_subset_monotone_tree,
    encode_submask_rle_2048,
    extract_tree_features,
    inject_disjoint_yolo_candidates,
    node_to_full_rle,
    solve_tree_antichain_dp,
    solve_two_tier_antichain_handoff,
)
from src.metrics import encode_binary_mask


def test_tree_subset_monotonicity_and_exact_dp():
    rng = np.random.default_rng(123)
    # Create synthetic probability landscape with bridged twins, nested peaks, and internal holes
    yy, xx = np.meshgrid(np.arange(256, dtype=np.float32), np.arange(256, dtype=np.float32), indexing="ij")
    p = (
        0.88 * np.exp(-((xx - 80) ** 2 + (yy - 100) ** 2) / (2 * 9.0 ** 2))
        + 0.82 * np.exp(-((xx - 115) ** 2 + (yy - 100) ** 2) / (2 * 9.0 ** 2))
        + 0.65 * np.exp(-((xx - 180) ** 2 / (2 * 6.0 ** 2) + (yy - 160) ** 2 / (2 * 28.0 ** 2)))
    ).astype(np.float32)

    nodes = build_subset_monotone_tree(p, min_area=20)
    assert len(nodes) >= 4

    # 1. Verify EVERY child is a strict pixel subset of its parent
    full_masks = []
    for nd in nodes:
        m = np.zeros((256, 256), dtype=bool)
        m[nd["bbox_slice"]] = nd["sub_mask"]
        full_masks.append(m)

    for i, nd in enumerate(nodes):
        p_id = nd["parent"]
        if p_id >= 0:
            diff = int((full_masks[i] & (~full_masks[p_id])).sum())
            assert diff == 0, f"Child {i} has {diff} pixels outside parent {p_id}!"
        # Verify children of same parent are 100% pairwise disjoint
        chs = nd["children"]
        for a in range(len(chs)):
            for b in range(a + 1, len(chs)):
                ov = int((full_masks[chs[a]] & full_masks[chs[b]]).sum())
                assert ov == 0, f"Siblings {chs[a]} and {chs[b]} overlap on {ov} pixels!"

    # 2. Brute-force verify exact Antichain DP optimality on a 5-level tree (N <= 16 so 2^N is exact & instant)
    nodes_5 = build_subset_monotone_tree(p, thresholds=(0.12, 0.28, 0.46, 0.65, 0.82), min_area=20)
    masks_5 = []
    for nd in nodes_5:
        m = np.zeros((256, 256), dtype=bool)
        m[nd["bbox_slice"]] = nd["sub_mask"]
        masks_5.append(m)
    N5 = len(nodes_5)
    assert 4 <= N5 <= 16, f"Unexpected N5={N5}"
    for trial in range(30):
        u_hat = rng.uniform(0.0, 0.9, size=N5)
        tau = float(rng.uniform(0.15, 0.35))
        dp_sel = solve_tree_antichain_dp(nodes_5, u_hat, tau)

        # Verify DP selection is 100% pixel-disjoint
        cov = np.zeros((256, 256), dtype=np.int32)
        for idx in dp_sel:
            cov += masks_5[idx].astype(np.int32)
        assert int(cov.max()) <= 1, "Antichain DP selected overlapping nodes!"
        dp_score = sum(float(u_hat[idx]) - tau for idx in dp_sel)

        # Compare against brute-force enumeration over all disjoint subsets
        best_bf = 0.0
        for mask_bits in range(1 << N5):
            chosen = [j for j in range(N5) if (mask_bits >> j) & 1]
            valid = True
            for a in range(len(chosen)):
                for b in range(a + 1, len(chosen)):
                    if (masks_5[chosen[a]] & masks_5[chosen[b]]).any():
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                sc = sum(float(u_hat[j]) - tau for j in chosen)
                if sc > best_bf:
                    best_bf = sc
        assert abs(dp_score - best_bf) < 1e-9, f"DP={dp_score} != BruteForce={best_bf}"

    print(f"[OK] Subset monotonicity (16-lvl N={len(nodes)}), sibling disjointness, and exact O(|V|) Antichain DP (N5={N5}, 30 trials) verified!")


def test_feature_extractor_dimensions():
    from src.config import CROP_SIZE
    p = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
    p[400:435, 600:740] = 0.72
    p[410:425, 620:720] = 0.88
    nodes = build_subset_monotone_tree(p, min_area=50)
    x_u8 = np.full((CROP_SIZE, CROP_SIZE, 3), 140, dtype=np.uint8)
    sp = p * 0.8
    std = np.full_like(p, 0.02)
    dis = np.full_like(p, 0.03)
    feats = extract_tree_features("20150101120000Lh", nodes, x_u8, p, sp, std, dis, [p, p * 0.95, p * 1.02], {"cx": 1023.0, "cy": 1024.0, "R": 902.8})
    assert len(FEATURE_NAMES) == 77, f"Expected 77 features, got {len(FEATURE_NAMES)}"
    assert feats.shape == (len(nodes), len(FEATURE_NAMES))
    assert np.isfinite(feats).all()

    # Verify 5-family feature extraction + Law-of-Total-Variance fam_std_corr
    fam5 = [p * 0.92, p * 0.96, p * 1.00, p * 1.03, p * 1.06]
    fstds5 = [np.full_like(p, 0.05) for _ in range(5)]
    feats5_raw = extract_tree_features("20150101120000Lh", nodes, x_u8, p, sp, std, dis, fam5, {"cx": 1023.0, "cy": 1024.0, "R": 902.8})
    feats5_corr = extract_tree_features("20150101120000Lh", nodes, x_u8, p, sp, std, dis, fam5, {"cx": 1023.0, "cy": 1024.0, "R": 902.8}, fam_fstds=fstds5)
    assert feats5_corr.shape == (len(nodes), 77) and np.isfinite(feats5_corr).all()
    idx_f3 = FEATURE_NAMES.index("fam3_fg")
    idx_f4 = FEATURE_NAMES.index("fam4_fg")
    idx_sc = FEATURE_NAMES.index("fam_std_corr")
    assert float(feats5_corr[0, idx_f4]) > float(feats5_corr[0, idx_f3]) > 0.5
    assert float(feats5_corr[0, idx_sc]) > float(feats5_raw[0, idx_sc]), "Expected fam_std_corr to increase when 5-family fstds > 0!"
    print(f"[OK] Feature extractor verified: 3-fam & 5-fam shape={feats5_corr.shape} ({len(FEATURE_NAMES)} features), zero NaNs.")


def test_and_or_yolo_candidates_and_fast_rle():
    from src.config import CROP_OFFSET, CROP_SIZE, FULL_SIZE

    rng = np.random.default_rng(999)
    # 1. Verify bit-exact RLE encoding on corners and random crops
    for y0, y1, x0, x1, val in [
        (0, 1, 0, 1, 1),
        (2047, 2048, 2047, 2048, 1),
        (200, 260, 300, 390, -1),
    ]:
        sub = np.ones((y1 - y0, x1 - x0), dtype=np.uint8) if val == 1 else rng.integers(0, 2, size=(y1 - y0, x1 - x0), dtype=np.uint8)
        full = np.zeros((FULL_SIZE, FULL_SIZE), dtype=np.uint8)
        full[y0:y1, x0:x1] = sub
        assert encode_binary_mask(full) == encode_submask_rle_2048(sub, y0, y1, x0, x1)

    # 2. Build synthetic 1856x1856 landscape with:
    #    - One wide merged dense blob [300:340, 400:560] that YOLO splits into 2 clean ribbons + 1 untouched dense lobe
    #    - One faint orphan filament [900:920, 900:960] missed by dense map (`fg=0.01`) but detected by YOLO
    p = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
    p[300:340, 400:560] = 0.35
    p[305:335, 520:555] = 0.80  # 3rd lobe captured as dense child

    def _box_rle(y0, y1, x0, x1):
        sub = np.ones((y1 - y0, x1 - x0), dtype=np.uint8)
        return {"size": [FULL_SIZE, FULL_SIZE], "counts": encode_submask_rle_2048(sub, y0 + CROP_OFFSET, y1 + CROP_OFFSET, x0 + CROP_OFFSET, x1 + CROP_OFFSET)}

    yolo_dets = [
        {"score": 0.78, "area": 1200, "rle": _box_rle(304, 336, 404, 448)},  # Split #1 inside merged blob
        {"score": 0.71, "area": 1200, "rle": _box_rle(304, 336, 456, 500)},  # Split #2 inside merged blob (disjoint from #1 and from lobe [520:555]!)
        {"score": 0.55, "area": 1100, "rle": _box_rle(900, 920, 900, 960)},  # Orphan rescue root outside dense tree!
    ]
    valid_disk = np.ones((CROP_SIZE, CROP_SIZE), dtype=bool)
    nodes = build_subset_monotone_tree(p, min_area=55)
    n_before = len(nodes)
    nodes = inject_disjoint_yolo_candidates(nodes, yolo_dets, valid_disk, min_score=0.10, min_area=55)
    assert len(nodes) == n_before + 3, f"Expected +3 YOLO cands (2 splits + 1 orphan), got {n_before} -> {len(nodes)}"

    # Verify across 40 random utility vectors that DP ALWAYS returns 100% pixel-disjoint masks!
    masks = []
    for nd in nodes:
        m = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.int32)
        m[nd["bbox_slice"]][nd["sub_mask"]] = 1
        masks.append(m)

    for _ in range(40):
        u_hat = rng.uniform(0.0, 0.95, size=len(nodes))
        tau = float(rng.uniform(0.18, 0.34))
        sel = solve_tree_antichain_dp(nodes, u_hat, tau)
        if sel:
            cov = sum(masks[idx] for idx in sel)
            assert int(cov.max()) <= 1, f"AND-OR DP produced pixel overlap! max={cov.max()}"

    # Specifically verify that when YOLO splits + compatible dense lobe + orphan root have high u_hat, ALL 4 are selected simultaneously!
    u_special = np.full(len(nodes), 0.10, dtype=np.float64)
    for idx, nd in enumerate(nodes):
        if nd.get("is_yolo_node", 0.0) > 0.5 or nd["max_tau"] >= 0.75:
            u_special[idx] = 0.85
    sel_sp = solve_tree_antichain_dp(nodes, u_special, tau=0.25)
    assert len(sel_sp) == 4, f"Expected 4 disjoint pieces (2 YOLO splits + 1 compat dense lobe + 1 orphan root), got {len(sel_sp)}"
    cov_sp = sum(masks[idx] for idx in sel_sp)
    assert int(cov_sp.max()) == 1

    x_u8 = np.full((CROP_SIZE, CROP_SIZE, 3), 140, dtype=np.uint8)
    feats = extract_tree_features(
        "20150101120000Lh", nodes, x_u8, p, p * 0.8,
        np.full_like(p, 0.02), np.full_like(p, 0.02), [p, p, p],
        {"cx": 1023.0, "cy": 1024.0, "R": 902.8}, yolo_dets=yolo_dets,
    )
    assert feats.shape == (len(nodes), len(FEATURE_NAMES)) and np.isfinite(feats).all()
    print("[OK] AND-OR Dense+YOLO candidate injection, 4-piece hybrid split+orphan selection, and 100% pixel disjointness verified!")


def test_law_of_total_variance_fstd_parity():
    from src.config import CROP_SIZE

    rng = np.random.default_rng(2026)
    # Verify across 4,000 Monte Carlo trials (M=3 families, K=5 folds) that Var_corr exactly matches 1-fold OOF Var!
    M, K, N_mc = 3, 5, 4000
    mu = rng.uniform(0.25, 0.75, size=(N_mc, M, 1))
    sig = rng.uniform(0.04, 0.14, size=(N_mc, M, 1))
    folds = np.clip(mu + rng.normal(0.0, sig, size=(N_mc, M, K)), 0.0, 1.0)

    oof_1fold = folds[:, :, 0]  # 1 held-out fold per family
    oof_var = np.var(oof_1fold, axis=1)
    oof_spread = oof_1fold.max(axis=1) - oof_1fold.min(axis=1)

    p_bar = folds.mean(axis=2)  # K-fold average on Test
    fstd = folds.std(axis=2)    # K-fold ddof=0 sample std on Test
    raw_test_var = np.var(p_bar, axis=1)
    corr_test_var = raw_test_var + ((M - 1.0) / M) * np.mean(fstd ** 2, axis=1)

    # Without correction, raw_test_var shrinks by ~30%; with correction, matches within <1.5% MC error!
    rel_err = abs(float(corr_test_var.mean() - oof_var.mean())) / float(oof_var.mean())
    assert rel_err < 0.02, f"Law-of-Total-Variance mismatch: corr={corr_test_var.mean():.6f} vs oof={oof_var.mean():.6f}"

    # Also check extract_tree_features with fam_fstds passed
    p = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
    p[500:530, 700:780] = 0.70
    nodes = build_subset_monotone_tree(p, min_area=55)
    x_u8 = np.full((CROP_SIZE, CROP_SIZE, 3), 135, dtype=np.uint8)
    f0 = extract_tree_features("20150101120000Lh", nodes, x_u8, p, p, np.zeros_like(p), np.zeros_like(p), [p * 0.96, p, p * 1.04], {"cx": 1023.0, "cy": 1024.0, "R": 902.8})
    f1 = extract_tree_features(
        "20150101120000Lh", nodes, x_u8, p, p, np.zeros_like(p), np.zeros_like(p),
        [p * 0.96, p, p * 1.04], {"cx": 1023.0, "cy": 1024.0, "R": 902.8},
        fam_fstds=[np.full_like(p, 0.06), np.full_like(p, 0.06), np.full_like(p, 0.06)],
    )
    spread_idx = FEATURE_NAMES.index("fam_spread")
    assert float(f1[0, spread_idx]) > float(f0[0, spread_idx]), "Expected fam_fstds to restore shrunk K-fold fam_spread!"
    print(f"[OK] Law-of-Total-Variance fstd parity verified (MC rel_err={rel_err:.4%}, fam_spread {f0[0, spread_idx]:.4f} -> {f1[0, spread_idx]:.4f})!")


def test_two_tier_antichain_handoff_no_oversplit():
    rng = np.random.default_rng(777)
    yy, xx = np.meshgrid(np.arange(256, dtype=np.float32), np.arange(256, dtype=np.float32), indexing="ij")
    # Blob A with two lobes (intact single filament at strict_tau=0.26, over-split if naive tau=0.205!) + isolated faint Blob B
    p = (
        0.84 * np.exp(-((xx - 70) ** 2 + (yy - 90) ** 2) / (2 * 9.0 ** 2))
        + 0.80 * np.exp(-((xx - 102) ** 2 + (yy - 90) ** 2) / (2 * 9.0 ** 2))
        + 0.55 * np.exp(-((xx - 190) ** 2 + (yy - 180) ** 2) / (2 * 8.0 ** 2))
    ).astype(np.float32)
    nodes = build_subset_monotone_tree(p, thresholds=(0.15, 0.32, 0.52, 0.72), min_area=18)
    masks = []
    for nd in nodes:
        m = np.zeros((256, 256), dtype=int)
        m[nd["bbox_slice"]][nd["sub_mask"]] = 1
        masks.append(m)

    for _ in range(40):
        u_hat = rng.uniform(0.18, 0.85, size=len(nodes))
        strict_tau = 0.2600
        rescue_tau = 0.2200
        sel_strict = solve_tree_antichain_dp(nodes, u_hat, tau=strict_tau)
        sel_two = solve_two_tier_antichain_handoff(nodes, u_hat, strict_tau=strict_tau, rescue_tau=rescue_tau)

        # 1. Exact preservation of confident DP selection (ZERO over-splitting!)
        recovered_strict = [i for i in sel_two if float(u_hat[i]) >= strict_tau]
        assert recovered_strict == sel_strict, f"Two-tier handoff altered strict DP: {recovered_strict} != {sel_strict}"

        # 2. 100% pixel disjointness across strict + rescue nodes
        if sel_two:
            cov = sum(masks[i] for i in sel_two)
            assert int(cov.max()) <= 1, f"Two-tier handoff produced overlap: max={cov.max()}"
    print(f"[OK] Two-tier Antichain handoff verified across 40 random trials (zero over-splitting + 100% pixel-disjoint)!")


if __name__ == "__main__":
    test_tree_subset_monotonicity_and_exact_dp()
    test_feature_extractor_dimensions()
    test_and_or_yolo_candidates_and_fast_rle()
    test_law_of_total_variance_fstd_parity()
    test_two_tier_antichain_handoff_no_oversplit()
