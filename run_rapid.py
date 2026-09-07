"""
RAPID: training-free classical baseline -- evaluation only, no learning.

Reuses the exact same preprocessing pipeline and spatial hold-out splits as
every learned model, so results are directly comparable.

Usage:
  python run_rapid.py --split A
  python run_rapid.py --split B2   # paper's "Split C"
"""
import argparse
import itertools
import json
import pickle

import numpy as np

from data.splits import get_train_val_test_ids, get_removed_sensors
from data.datasets import TRANSDUCER_COORDS, DAMAGE_LABELS, parse_damage_label
from data.precompute import build_all_stats, get_reduced_sensor_mapping
from models.baselines.rapid import compute_full_delta_e, build_grid, rapid_predict

PLATE_MM = 500.0


def get_targets_and_deltae(sample_ids, norm_data, stats):
    avg_baseline = stats["average_baseline_energy_profile"].numpy()
    targets, deltae = [], []
    for sid in sample_ids:
        sig = norm_data[sid]
        de = compute_full_delta_e(sig, stats["amp_means"], stats["amp_stds"], avg_baseline,
                                  stats["fixed_fft_bin_indices"], stats["lookback_fft"])
        deltae.append(de)
        dmg = parse_damage_label(sid)
        if dmg != "undamaged" and dmg in DAMAGE_LABELS:
            targets.append(np.array(DAMAGE_LABELS[dmg], dtype=np.float64))
        else:
            targets.append(np.array([-0.001, -0.001]))
    return np.stack(targets), np.stack(deltae)


def images_and_peaks(deltae, coords, beta, gx, gy, pairs=None):
    preds, peaks = [], []
    for de in deltae:
        pred, peak = rapid_predict(de, coords, beta, gx, gy, pairs=pairs)
        preds.append(pred); peaks.append(peak)
    return np.stack(preds), np.array(peaks)


def apply_threshold(preds_raw, peaks, threshold):
    preds = preds_raw.copy()
    preds[peaks < threshold] = np.array([-0.001, -0.001])
    return preds


def mae_mm(preds, targets):
    dmg_mask = targets[:, 0] > 0
    if not dmg_mask.any():
        return float("nan")
    err = np.sqrt(((preds[dmg_mask] - targets[dmg_mask]) ** 2).sum(axis=1))
    return float(err.mean() * PLATE_MM)


def fpr_metric(preds, targets):
    ud_mask = targets[:, 0] <= 0
    if not ud_mask.any():
        return float("nan"), 0, 0
    pc = preds[ud_mask]
    inside = (pc[:, 0] >= 0) & (pc[:, 0] <= 1) & (pc[:, 1] >= 0) & (pc[:, 1] <= 1)
    return float(inside.mean()), int(inside.sum()), int(ud_mask.sum())


def select_beta(val_deltae, val_targets, coords, betas, gx, gy, pairs=None):
    """Pick beta by pure localization accuracy on damaged val samples."""
    best_beta, best_mae, trace = None, float("inf"), []
    for beta in betas:
        preds_raw, _ = images_and_peaks(val_deltae, coords, beta, gx, gy, pairs=pairs)
        mae = mae_mm(preds_raw, val_targets)
        trace.append((beta, mae))
        if mae < best_mae:
            best_mae, best_beta = mae, beta
    return best_beta, best_mae, trace


def select_threshold(val_peaks, val_targets):
    """Pick the peak-intensity threshold minimizing 0.5*(FPR + miss-rate) on val."""
    dmg_mask = val_targets[:, 0] > 0
    ud_mask = ~dmg_mask
    candidates = np.unique(val_peaks)
    if len(candidates) == 0:
        return 0.0, float("nan")
    cand = np.concatenate([[0.0], (candidates[:-1] + candidates[1:]) / 2.0, [candidates[-1] + 1.0]])
    best_t, best_cost = 0.0, float("inf")
    for t in cand:
        fpr = (val_peaks[ud_mask] >= t).mean() if ud_mask.any() else 0.0
        miss = (val_peaks[dmg_mask] < t).mean() if dmg_mask.any() else 0.0
        cost = 0.5 * (fpr + miss)
        if cost < best_cost:
            best_cost, best_t = cost, t
    return float(best_t), float(best_cost)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="A", choices=["A", "A2", "B", "B2", "C"])
    p.add_argument("--grid_res", type=int, default=150)
    p.add_argument("--betas", type=float, nargs="+", default=[1.02, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0])
    p.add_argument("--seed", type=int, default=0,
                   help="Only affects the baseline train/val/test shuffle (damaged-sample "
                        "assignment is deterministic by label).")
    args = p.parse_args()

    with open("data/processed/ogw_data.pkl", "rb") as f:
        raw = pickle.load(f)
    train_ids, val_ids, test_ids = get_train_val_test_ids(args.split, list(raw.keys()), seed=args.seed)

    print(f"\n{'='*65}\n  RAPID (training-free) | split={args.split}\n{'='*65}", flush=True)

    stats = build_all_stats(raw, train_ids)
    norm_data = stats["normalized_data_map"]
    coords_full = np.array([TRANSDUCER_COORDS[i + 1] for i in range(12)], dtype=np.float64)

    removed_sensors = get_removed_sensors(args.split)
    if removed_sensors:
        keep_idxs, keep_cols = get_reduced_sensor_mapping(removed_sensors)
        coords = coords_full[keep_idxs]
        pairs = sorted(itertools.combinations(range(len(keep_idxs)), 2))
        print(f"  [Split {args.split}] removed sensors {sorted(removed_sensors)} -> "
              f"{len(keep_idxs)} transducers, {len(pairs)} paths", flush=True)
    else:
        coords, keep_cols, pairs = coords_full, None, None

    val_targets, val_deltae = get_targets_and_deltae(val_ids, norm_data, stats)
    test_targets, test_deltae = get_targets_and_deltae(test_ids, norm_data, stats)
    if keep_cols is not None:
        val_deltae, test_deltae = val_deltae[:, keep_cols], test_deltae[:, keep_cols]

    gx, gy = build_grid(args.grid_res)

    best_beta, best_val_mae_raw, beta_trace = select_beta(
        val_deltae, val_targets, coords, args.betas, gx, gy, pairs=pairs)
    print("  beta search (raw centroid MAE on val damaged samples):")
    for beta, mae in beta_trace:
        marker = " <-- selected" if beta == best_beta else ""
        print(f"    beta={beta:<5} val_MAE={mae:7.1f}mm{marker}")

    val_preds_raw, val_peaks = images_and_peaks(val_deltae, coords, best_beta, gx, gy, pairs=pairs)
    threshold, thresh_cost = select_threshold(val_peaks, val_targets)
    print(f"  selected beta={best_beta}, threshold={threshold:.6f} "
          f"(val 0.5*(FPR+miss)={thresh_cost:.3f})", flush=True)

    val_preds = apply_threshold(val_preds_raw, val_peaks, threshold)
    seen_mae = mae_mm(val_preds, val_targets)
    seen_fpr, seen_fp_n, seen_ud_n = fpr_metric(val_preds, val_targets)

    test_preds_raw, test_peaks = images_and_peaks(test_deltae, coords, best_beta, gx, gy, pairs=pairs)
    test_preds = apply_threshold(test_preds_raw, test_peaks, threshold)
    unseen_mae = mae_mm(test_preds, test_targets)
    unseen_fpr, unseen_fp_n, unseen_ud_n = fpr_metric(test_preds, test_targets)

    print(f"\n[RESULT] RAPID | split={args.split} | beta={best_beta} thr={threshold:.4f}")
    print(f"  Seen  (val,  damaged) MAE  = {seen_mae:.1f} mm")
    print(f"  Unseen(test, damaged) MAE  = {unseen_mae:.1f} mm")
    print(f"  Unseen(test) FPR           = {unseen_fpr*100:.1f}% ({unseen_fp_n}/{unseen_ud_n})")
    print(f"  Seen  (val)  FPR           = {seen_fpr*100:.1f}% ({seen_fp_n}/{seen_ud_n})", flush=True)

    out = dict(split=args.split, beta=best_beta, threshold=threshold,
              seen_mae_mm=seen_mae, unseen_mae_mm=unseen_mae,
              seen_fpr=seen_fpr, seen_fp_n=seen_fp_n, seen_ud_n=seen_ud_n,
              unseen_fpr=unseen_fpr, unseen_fp_n=unseen_fp_n, unseen_ud_n=unseen_ud_n,
              beta_search_trace=beta_trace)
    out_path = f"rapid_evaluation_split{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=4)
    print(f"\n[Saved] -> {out_path}")


if __name__ == "__main__":
    main()
