"""
Evaluate WaveGraphNet: standalone localization MAE, test-time refinement,
and false-positive rate, in one pass.

The inverse branch produces a one-shot coordinate estimate. That estimate
then warm-starts test-time refinement: with both branches frozen, gradient
steps are taken on the COORDINATE ITSELF to reduce the mismatch between the
forward branch's predicted path-wise energy deviation and the measured one
(Adam, 60 steps, lr = 0.01, keeping the running-best iterate per sample).
No network weights are updated. Refinement is applied to damaged samples
only; the false-positive rate is computed from the standalone inverse
predictions on undamaged samples.

Usage:
  python evaluate_refinement.py --split A --seeds 0 1 42 \
      --refine_steps 60 --refine_lr 0.01
"""
import argparse
import pickle

import torch

from data.splits import get_train_val_test_ids, get_removed_sensors
from data.datasets import TRANSDUCER_COORDS, CoupledModelDataset
from data.precompute import build_all_stats, get_all_paths_edge_index_and_col_idxs, get_reduced_sensor_mapping
from models.wavegraphnet_inverse import DeltaEOnlyInverseGNN
from models.wavegraphnet_forward import ForwardGNN
from utils.checkpointer import checkpoint_path, load_checkpoint
from utils.forward_utils import compute_forward_context, make_fwd_batch, build_fixed_damaged_batch
from train_inverse import canonical_pairs

PLATE_MM = 500.0


def refine(fwd_model, di_fixed, p_init, de_scaled, e_pristine_own, device, steps, lr):
    """Identical to evaluate_refinement.py's refine() -- batched test-time
    refinement, running-best iterate per sample."""
    p = p_init.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr)

    n = p.shape[0]
    loss_best = torch.full((n,), float("inf"), device=device)
    p_best = p_init.clone().detach()

    for _ in range(steps):
        opt.zero_grad()
        pred_de = fwd_model(di_fixed, p, e_pristine_own)
        per_sample_loss = ((pred_de - de_scaled) ** 2).mean(dim=1)
        per_sample_loss.mean().backward()
        opt.step()
        with torch.no_grad():
            improved = per_sample_loss.detach() < loss_best
            loss_best = torch.where(improved, per_sample_loss.detach(), loss_best)
            p_best = torch.where(improved.unsqueeze(-1), p.detach(), p_best)

    return p_best, loss_best


def mae_mm(p, yt):
    return torch.sqrt(((p - yt) ** 2).sum(dim=1)) * PLATE_MM


def fpr(preds, targets):
    ud_mask = targets[:, 0] <= 0
    if not ud_mask.any():
        return float("nan"), 0, 0
    pc = preds[ud_mask]
    inside = (pc[:, 0] >= 0) & (pc[:, 0] <= 1) & (pc[:, 1] >= 0) & (pc[:, 1] <= 1)
    return float(inside.float().mean().item()), int(inside.sum().item()), int(ud_mask.sum().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="A", choices=["A", "A2", "B", "B2", "C"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 42])
    ap.add_argument("--refine_steps", type=int, default=60)
    ap.add_argument("--refine_lr", type=float, default=0.01)
    ap.add_argument("--inv_hidden_dim", type=int, default=256)
    ap.add_argument("--inv_num_interaction_layers", type=int, default=3)
    ap.add_argument("--fwd_hidden_dim", type=int, default=256)
    ap.add_argument("--fwd_num_interaction_layers", type=int, default=3)
    ap.add_argument("--ckpt_root", default="checkpoints")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("data/processed/ogw_data.pkl", "rb") as f:
        raw = pickle.load(f)

    removed_sensors = get_removed_sensors(args.split)
    if removed_sensors:
        keep_idxs, keep_cols = get_reduced_sensor_mapping(removed_sensors)
        keep_cols_t = torch.tensor(keep_cols, dtype=torch.long, device=device)
        num_transducers = len(keep_idxs)
    else:
        keep_idxs, keep_cols_t, num_transducers = None, None, 12

    prop_ei, prop_col_idxs, prop_unique = get_all_paths_edge_index_and_col_idxs(num_transducers=num_transducers)
    n_prop = len(prop_unique)

    inv_label = "WaveGraphNet (DeltaE-Only Inverse)"
    fwd_label = "WaveGraphNet (Forward)"

    raw_val_all, raw_test_all, ref_val_all, ref_test_all = [], [], [], []
    fpr_val_all, fpr_test_all = [], []

    for seed in args.seeds:
        train_ids, val_ids, test_ids = get_train_val_test_ids(args.split, list(raw.keys()), seed=seed)
        stats = build_all_stats(raw, train_ids)
        norm_data = stats["normalized_data_map"]
        ds_kw = dict(
            inv_static_edge_index=stats["k12_edge_index"],
            inv_edge_feature_col_idxs=stats["inv_edge_feature_col_idxs"],
            fwd_propagation_col_idxs=stats["propagation_col_idxs"],
            fixed_fft_bin_indices=stats["fixed_fft_bin_indices"],
            amp_means=stats["amp_means"], amp_stds=stats["amp_stds"],
            lookback_fft=stats["lookback_fft"],
            average_baseline_energy_profile=stats["average_baseline_energy_profile"],
            global_max_delta_e=stats["global_max_delta_e"],
        )

        di_val, yt_val, de_val = build_fixed_damaged_batch(norm_data, val_ids, ds_kw, device)
        di_test, yt_test, de_test = build_fixed_damaged_batch(norm_data, test_ids, ds_kw, device)
        if keep_cols_t is not None:
            de_val, de_test = de_val[:, keep_cols_t], de_test[:, keep_cols_t]

        coords_full = torch.tensor([TRANSDUCER_COORDS[i + 1] for i in range(12)],
                                   dtype=torch.float32, device=device)
        coords_t = coords_full if keep_idxs is None else coords_full[keep_idxs]
        s_idx, r_idx = canonical_pairs(num_transducers=num_transducers, device=device)
        edge_length = torch.norm(coords_t[r_idx] - coords_t[s_idx], dim=-1)
        e_pristine_own, scale = compute_forward_context(stats, coords_t, prop_col_idxs, device,
                                                        keep_cols=keep_cols_t)

        inv_model = DeltaEOnlyInverseGNN(hidden_dim=args.inv_hidden_dim,
                                         num_interaction_layers=args.inv_num_interaction_layers).to(device)
        inv_ckpt = load_checkpoint(checkpoint_path(args.split, inv_label, seed, root=args.ckpt_root))
        inv_model.load_state_dict(inv_ckpt["model"])
        inv_model.eval()

        fwd_model = ForwardGNN(
            raw_node_feat_dim=2, physical_edge_feat_dim=6, hidden_dim=args.fwd_hidden_dim,
            num_propagation_pairs=n_prop, num_interaction_layers=args.fwd_num_interaction_layers,
        ).to(device)
        fwd_ckpt = load_checkpoint(checkpoint_path(args.split, fwd_label, seed, root=args.ckpt_root))
        fwd_model.load_state_dict(fwd_ckpt["fwd_model"])
        fwd_model.eval()
        for prm in fwd_model.parameters():
            prm.requires_grad = False

        print(f"\n=== seed {seed} ===")
        for name, di_fixed, yt_fixed, de_fixed in [("val", di_val, yt_val, de_val),
                                                    ("test", di_test, yt_test, de_test)]:
            with torch.no_grad():
                p_hat = inv_model(coords_t, s_idx, r_idx, edge_length, de_fixed)
            raw_mae = mae_mm(p_hat, yt_fixed)

            gf = make_fwd_batch(di_fixed, prop_ei, device, keep_node_idxs=keep_idxs)
            de_scaled = de_fixed / scale
            p_best, _ = refine(fwd_model, gf, p_hat, de_scaled, e_pristine_own,
                               device, args.refine_steps, args.refine_lr)
            ref_mae = mae_mm(p_best, yt_fixed)

            print(f"  {name}: n={yt_fixed.shape[0]} | raw MAE={raw_mae.mean().item():.1f}mm | "
                  f"refined MAE={ref_mae.mean().item():.1f}mm")

            if name == "val":
                raw_val_all.append(raw_mae); ref_val_all.append(ref_mae)
            else:
                raw_test_all.append(raw_mae); ref_test_all.append(ref_mae)

        for name, sample_ids, store in [("val", val_ids, fpr_val_all), ("test", test_ids, fpr_test_all)]:
            ds = CoupledModelDataset(norm_data, sample_ids, **ds_kw)
            items = [ds[i] for i in range(len(sample_ids))]
            de_all = torch.stack([it["delta_e_full"] for it in items]).to(device)
            yt_all = torch.stack([it["y_true"].squeeze(0) for it in items]).to(device)
            if keep_cols_t is not None:
                de_all = de_all[:, keep_cols_t]
            with torch.no_grad():
                p_all = inv_model(coords_t, s_idx, r_idx, edge_length, de_all)
            fpr_val, n_fp, n_ud = fpr(p_all, yt_all)
            print(f"  {name}: FPR={fpr_val*100:.1f}% ({n_fp}/{n_ud})")
            store.append(fpr_val)

    print(f"\n{'='*65}\nAGGREGATE across seeds")
    for name, raw_all, ref_all, fpr_all in [("val", raw_val_all, ref_val_all, fpr_val_all),
                                            ("test", raw_test_all, ref_test_all, fpr_test_all)]:
        raw_cat, ref_cat = torch.cat(raw_all), torch.cat(ref_all)
        print(f"  {name}: raw MAE={raw_cat.mean().item():.1f}mm (n={raw_cat.numel()}) | "
              f"refined MAE={ref_cat.mean().item():.1f}mm | "
              f"standalone FPR={sum(fpr_all)/len(fpr_all)*100:.1f}%")
    combined_raw = torch.cat(raw_val_all + raw_test_all)
    combined_ref = torch.cat(ref_val_all + ref_test_all)
    print(f"  combined(val+test): raw MAE={combined_raw.mean().item():.1f}mm | "
          f"refined MAE={combined_ref.mean().item():.1f}mm")


if __name__ == "__main__":
    main()
