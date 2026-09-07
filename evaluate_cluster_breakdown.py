"""
Per-cluster test-MAE breakdown for the held-out damage clusters, using the
forward branch (`ForwardGNN`) for test-time refinement.

This reproduces the per-cluster tables in the paper:
  --split B   -> cluster 1 (D1-D4)   vs cluster 6 (D21-D24)
  --split B2  -> cluster 6 (D21-D24) vs cluster 7 (D25-D28)   [paper "Split C"]

It mirrors `evaluate_refinement.py` exactly (same inverse branch, same
Adam refinement, same 60 steps / lr 0.01) but keeps each sample's damage label
so the errors can be grouped by cluster instead of only aggregated.

Reported as mean +/- std over the three seeds, where the std is the population
standard deviation over the per-seed cluster means (same convention as every
other mean +/- std in the paper).

Usage:
  python evaluate_cluster_breakdown.py --split B  --seeds 0 1 42
  python evaluate_cluster_breakdown.py --split B2 --seeds 0 1 42
"""
import argparse
import pickle

import torch

from data.splits import get_train_val_test_ids, get_removed_sensors
from data.datasets import TRANSDUCER_COORDS, parse_damage_label, DAMAGE_LABELS
from data.precompute import build_all_stats, get_all_paths_edge_index_and_col_idxs, get_reduced_sensor_mapping
from models.wavegraphnet_inverse import DeltaEOnlyInverseGNN
from models.wavegraphnet_forward import ForwardGNN
from utils.checkpointer import checkpoint_path, load_checkpoint
from utils.forward_utils import compute_forward_context, make_fwd_batch, build_fixed_damaged_batch
from train_inverse import canonical_pairs

PLATE_MM = 500.0

# Held-out clusters per split (see the paper's Fig. 2 / Section 5.4).
CLUSTERS = {
    "B":  {"cluster 1 (D1-D4)":    {"D1", "D2", "D3", "D4"},
           "cluster 6 (D21-D24)":  {"D21", "D22", "D23", "D24"}},
    "B2": {"cluster 6 (D21-D24)":  {"D21", "D22", "D23", "D24"},
           "cluster 7 (D25-D28)":  {"D25", "D26", "D27", "D28"}},
}


def refine(fwd_model, di_fixed, p_init, de_scaled, e_pristine_own, device, steps, lr):
    """Identical to evaluate_refinement.py's refine(): batched test-time
    refinement keeping the running-best iterate per sample."""
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


def pop_mean_std(vals):
    """Mean and POPULATION std (divide by N), the convention used in the paper."""
    m = sum(vals) / len(vals)
    return m, (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="B", choices=["B", "B2"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 42])
    ap.add_argument("--refine_steps", type=int, default=60)
    ap.add_argument("--refine_lr", type=float, default=0.01)
    ap.add_argument("--inv_hidden_dim", type=int, default=256)
    ap.add_argument("--inv_num_interaction_layers", type=int, default=3)
    ap.add_argument("--fwd_hidden_dim", type=int, default=256)
    ap.add_argument("--fwd_num_interaction_layers", type=int, default=3)
    ap.add_argument("--ckpt_root", default="checkpoints")
    args = ap.parse_args()

    clusters = CLUSTERS[args.split]
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

    per_seed = {c: {"raw": [], "ref": []} for c in clusters}

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

        # Rebuild the damaged-sample id list the same way build_fixed_damaged_batch
        # does, so the labels line up row-for-row with the batched tensors.
        dmg_ids = [s for s in test_ids
                   if parse_damage_label(s) != "undamaged" and parse_damage_label(s) in DAMAGE_LABELS]
        labels = [parse_damage_label(s) for s in dmg_ids]

        di_test, yt_test, de_test = build_fixed_damaged_batch(norm_data, test_ids, ds_kw, device)
        if keep_cols_t is not None:
            de_test = de_test[:, keep_cols_t]

        coords_full = torch.tensor([TRANSDUCER_COORDS[i + 1] for i in range(12)],
                                   dtype=torch.float32, device=device)
        coords_t = coords_full if keep_idxs is None else coords_full[keep_idxs]
        s_idx, r_idx = canonical_pairs(num_transducers=num_transducers, device=device)
        edge_length = torch.norm(coords_t[r_idx] - coords_t[s_idx], dim=-1)
        e_pristine_own, scale = compute_forward_context(stats, coords_t, prop_col_idxs, device,
                                                        keep_cols=keep_cols_t)

        inv_model = DeltaEOnlyInverseGNN(hidden_dim=args.inv_hidden_dim,
                                         num_interaction_layers=args.inv_num_interaction_layers).to(device)
        inv_model.load_state_dict(load_checkpoint(
            checkpoint_path(args.split, inv_label, seed, root=args.ckpt_root))["model"])
        inv_model.eval()

        fwd_model = ForwardGNN(
            raw_node_feat_dim=2, physical_edge_feat_dim=6, hidden_dim=args.fwd_hidden_dim,
            num_propagation_pairs=n_prop, num_interaction_layers=args.fwd_num_interaction_layers,
        ).to(device)
        fwd_model.load_state_dict(load_checkpoint(
            checkpoint_path(args.split, fwd_label, seed, root=args.ckpt_root))["fwd_model"])
        fwd_model.eval()
        for prm in fwd_model.parameters():
            prm.requires_grad = False

        with torch.no_grad():
            p_hat = inv_model(coords_t, s_idx, r_idx, edge_length, de_test)
        raw_mae = mae_mm(p_hat, yt_test)

        gf = make_fwd_batch(di_test, prop_ei, device, keep_node_idxs=keep_idxs)
        p_best, _ = refine(fwd_model, gf, p_hat, de_test / scale, e_pristine_own,
                           device, args.refine_steps, args.refine_lr)
        ref_mae = mae_mm(p_best, yt_test)

        assert len(labels) == raw_mae.shape[0], (len(labels), raw_mae.shape[0])

        print(f"\n=== seed {seed} ===")
        for cname, members in clusters.items():
            idxs = [i for i, lab in enumerate(labels) if lab in members]
            r = raw_mae[idxs].mean().item()
            f = ref_mae[idxs].mean().item()
            per_seed[cname]["raw"].append(r)
            per_seed[cname]["ref"].append(f)
            print(f"  {cname} (n={len(idxs)}): standalone={r:.1f}mm | refined={f:.1f}mm")

    print(f"\n{'='*66}\nAGGREGATE (mean +/- population std over {len(args.seeds)} seeds)")
    for cname in clusters:
        rm, rs = pop_mean_std(per_seed[cname]["raw"])
        fm, fs = pop_mean_std(per_seed[cname]["ref"])
        print(f"  {cname}: standalone={rm:.1f} +/- {rs:.1f} mm | refined={fm:.1f} +/- {fs:.1f} mm")


if __name__ == "__main__":
    main()
