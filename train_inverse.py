"""
Train WaveGraphNet's inverse localization branch (`DeltaEOnlyInverseGNN`)
standalone -- no forward model involved at all during training.

Usage:
  python train_inverse.py --split A --seed 0 --hidden_dim 256 \
      --epochs 10000 --val_every 2 --early_stop_patience 50 --batch_size 8

Checkpoints are saved to checkpoints/<split>/WaveGraphNet__DeltaE_Only_Inverse_seed<seed>.pt
whenever validation MAE improves.

Reproducibility note: the message-passing layers use `index_add_`-based
scatter aggregation, which on CUDA is non-deterministic by default (the
order floating-point contributions are summed in is not fixed run-to-run,
even with the same seed) -- `torch.backends.cudnn.deterministic=True` alone
does NOT cover this, it only fixes cuDNN's convolution algorithm choice.
Two training runs with an identical seed can therefore converge to visibly
different final weights. This script forces full determinism instead (see
`set_seed` and `CUBLAS_WORKSPACE_CONFIG` below), verified to give
bit-identical results across repeated runs with the same seed -- at some
throughput cost, since the deterministic scatter/GEMM kernels are slower.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # required for deterministic cuBLAS GEMM

import argparse
import itertools
import pickle
import random
import sys

import numpy as np
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

from data.splits import get_train_val_test_ids, get_removed_sensors
from data.datasets import CoupledModelDataset, TRANSDUCER_COORDS
from data.precompute import build_all_stats, get_reduced_sensor_mapping
from models.wavegraphnet_inverse import DeltaEOnlyInverseGNN
from utils.checkpointer import save_checkpoint, checkpoint_path
from utils.logger import EpochLogger

_TQDM_DISABLE = not sys.stdout.isatty()
PLATE_MM = 500.0
LABEL = "WaveGraphNet (DeltaE-Only Inverse)"


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def canonical_pairs(num_transducers=12, device="cpu"):
    """The 66 (or fewer) canonical, undirected path endpoints: s_idx[k] < r_idx[k] always."""
    all_pairs = sorted(itertools.combinations(range(num_transducers), 2))
    s_idx = torch.tensor([i for i, j in all_pairs], dtype=torch.long, device=device)
    r_idx = torch.tensor([j for i, j in all_pairs], dtype=torch.long, device=device)
    return s_idx, r_idx


def euclidean_mae(model, loader, coords, s_idx, r_idx, edge_length, device, keep_cols=None):
    model.eval(); total = 0.0; count = 0
    with torch.no_grad():
        for batch in loader:
            de = batch["delta_e_full"].to(device)
            if keep_cols is not None:
                de = de[:, keep_cols]
            yt = batch["y_true"].to(device).squeeze(1)
            mask = yt[:, 0] > 0
            if not mask.any():
                continue
            pred = model(coords, s_idx, r_idx, edge_length, de)[mask]
            total += torch.sqrt(((pred - yt[mask]) ** 2).sum(dim=1)).mean().item()
            count += 1
    return (total / count) if count > 0 else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="A", choices=["A", "A2", "B", "B2", "C"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_interaction_layers", type=int, default=3)
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--val_every", type=int, default=2)
    p.add_argument("--early_stop_patience", type=int, default=50,
                   help="Stop after this many evaluations (each --val_every epochs) with "
                        "no val MAE improvement. 0 = disabled.")
    p.add_argument("--ckpt_root", default="checkpoints")
    p.add_argument("--log_dir", default="logs")
    p.add_argument("--p_ud", type=float, nargs=2, default=[-0.5, -0.5],
                   help="No-damage sentinel coordinate. Default (-0.5,-0.5), NOT the "
                        "(-0.001,-0.001) used by data.datasets.NO_DAMAGE_TARGET (shared by "
                        "the baselines/RAPID). Reasoning: p_hat = damage_prob*p_conv + "
                        "(1-damage_prob)*p_ud, so the coordinate-domain decision boundary "
                        "sits at damage_prob* = |p_ud|/(|p_conv|+|p_ud|). With |p_ud|=0.001 "
                        "and typical |p_conv|~0.7 (measured empirically), damage_prob* ~ "
                        "0.0014 -- the model must be almost machine-precision certain before "
                        "the FPR check reflects it, even though damage_prob is a well- "
                        "calibrated, confident (<0.02) signal on every undamaged sample "
                        "checked. (-0.5,-0.5) (|p_ud|~0.71, matching |p_conv|'s scale) moves "
                        "the boundary to damage_prob* ~ 0.5 -- the semantically correct spot.")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*65}\n  {LABEL} | split={args.split} seed={args.seed}")
    print(f"  epochs={args.epochs} lr={args.lr} hidden_dim={args.hidden_dim}\n{'='*65}", flush=True)

    with open("data/processed/ogw_data.pkl", "rb") as f:
        raw = pickle.load(f)
    train_ids, val_ids, test_ids = get_train_val_test_ids(args.split, list(raw.keys()), seed=args.seed)

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
    tr = DataLoader(CoupledModelDataset(norm_data, train_ids, **ds_kw), batch_size=args.batch_size, shuffle=True)
    vl = DataLoader(CoupledModelDataset(norm_data, val_ids, **ds_kw), batch_size=args.batch_size, shuffle=False)
    te = DataLoader(CoupledModelDataset(norm_data, test_ids, **ds_kw), batch_size=args.batch_size, shuffle=False)

    coords_full = torch.tensor([TRANSDUCER_COORDS[i + 1] for i in range(12)], dtype=torch.float32, device=device)
    removed_sensors = get_removed_sensors(args.split)
    if removed_sensors:
        keep_idxs, keep_cols = get_reduced_sensor_mapping(removed_sensors)
        coords = coords_full[keep_idxs]
        keep_cols_t = torch.tensor(keep_cols, dtype=torch.long, device=device)
        num_transducers = len(keep_idxs)
        print(f"  [Split {args.split}] removed sensors {sorted(removed_sensors)} -> "
              f"{num_transducers} transducers, {len(keep_cols)} paths", flush=True)
    else:
        coords, keep_cols_t, num_transducers = coords_full, None, 12

    s_idx, r_idx = canonical_pairs(num_transducers=num_transducers, device=device)
    edge_length = torch.norm(coords[r_idx] - coords[s_idx], dim=-1)

    p_ud = torch.tensor(args.p_ud, dtype=torch.float32, device=device)
    model = DeltaEOnlyInverseGNN(hidden_dim=args.hidden_dim,
                                 num_interaction_layers=args.num_interaction_layers,
                                 p_ud=tuple(args.p_ud)).to(device)

    ckpt_path = checkpoint_path(args.split, LABEL, args.seed, root=args.ckpt_root)
    logger = EpochLogger(args.split, LABEL, args.seed, log_dir=args.log_dir)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.8, patience=20)

    best_val_mae = float("inf"); best_test_mae = float("nan"); best_test_mse = float("nan")
    evals_since_improve = 0

    for ep in range(1, args.epochs + 1):
        model.train(); tl = 0.0; n = 0
        for batch in tr:
            opt.zero_grad()
            de = batch["delta_e_full"].to(device)
            if keep_cols_t is not None:
                de = de[:, keep_cols_t]
            yt = batch["y_true"].to(device).squeeze(1)
            # Remap the dataset's shared no-damage sentinel ((-0.001,-0.001), see
            # data.datasets.NO_DAMAGE_TARGET) to this model's own, better-scaled p_ud
            # (see --p_ud's help text) -- undamaged rows are identified the same way
            # euclidean_mae's damaged-mask does (first coordinate <= 0), just inverted.
            undamaged = yt[:, 0] <= 0
            if undamaged.any():
                yt = yt.clone()
                yt[undamaged] = p_ud.unsqueeze(0).expand(int(undamaged.sum()), -1)
            pred = model(coords, s_idx, r_idx, edge_length, de)
            loss = ((pred - yt) ** 2).mean()
            loss.backward(); opt.step()
            tl += loss.item() * de.shape[0]; n += de.shape[0]
        tl /= n

        if ep % args.val_every == 0 or ep in (1, args.epochs):
            vm = euclidean_mae(model, vl, coords, s_idx, r_idx, edge_length, device, keep_cols=keep_cols_t)
            vmm = vm * PLATE_MM
            sch.step(vm)
            tm = euclidean_mae(model, te, coords, s_idx, r_idx, edge_length, device, keep_cols=keep_cols_t)
            tmm = tm * PLATE_MM
            logger.log(ep, "inv", tl, vm, vmm, tm, tmm, lr=opt.param_groups[0]["lr"])
            print(f"[{LABEL}] Ep {ep:04d} | Loss={tl:.5f} | ValMAE={vmm:.1f}mm | "
                  f"TestMAE={tmm:.1f}mm | LR={opt.param_groups[0]['lr']:.2e}", flush=True)

            if vmm < best_val_mae:
                best_val_mae, best_test_mae, best_test_mse = vmm, tmm, tm
                evals_since_improve = 0
                save_checkpoint(ckpt_path, config=vars(args), test_loss=tm, val_loss=vmm,
                                model=model.state_dict())
                print(f"  * Ep {ep:04d} | ValMAE={vmm:.1f}mm -> TestMAE={tmm:.1f}mm [saved]", flush=True)
            else:
                evals_since_improve += 1

            if args.early_stop_patience > 0 and evals_since_improve >= args.early_stop_patience:
                print(f"  [early stop] no val MAE improvement in {evals_since_improve} evaluations "
                      f"-- stopping at epoch {ep}", flush=True)
                break

    logger.close()
    print(f"\n[DONE] {LABEL} | best_val_mae={best_val_mae:.1f}mm | "
          f"test_mae={best_test_mae:.1f}mm | test_mse={best_test_mse:.5f}", flush=True)


if __name__ == "__main__":
    main()
