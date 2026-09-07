"""
Train the forward consistency branch (`ForwardGNN`) standalone.

Trained only on damaged samples: each sample's TRUE defect coordinate is
used to build the forward graph, and the model regresses the measured
path-wise energy deviation. Both the pristine energy input and the ΔE
target are divided by the same global scaling constant (the mean pristine
energy over the training paths). Checkpoint selection uses forward
prediction error on the damaged validation samples.

The inverse branch is trained completely independently (train_inverse.py);
the two share no gradients and are only combined at inference, inside
test-time refinement.

Usage:
  python train_forward.py --split A --seed 0 --fwd_hidden_dim 256 \
      --num_interaction_layers 3 --epochs 10000 --val_every 2 --early_stop_patience 50
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # required for deterministic cuBLAS GEMM

import argparse
import pickle
import sys

import numpy as np
import random
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data.splits import get_train_val_test_ids, get_removed_sensors
from data.datasets import CoupledModelDataset, TRANSDUCER_COORDS
from data.precompute import build_all_stats, get_all_paths_edge_index_and_col_idxs, get_reduced_sensor_mapping
from models.wavegraphnet_forward import ForwardGNN
from utils.checkpointer import save_checkpoint, checkpoint_path
from utils.logger import EpochLogger
from utils.forward_utils import make_fwd_batch, compute_forward_context, build_fixed_damaged_batch

_TQDM_DISABLE = not sys.stdout.isatty()


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def train_phase1_fwd(fwd_model, loader, opt_fwd, prop_ei, e_pristine_own, scale, device,
                     keep_cols=None, keep_node_idxs=None):
    """One epoch. Same loss/masking convention as train_forward.py."""
    fwd_model.train()
    total = 0.0; n_dmg_total = 0
    for batch in tqdm(loader, leave=False, disable=_TQDM_DISABLE):
        di = batch["data_inv"].to(device)
        yt_full = batch["y_true"].to(device).squeeze(1)
        dmg_mask = yt_full[:, 0] > 0
        if not dmg_mask.any():
            continue
        opt_fwd.zero_grad()
        dmg_idx = dmg_mask.nonzero(as_tuple=True)[0].tolist()
        yt = yt_full[dmg_mask]
        de = batch["delta_e_full"].to(device)[dmg_mask]
        if keep_cols is not None:
            de = de[:, keep_cols]
        de = de / scale
        gf = make_fwd_batch(di, prop_ei, device, indices=dmg_idx, keep_node_idxs=keep_node_idxs)
        loss = ((fwd_model(gf, yt, e_pristine_own) - de) ** 2).mean()
        loss.backward(); opt_fwd.step()
        n = len(dmg_idx)
        total += loss.item() * n; n_dmg_total += n
    return total / max(n_dmg_total, 1)


def evaluate_phase1_fwd(fwd_model, di_fixed, yt_fixed, de_fixed, prop_ei,
                        e_pristine_own, scale, device, keep_cols=None, keep_node_idxs=None):
    """Same loss formula as training, on a FIXED batch of val damaged samples, no gradient."""
    fwd_model.eval()
    with torch.no_grad():
        gf = make_fwd_batch(di_fixed, prop_ei, device, keep_node_idxs=keep_node_idxs)
        de = de_fixed if keep_cols is None else de_fixed[:, keep_cols]
        de_scaled = de / scale
        pred = fwd_model(gf, yt_fixed, e_pristine_own)
        return ((pred - de_scaled) ** 2).mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="A", choices=["A", "A2", "B", "B2", "C"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fwd_hidden_dim", type=int, default=256)
    p.add_argument("--num_interaction_layers", type=int, default=3)
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--val_every", type=int, default=2)
    p.add_argument("--early_stop_patience", type=int, default=50)
    p.add_argument("--ckpt_root", default="checkpoints")
    p.add_argument("--log_dir", default="logs")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label = "WaveGraphNet (Forward)"

    print(f"\n{'='*65}\n  {label} | split={args.split} seed={args.seed}")
    print(f"  epochs={args.epochs} lr={args.lr} hidden_dim={args.fwd_hidden_dim}\n{'='*65}", flush=True)

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
    val_di_fixed, val_yt_fixed, val_de_fixed = build_fixed_damaged_batch(norm_data, val_ids, ds_kw, device)
    print(f"  fixed damaged-only val batch: {val_yt_fixed.shape[0]} samples", flush=True)

    removed_sensors = get_removed_sensors(args.split)
    if removed_sensors:
        keep_idxs, keep_cols = get_reduced_sensor_mapping(removed_sensors)
        num_transducers = len(keep_idxs)
        keep_cols_t = torch.tensor(keep_cols, dtype=torch.long, device=device)
        print(f"  [Split {args.split}] removed sensors {sorted(removed_sensors)} -> "
              f"{num_transducers} transducers, {len(keep_cols)} paths", flush=True)
    else:
        keep_idxs, keep_cols_t, num_transducers = None, None, 12

    prop_ei, prop_col_idxs, prop_unique = get_all_paths_edge_index_and_col_idxs(num_transducers=num_transducers)
    n_prop = len(prop_unique)   # 66, or 45 for the reduced-sensor split

    fwd_model = ForwardGNN(
        raw_node_feat_dim=2, physical_edge_feat_dim=6, hidden_dim=args.fwd_hidden_dim,
        num_propagation_pairs=n_prop, num_interaction_layers=args.num_interaction_layers,
    ).to(device)

    coords_full = torch.tensor([TRANSDUCER_COORDS[i + 1] for i in range(12)], dtype=torch.float32, device=device)
    coords_t = coords_full if keep_idxs is None else coords_full[keep_idxs]
    e_pristine_own, scale = compute_forward_context(stats, coords_t, prop_col_idxs, device, keep_cols=keep_cols_t)
    print(f"  forward context: e_pristine_own shape={tuple(e_pristine_own.shape)} scale={scale.item():.4f}",
          flush=True)

    opt = optim.Adam(fwd_model.parameters(), lr=args.lr)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.8, patience=20)
    ckpt_path = checkpoint_path(args.split, label, args.seed, root=args.ckpt_root)
    logger = EpochLogger(args.split, label, args.seed, log_dir=args.log_dir)

    best_val_loss = float("inf"); evals_since_improve = 0

    for ep in range(1, args.epochs + 1):
        fl = train_phase1_fwd(fwd_model, tr, opt, prop_ei, e_pristine_own, scale, device,
                              keep_cols=keep_cols_t, keep_node_idxs=keep_idxs)

        if ep % args.val_every == 0 or ep in (1, args.epochs):
            vl = evaluate_phase1_fwd(fwd_model, val_di_fixed, val_yt_fixed, val_de_fixed,
                                     prop_ei, e_pristine_own, scale, device,
                                     keep_cols=keep_cols_t, keep_node_idxs=keep_idxs)
            sch.step(vl)
            logger.log(ep, "fwd", fl, vl, float("nan"), float("nan"), float("nan"),
                       lr=opt.param_groups[0]["lr"], quiet=True)
            print(f"[{label}] Ep {ep:04d} | TrainLoss={fl:.6f} | ValLoss={vl:.6f} | "
                  f"LR={opt.param_groups[0]['lr']:.2e} | evals_since_improve={evals_since_improve}", flush=True)

            if vl < best_val_loss:
                best_val_loss = vl; evals_since_improve = 0
                save_checkpoint(ckpt_path, config=vars(args), test_loss=fl, val_loss=vl,
                                fwd_model=fwd_model.state_dict())
                print(f"  * Ep {ep:04d} | ValLoss={vl:.6f} [saved]", flush=True)
            else:
                evals_since_improve += 1

            if args.early_stop_patience > 0 and evals_since_improve >= args.early_stop_patience:
                print(f"  [early stop] no val-loss improvement in {evals_since_improve} evaluations "
                      f"-- stopping at epoch {ep}", flush=True)
                break

    logger.close()
    print(f"\n[DONE] {label} | best_val_loss={best_val_loss:.6f}", flush=True)


if __name__ == "__main__":
    main()
