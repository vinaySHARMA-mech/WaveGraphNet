# utils/logger.py
"""Result-logging and evaluation helpers shared by every train_*.py script."""

import json
import os
import csv
import time

import torch
import torch.nn as nn

PLATE_SIZE_MM = 500.0


def log_result(split, model_name, test_loss, filepath="results.json"):
    results = {}
    if os.path.exists(filepath):
        try:
            with open(filepath) as f:
                results = json.load(f)
        except json.JSONDecodeError:
            results = {}
    results.setdefault(split, {})[model_name] = test_loss
    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)


def log_mae_result(split, model_name, seed, val_mae_mm, test_mae_mm, filepath="results_mae.json"):
    """Append per-seed MAE results; aggregate across seeds when reporting a table."""
    results = {}
    if os.path.exists(filepath):
        try:
            with open(filepath) as f:
                results = json.load(f)
        except json.JSONDecodeError:
            results = {}
    results.setdefault(split, {}).setdefault(model_name, []).append(
        {"seed": seed, "val_mae_mm": val_mae_mm, "test_mae_mm": test_mae_mm})
    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)


class EpochLogger:
    """Writes one CSV row per validation checkpoint to `logs/split<X>_<model>_seed<seed>.csv`."""

    def __init__(self, split, model_label, seed, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        safe = model_label.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        fname = f"{log_dir}/split{split}_{safe}_seed{seed}.csv"
        self._f = open(fname, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(["epoch", "phase", "train_loss", "val_mae_norm", "val_mae_mm",
                          "test_mse", "test_mae_mm", "lambda_fwd", "lr", "elapsed_s"])
        self._f.flush()
        self._t0 = time.time()
        self._label = model_label
        print(f"\n[Logger] Per-epoch metrics -> {fname}", flush=True)

    def log(self, epoch, phase, train_loss, val_mae_norm, val_mae_mm, test_mse, test_mae_mm,
            lambda_fwd=0.0, lr=0.0, quiet=False):
        """`quiet=True`: write the CSV row but skip the printed line -- for callers
        with no val/test concept (e.g. forward-branch training), where those
        fields are NaN placeholders."""
        elapsed = time.time() - self._t0
        self._w.writerow([epoch, phase, f"{train_loss:.6f}", f"{val_mae_norm:.6f}", f"{val_mae_mm:.2f}",
                          f"{test_mse:.6f}", f"{test_mae_mm:.2f}", f"{lambda_fwd:.2f}", f"{lr:.2e}",
                          f"{elapsed:.1f}"])
        self._f.flush()
        if quiet:
            return
        print(f"[{self._label}] Ep {epoch:04d} | {phase:6s} | Loss={train_loss:.5f} | "
              f"ValMAE={val_mae_mm:.1f}mm | TestMAE={test_mae_mm:.1f}mm TestMSE={test_mse:.5f} | "
              f"LR={lr:.2e} | {elapsed/60:.1f}min", flush=True)

    def close(self):
        self._f.close()


# ── Evaluation helpers ────────────────────────────────────────────────────────
def _run_loader(model, loader, device):
    """
    Returns (mse, mae_mm_damaged_only). Dispatches on batch type so the same
    helper works for every baseline: dict batches (WaveGraphNet's
    CoupledModelDataset), (x, y) tuples (1D-CNN / LSTM), and PyG `Batch`
    objects (GNN-MLP / GAT).
    """
    crit = nn.MSELoss()
    model.eval()
    tot_mse = dist_sum = 0.0
    dist_n = 0
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                di = batch["data_inv"].to(device)
                yt = batch["y_true"].to(device).squeeze(1)
                pr, n = model(di), di.num_graphs
            elif isinstance(batch, (list, tuple)):
                x, yt = batch[0].to(device), batch[1].to(device)
                pr, n = model(x), x.size(0)
            else:
                batch = batch.to(device)
                yt = batch.y.squeeze(1)
                pr, n = model(batch), yt.size(0)
            tot_mse += crit(pr, yt).item() * n
            mask = yt[:, 0] > 0
            if mask.any():
                dist_sum += ((pr[mask] - yt[mask]) * PLATE_SIZE_MM).norm(dim=-1).sum().item()
                dist_n += mask.sum().item()
    N = len(loader.dataset)
    return tot_mse / N, (dist_sum / dist_n if dist_n > 0 else float("nan"))


def evaluate_val(model, loader, device):
    return _run_loader(model, loader, device)


def evaluate_test(model, loader, device):
    return _run_loader(model, loader, device)
