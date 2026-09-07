# utils/checkpointer.py
"""
Shared checkpoint utilities.

Naming convention (used by every train_*.py script):
    checkpoints/<split>/<safe_model_name>_seed<seed>.pt
    e.g.  checkpoints/A/1D_CNN_seed42.pt
          checkpoints/A/WaveGraphNet__DeltaE_Only_Inverse_seed0.pt
"""

import os
import re
import torch


def _safe_name(name: str) -> str:
    """Turn a human-readable model label into a filesystem-safe filename stem."""
    return re.sub(r"[^\w]", "_", name).strip("_")


def checkpoint_path(split: str, model_label: str, seed: int, root: str = "checkpoints") -> str:
    """Return the canonical path for a checkpoint file."""
    return os.path.join(root, split, f"{_safe_name(model_label)}_seed{seed}.pt")


def save_checkpoint(path: str, config: dict, test_loss: float, **state_dicts):
    """
    Save a checkpoint: a dict with `config`, `test_loss`, and any number of
    named state dicts / extra scalars, e.g.
        save_checkpoint(path, config=vars(args), test_loss=tm, val_loss=vm,
                        model=model.state_dict())
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"config": config, "test_loss": test_loss}
    payload.update(state_dicts)
    torch.save(payload, path)
    print(f"  [checkpoint] saved -> {path}  (test_loss={test_loss:.6f})")


def load_checkpoint(path: str) -> dict:
    """Load and return the checkpoint dict from `path`."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)
