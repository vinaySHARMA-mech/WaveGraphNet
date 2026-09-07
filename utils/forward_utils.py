# utils/forward_utils.py
"""
Shared helpers for building the forward branch's graph batches and its
fixed pristine-energy input context. Used by both `train_forward.py` and
`evaluate_refinement.py`.
"""

import torch
from torch_geometric.data import Data as PyGData, Batch

from data.datasets import CoupledModelDataset, parse_damage_label, DAMAGE_LABELS


def make_fwd_batch(data_inv, prop_ei, device, indices=None, keep_node_idxs=None):
    """
    Builds a PyG `Batch` of forward-branch graphs (sensor coordinates as
    nodes, `prop_ei` as the fixed path topology) from an inverse-branch
    batch's node features.

    indices        : optional subset of graph indices to include (e.g.
                     damaged-only rows within a mixed damaged+baseline
                     batch). None = all graphs.
    keep_node_idxs : optional list of 0-indexed sensor positions to keep,
                     for a reduced-sensor split (see
                     `data.precompute.get_reduced_sensor_mapping`). None =
                     all 12 nodes.
    """
    prop_ei = prop_ei.to(device)
    idxs = range(data_inv.num_graphs) if indices is None else indices

    def node_feats(i):
        x = data_inv.x[data_inv.batch == i]
        return x[keep_node_idxs] if keep_node_idxs is not None else x

    return Batch.from_data_list(
        [PyGData(x=node_feats(i), edge_index=prop_ei) for i in idxs]
    ).to(device)


def compute_pristine_basis(stats, device):
    """Each of the 66 measured paths' own pristine (undamaged-baseline) energy level: [66]."""
    return stats["average_baseline_energy_profile"].mean(dim=-1).to(device)


def compute_forward_context(stats, coords, prop_col_idxs, device, keep_cols=None):
    """
    Returns (e_pristine_own, scale).

    e_pristine_own : [n_paths] each canonical path's own pristine energy
                     level, already divided by `scale` -- ONE scalar per
                     edge, deliberately not the full pristine basis of
                     every other path (an earlier design concatenated all
                     66 into every edge's input and let the network learn
                     a location-blind shortcut; see
                     `models.wavegraphnet_forward` module docstring).
    scale          : a single global constant (E_pristine.mean()) both
                     e_pristine_own and the ΔE target are divided by, so
                     input and target sit on a comparable range.
    keep_cols      : optional reduced-sensor column filter (Split "C").
    """
    e_pristine = compute_pristine_basis(stats, device)   # [66]
    if keep_cols is not None:
        e_pristine = e_pristine[keep_cols]               # [45] for the reduced-sensor split
    scale = e_pristine.mean().clamp(min=1e-6)
    return e_pristine / scale, scale


def build_fixed_damaged_batch(norm_data, sample_ids, ds_kw, device):
    """
    One-time construction of a single, fixed PyG batch containing ONLY the
    damaged samples from `sample_ids` -- built once per split/seed, reused
    identically for every validation/evaluation call (baseline/undamaged
    samples contribute nothing informative to a damage-localization score).

    Returns (di_batch, yt_batch, de_batch): a PyG `Batch` plus stacked
    ground-truth-coordinate and ΔE tensors, ready to feed directly to
    the inverse/forward branches.
    """
    dmg_ids = [s for s in sample_ids
              if parse_damage_label(s) != "undamaged" and parse_damage_label(s) in DAMAGE_LABELS]
    ds = CoupledModelDataset(norm_data, dmg_ids, **ds_kw)
    items = [ds[i] for i in range(len(dmg_ids))]
    di_batch = Batch.from_data_list([it["data_inv"] for it in items]).to(device)
    yt_batch = torch.stack([it["y_true"].squeeze(0) for it in items]).to(device)
    de_batch = torch.stack([it["delta_e_full"] for it in items]).to(device)
    return di_batch, yt_batch, de_batch
