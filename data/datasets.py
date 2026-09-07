# data/datasets.py
"""
PyTorch Dataset classes for every model in this repository.

Every dataset here consumes `stats["normalized_data_map"]` (built once per
split by `data.precompute.build_all_stats`) plus a handful of precomputed
per-split statistics — never the raw signal pickle directly.

Two families:
  - "raw" input: the full per-path spectral descriptor (256 frequency bins x
    [amplitude, phase] = 512-dim, plus 3-dim geometry = 515-dim per path).
    Used by the four standard baselines (1D-CNN, LSTM, GNN-MLP, GAT).
  - "delta_e" input: the single scalar path-wise energy-deviation ΔE_ij (the
    same quantity the forward branch predicts). Used by WaveGraphNet's
    inverse/forward branches, and as a fairness-ablation input option for
    every baseline (so "does ΔE alone explain the gain" can be tested
    independently of architecture).
"""

import itertools

import numpy as np
import scipy.fft
import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data as PyGData

TRANSDUCER_COORDS = {
    1: [0.9, 0.94], 2: [0.74, 0.94], 3: [0.58, 0.94], 4: [0.42, 0.94],
    5: [0.26, 0.94], 6: [0.1, 0.94], 7: [0.9, 0.06], 8: [0.74, 0.06],
    9: [0.58, 0.06], 10: [0.42, 0.06], 11: [0.26, 0.06], 12: [0.1, 0.06],
}

DAMAGE_LABELS = {
    "D1": [0.1, 0.83], "D2": [0.13, 0.83], "D3": [0.1, 0.8],
    "D4": [0.13, 0.8], "D5": [0.5, 0.854], "D6": [0.53, 0.854],
    "D7": [0.5, 0.824], "D8": [0.53, 0.824], "D9": [0.36, 0.69],
    "D10": [0.39, 0.69], "D11": [0.36, 0.66], "D12": [0.39, 0.66],
    "D13": [0.64, 0.55], "D14": [0.67, 0.55], "D15": [0.64, 0.52],
    "D16": [0.67, 0.52], "D17": [0.26, 0.39], "D18": [0.29, 0.39],
    "D19": [0.26, 0.36], "D20": [0.29, 0.36], "D21": [0.87, 0.41],
    "D22": [0.9, 0.41], "D23": [0.87, 0.38], "D24": [0.9, 0.38],
    "D25": [0.5, 0.18], "D26": [0.53, 0.18], "D27": [0.5, 0.15],
    "D28": [0.53, 0.15],
}

# The out-of-domain "no damage" target: outside [0,1]^2, so it is never
# confusable with a real in-plate location, and every model's regression
# head must learn to reach it for undamaged samples.
#
# (-0.5, -0.5), not (-0.001, -0.001): the original value sat only 0.001
# outside the domain boundary, which made the FPR check (does the
# prediction land in [0,1]^2) pathologically sensitive to tiny prediction
# noise for EVERY model trained against it, not just WaveGraphNet's gated
# decoder -- a well-converged model needs a real margin to reliably land
# on the correct side of the boundary. Confirmed for WaveGraphNet's own
# sentinel-interpolation gate: moving from -0.001 to -0.5 made its
# reported FPR agree with its own damage_prob>0.5 decision on every
# tested seed, where the tight value did not on 2 of 3.
NO_DAMAGE_TARGET = (-0.5, -0.5)


def parse_damage_label(sample_id: str) -> str:
    if "baseline" in sample_id:
        return "undamaged"
    return sample_id.split("_")[0]


def _target_xy(sample_id: str):
    dmg = parse_damage_label(sample_id)
    if dmg != "undamaged" and dmg in DAMAGE_LABELS:
        return DAMAGE_LABELS[dmg]
    return list(NO_DAMAGE_TARGET)


def compute_delta_e_66(sig, lookback, fft_bins, amp_means, amp_stds, avg_baseline):
    """
    Per-sample path-wise energy-deviation vector ΔE_ij, over all 66 measured
    paths, scaled by a single global constant (`avg_baseline.mean()`) so
    every ΔE-consuming model receives it on a directly comparable scale.

    sig : np.ndarray [T, 66] normalized differential signal
    Returns delta_e_scaled : [66] tensor.
    """
    fft_full = scipy.fft.rfft(sig[:lookback, :], n=lookback, axis=0)
    bins = fft_bins.numpy() if torch.is_tensor(fft_bins) else np.asarray(fft_bins)
    amps = torch.from_numpy(np.abs(fft_full[bins, :]).astype(np.float32)).T  # [66, 256]
    norm_amps = (amps - amp_means) / amp_stds
    cur_energy = torch.abs(norm_amps)
    delta_e = (cur_energy - avg_baseline).mean(dim=-1).clamp(min=0)  # [66]
    scale = avg_baseline.mean().clamp(min=1e-6)
    return delta_e / scale


# =============================================================================
#  WaveGraphNet (inverse + forward branches): raw-signal-derived dataset
# =============================================================================
class CoupledModelDataset(TorchDataset):
    """
    Produces everything both the inverse graph (raw 515-dim per-path signal,
    for the pre-existing raw-signal ablation) and the forward branch (via
    `delta_e_full`) need. `DeltaEOnlyInverseGNN` (the proposed inverse model)
    only ever reads `delta_e_full` and `y_true` from this dict — see
    `train_inverse.py`.
    """

    def __init__(self, data_map, sample_id_list,
                inv_static_edge_index, inv_edge_feature_col_idxs,
                fwd_propagation_col_idxs, fixed_fft_bin_indices,
                amp_means, amp_stds, lookback_fft,
                average_baseline_energy_profile, global_max_delta_e):
        self.data_map = data_map
        self.sample_id_list = sample_id_list
        self.node_coords = torch.tensor(
            np.array([TRANSDUCER_COORDS[i + 1] for i in range(12)]), dtype=torch.float)
        self.inv_edge_index = inv_static_edge_index
        self.inv_col_idxs = torch.as_tensor(inv_edge_feature_col_idxs, dtype=torch.long)
        self.lookback_fft = lookback_fft
        self.fft_bins = torch.as_tensor(fixed_fft_bin_indices, dtype=torch.long)
        self.amp_means = torch.tensor(amp_means, dtype=torch.float32)
        self.amp_stds  = torch.tensor(amp_stds,  dtype=torch.float32)
        self.global_max_delta_e = max(float(global_max_delta_e), 1e-6)
        self.prop_pair_indices = torch.tensor(sorted(set(fwd_propagation_col_idxs)), dtype=torch.long)
        self.avg_baseline = average_baseline_energy_profile  # [66, 256]

    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, idx):
        sample_id = self.sample_id_list[idx]
        sig = torch.from_numpy(self.data_map[sample_id]).float()
        num_pairs = sig.shape[1]

        fft_full = scipy.fft.rfft(sig[:self.lookback_fft, :].numpy(), n=self.lookback_fft, axis=0)
        bins = self.fft_bins.numpy()
        amps   = torch.from_numpy(np.abs(fft_full[bins, :]).astype(np.float32)).T    # [66, 256]
        phases = torch.from_numpy(np.angle(fft_full[bins, :]).astype(np.float32)).T  # [66, 256]

        norm_amps = (amps - self.amp_means) / self.amp_stds
        full_freq_profile = torch.stack([norm_amps, phases], dim=-1)   # [66, 256, 2]
        flat = full_freq_profile.view(num_pairs, -1)                    # [66, 512]

        row, col = self.inv_edge_index
        dist = (self.node_coords[row] - self.node_coords[col]).norm(dim=-1, keepdim=True)
        vec  = self.node_coords[row] - self.node_coords[col]
        sp   = torch.cat([dist, vec], dim=1)       # [132, 3]
        freq = flat[self.inv_col_idxs]              # [132, 512]
        edge_attr_inv = torch.cat([sp, freq], dim=1)  # [132, 515]

        data_inv = PyGData(x=self.node_coords, edge_index=self.inv_edge_index, edge_attr=edge_attr_inv)

        xd, yd = _target_xy(sample_id)
        y_true = torch.tensor([[xd, yd]], dtype=torch.float)

        cur_energy   = torch.abs(norm_amps)                                       # [66, 256]
        delta_e_full = (cur_energy - self.avg_baseline).mean(dim=-1).clamp(min=0)  # [66], ALL paths
        delta_e_prop = delta_e_full[self.prop_pair_indices]
        delta_e_norm = delta_e_prop / self.global_max_delta_e

        return {
            "data_inv": data_inv,
            "delta_e_true": delta_e_norm,   # normalized, subset — legacy raw-signal-model target
            "delta_e_full": delta_e_full,   # all 66 paths, raw scale — used by DeltaEOnlyInverseGNN/RAPID
            "y_true": y_true,
            "sample_id": sample_id,
        }


# =============================================================================
#  1D-CNN baseline datasets
# =============================================================================
class Cnn1DDataset(TorchDataset):
    """Raw-input variant: [num_pairs*2, num_freqs], even channels = amplitude, odd = phase."""

    def __init__(self, data_map, sample_id_list, bins, amp_means, amp_stds, lookback):
        self.data_map, self.sample_ids = data_map, sample_id_list
        self.lookback, self.bins = lookback, bins
        self.amp_means = torch.tensor(amp_means, dtype=torch.float32)
        self.amp_stds  = torch.tensor(amp_stds,  dtype=torch.float32)
        self.nf = len(bins)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        sig = self.data_map[sid]
        num_pairs = sig.shape[1]

        ff     = scipy.fft.rfft(sig[:self.lookback, :], n=self.lookback, axis=0)
        amps   = torch.from_numpy(np.abs(ff[self.bins, :]).astype(np.float32)).T
        phases = torch.from_numpy(np.angle(ff[self.bins, :]).astype(np.float32)).T
        norm_amps = (amps - self.amp_means) / self.amp_stds

        x = torch.zeros((num_pairs * 2, self.nf), dtype=torch.float32)
        x[0::2] = norm_amps
        x[1::2] = phases

        xd, yd = _target_xy(sid)
        return x, torch.tensor([xd, yd], dtype=torch.float)


class Cnn1DDeltaEDataset(TorchDataset):
    """
    ΔE-input fairness ablation for the 1D-CNN baseline: same architecture,
    input is the 66-scalar ΔE vector instead of the 515-dim raw signal.
    Returned x is [1, 66] (single channel).
    """

    def __init__(self, data_map, sample_id_list, bins, amp_means, amp_stds,
                lookback, average_baseline_energy_profile):
        self.data_map, self.sample_ids = data_map, sample_id_list
        self.lookback, self.bins = lookback, bins
        self.amp_means = torch.tensor(amp_means, dtype=torch.float32)
        self.amp_stds  = torch.tensor(amp_stds,  dtype=torch.float32)
        self.avg_baseline = average_baseline_energy_profile

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        sig = torch.from_numpy(self.data_map[sid]).float()
        de = compute_delta_e_66(sig, self.lookback, self.bins, self.amp_means,
                                self.amp_stds, self.avg_baseline)
        xd, yd = _target_xy(sid)
        return de.unsqueeze(0), torch.tensor([xd, yd], dtype=torch.float)


# =============================================================================
#  LSTM baseline datasets
# =============================================================================
class LstmDataset(TorchDataset):
    """Raw-input variant: [66, 256, 2] (pairs, freqs, [norm_amp, phase])."""

    def __init__(self, data_map, sample_id_list, bins, amp_means, amp_stds, lookback):
        self.data_map, self.sample_ids = data_map, sample_id_list
        self.lookback, self.bins = lookback, bins
        self.amp_means = torch.tensor(amp_means, dtype=torch.float32)
        self.amp_stds  = torch.tensor(amp_stds,  dtype=torch.float32)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        sig = self.data_map[sid]
        ff     = scipy.fft.rfft(sig[:self.lookback, :], n=self.lookback, axis=0)
        amps   = torch.from_numpy(np.abs(ff[self.bins, :]).astype(np.float32)).T
        phases = torch.from_numpy(np.angle(ff[self.bins, :]).astype(np.float32)).T
        norm_amps = (amps - self.amp_means) / self.amp_stds
        x = torch.stack([norm_amps, phases], dim=-1)

        xd, yd = _target_xy(sid)
        return x, torch.tensor([xd, yd], dtype=torch.float)


class LstmDeltaEDataset(TorchDataset):
    """
    ΔE-input fairness ablation for the LSTM baseline: the 66 ΔE scalars
    become a single 66-timestep sequence (1 feature each), fed to
    `models.baselines.lstm.LSTM_baseline_deltae`.
    """

    def __init__(self, data_map, sample_id_list, bins, amp_means, amp_stds,
                lookback, average_baseline_energy_profile):
        self.data_map, self.sample_ids = data_map, sample_id_list
        self.lookback, self.bins = lookback, bins
        self.amp_means = torch.tensor(amp_means, dtype=torch.float32)
        self.amp_stds  = torch.tensor(amp_stds,  dtype=torch.float32)
        self.avg_baseline = average_baseline_energy_profile

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        sig = torch.from_numpy(self.data_map[sid]).float()
        de = compute_delta_e_66(sig, self.lookback, self.bins, self.amp_means,
                                self.amp_stds, self.avg_baseline)
        xd, yd = _target_xy(sid)
        return de.unsqueeze(-1), torch.tensor([xd, yd], dtype=torch.float)  # [66, 1]


# =============================================================================
#  GNN-MLP / GAT baseline datasets
# =============================================================================
class StandardGraphDataset(TorchDataset):
    """Raw-input variant: edge_attr = [dist, vec, 512-dim raw per-frequency descriptor]."""

    def __init__(self, data_map, sample_id_list, static_edge_index, edge_feature_col_idxs,
                fixed_fft_bin_indices, amp_means, amp_stds, lookback_fft):
        self.data_map, self.sample_ids = data_map, sample_id_list
        self.node_coords = torch.tensor(
            np.array([TRANSDUCER_COORDS[i + 1] for i in range(12)]), dtype=torch.float)
        self.edge_index = static_edge_index
        self.col_idxs = torch.as_tensor(edge_feature_col_idxs, dtype=torch.long)
        self.lookback = lookback_fft
        self.fft_bins = torch.as_tensor(fixed_fft_bin_indices, dtype=torch.long)
        self.amp_means = torch.tensor(amp_means, dtype=torch.float32)
        self.amp_stds  = torch.tensor(amp_stds,  dtype=torch.float32)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        sig = torch.from_numpy(self.data_map[sample_id]).float()
        num_pairs = sig.shape[1]

        fft_full = scipy.fft.rfft(sig[:self.lookback, :].numpy(), n=self.lookback, axis=0)
        bins = self.fft_bins.numpy()
        amps   = torch.from_numpy(np.abs(fft_full[bins, :]).astype(np.float32)).T
        phases = torch.from_numpy(np.angle(fft_full[bins, :]).astype(np.float32)).T
        norm_amps = (amps - self.amp_means) / self.amp_stds
        flat = torch.stack([norm_amps, phases], dim=-1).view(num_pairs, -1)  # [66, 512]

        row, col = self.edge_index
        dist = (self.node_coords[row] - self.node_coords[col]).norm(dim=-1, keepdim=True)
        vec  = self.node_coords[row] - self.node_coords[col]
        sp   = torch.cat([dist, vec], dim=1)
        edge_attr = torch.cat([sp, flat[self.col_idxs]], dim=1)

        data = PyGData(x=self.node_coords, edge_index=self.edge_index, edge_attr=edge_attr)
        xd, yd = _target_xy(sample_id)
        data.y = torch.tensor([[xd, yd]], dtype=torch.float)
        return data


class StandardGraphDeltaEDataset(TorchDataset):
    """
    ΔE-input fairness ablation for GNN-MLP/GAT: edge_attr = [dist, vec,
    ΔE_edge] (4-dim) instead of the 512-dim raw descriptor. NOTE: the
    "GNN-MLP" vs "GAT" distinction is only in the edge encoder (attention
    over 256 frequency bins vs a plain MLP), which is meaningless with a
    single ΔE scalar — both collapse to the same model here, so only one
    "GNN (delta_e)" run is needed, not two.
    """

    def __init__(self, data_map, sample_id_list, static_edge_index, fixed_fft_bin_indices,
                amp_means, amp_stds, lookback_fft, average_baseline_energy_profile,
                edge_feature_col_idxs):
        self.data_map, self.sample_ids = data_map, sample_id_list
        self.node_coords = torch.tensor(
            np.array([TRANSDUCER_COORDS[i + 1] for i in range(12)]), dtype=torch.float)
        self.edge_index = static_edge_index
        self.col_idxs = torch.as_tensor(edge_feature_col_idxs, dtype=torch.long)
        self.lookback = lookback_fft
        self.fft_bins = torch.as_tensor(fixed_fft_bin_indices, dtype=torch.long)
        self.amp_means = torch.tensor(amp_means, dtype=torch.float32)
        self.amp_stds  = torch.tensor(amp_stds,  dtype=torch.float32)
        self.avg_baseline = average_baseline_energy_profile

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        sig = torch.from_numpy(self.data_map[sample_id]).float()
        de = compute_delta_e_66(sig, self.lookback, self.fft_bins, self.amp_means,
                                self.amp_stds, self.avg_baseline)  # [66]

        row, col = self.edge_index
        dist = (self.node_coords[row] - self.node_coords[col]).norm(dim=-1, keepdim=True)
        vec  = self.node_coords[row] - self.node_coords[col]
        sp   = torch.cat([dist, vec], dim=1)                    # [132, 3]
        de_edge = de[self.col_idxs].unsqueeze(-1)                 # [132, 1]
        edge_attr = torch.cat([sp, de_edge], dim=1)               # [132, 4]

        data = PyGData(x=self.node_coords, edge_index=self.edge_index, edge_attr=edge_attr)
        xd, yd = _target_xy(sample_id)
        data.y = torch.tensor([[xd, yd]], dtype=torch.float)
        return data
