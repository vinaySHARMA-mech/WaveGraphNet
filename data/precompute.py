# data/precompute.py
"""
Per-split preprocessing statistics and graph-topology helpers.

`build_all_stats(raw_data_map, train_ids)` is the single entry point every
training/evaluation script calls once, immediately after computing a split's
train/val/test ids. It derives every normalization statistic from the
TRAINING split only (never val/test), then returns a dict of everything
downstream code needs (normalized signals, per-path frequency statistics,
graph topology, etc.).

Preprocessing steps (mirrors the original dataset-construction notebook):
  Step 0: Baseline subtraction — average undamaged (pristine) signal over
          training baselines, subtracted from every sample.
  Step 1: Time-domain z-normalization of the baseline-subtracted signal,
          using training-set mean/std.
  Step 2: Per-(path, frequency) amplitude statistics (mean/std), used to
          normalize each path's spectral amplitude before computing ΔE.
  Step 3: Average baseline energy profile — the fixed, per-path "what
          energy looks like with no damage" reference ΔE is measured against.
  Step 4: A global scalar (max ΔE over training) used to put ΔE on the
          same normalized scale as the model's target space.
"""

import itertools
import numpy as np
import scipy.fft
import torch
from tqdm import tqdm

# ── Physical / signal-processing constants ───────────────────────────────────
LOOKBACK_POINTS   = 13_108
SAMPLING_RATE     = 10_000_000
MIN_FREQ_HZ       = 69_400
MAX_FREQ_HZ       = 128_000
N_ATTENTION_FREQS = 256
NUM_TRANSDUCERS   = 12
NUM_DATA_COLUMNS  = 66   # C(12,2) measured pitch-catch paths


def _compute_fixed_fft_bin_indices():
    freqs   = scipy.fft.rfftfreq(LOOKBACK_POINTS, d=1.0 / SAMPLING_RATE)
    targets = np.linspace(MIN_FREQ_HZ, MAX_FREQ_HZ, N_ATTENTION_FREQS)
    return np.array([np.argmin(np.abs(freqs - f)) for f in targets], dtype=np.int64)


FIXED_FFT_BIN_INDICES = _compute_fixed_fft_bin_indices()


# ── Step 0 + 1: time-domain preprocessing ────────────────────────────────────
def build_normalized_differential_data(raw_data_map: dict, train_ids: list):
    """
    Returns (normalized_differential_data, average_raw_baseline, diff_mean,
    diff_std). `normalized_differential_data` has the same keys as
    `raw_data_map`; every other split/dataset script only ever consumes this
    dict, never the raw signals directly.
    """
    baseline_train_ids = [s for s in train_ids if "baseline" in s]

    min_len = min(raw_data_map[s].shape[0] for s in baseline_train_ids)
    stacked = np.stack([raw_data_map[s][:min_len] for s in baseline_train_ids], axis=0)
    average_raw_baseline = stacked.mean(axis=0).astype(np.float32)

    differential_data = {}
    for sid, sig in raw_data_map.items():
        L = min(sig.shape[0], average_raw_baseline.shape[0])
        differential_data[sid] = sig[:L] - average_raw_baseline[:L]

    all_train_diff = np.concatenate([differential_data[s] for s in train_ids], axis=0)
    diff_mean = all_train_diff.mean(axis=0, keepdims=True).astype(np.float32)
    diff_std  = all_train_diff.std(axis=0, keepdims=True).astype(np.float32)
    diff_std[diff_std == 0] = 1e-6

    normalized_differential_data = {
        sid: (sig - diff_mean) / diff_std for sid, sig in differential_data.items()
    }
    return normalized_differential_data, average_raw_baseline, diff_mean, diff_std


# ── Step 2: per-(path, frequency) amplitude statistics ───────────────────────
def compute_amp_stats(normalized_data_map, train_ids,
                      fixed_fft_bin_indices=FIXED_FFT_BIN_INDICES,
                      lookback=LOOKBACK_POINTS):
    """Returns (amp_means, amp_stds), both shape [66, 256]."""
    n_pairs, n_freqs = NUM_DATA_COLUMNS, len(fixed_fft_bin_indices)
    amps_list = [[[] for _ in range(n_freqs)] for _ in range(n_pairs)]

    for sid in tqdm(train_ids, desc="Computing amp stats", leave=False):
        sig = normalized_data_map[sid]
        for pj in range(n_pairs):
            window = sig[:lookback, pj]
            if len(window) < lookback:
                continue
            fft_c = scipy.fft.rfft(window, n=lookback)
            amps_at_bins = np.abs(fft_c[fixed_fft_bin_indices])
            for fi in range(n_freqs):
                amps_list[pj][fi].append(float(amps_at_bins[fi]))

    amp_means = np.zeros((n_pairs, n_freqs), dtype=np.float32)
    amp_stds  = np.ones((n_pairs, n_freqs), dtype=np.float32)
    for pj in range(n_pairs):
        for fi in range(n_freqs):
            vals = amps_list[pj][fi]
            if vals:
                amp_means[pj, fi] = float(np.mean(vals))
                amp_stds[pj, fi]  = max(float(np.std(vals)) if len(vals) > 1 else 1.0, 1e-6)
    return amp_means, amp_stds


# ── Step 3: average baseline energy profile ──────────────────────────────────
def compute_baseline_energy_profile(normalized_data_map, train_ids, amp_means, amp_stds,
                                    fixed_fft_bin_indices=FIXED_FFT_BIN_INDICES,
                                    lookback=LOOKBACK_POINTS) -> torch.Tensor:
    """Returns tensor [66, 256]."""
    baseline_ids = [s for s in train_ids if "baseline" in s]
    amp_m, amp_s = torch.from_numpy(amp_means), torch.from_numpy(amp_stds)
    profiles = []
    for sid in tqdm(baseline_ids, desc="Baseline energy profile", leave=False):
        sig  = torch.from_numpy(normalized_data_map[sid]).float()
        fftc = torch.fft.rfft(sig[:lookback], n=lookback, dim=0)
        amps = torch.abs(fftc[fixed_fft_bin_indices]).T       # [66, 256]
        profiles.append(torch.abs((amps - amp_m) / amp_s))
    if profiles:
        return torch.stack(profiles, dim=0).mean(dim=0)
    return torch.zeros(NUM_DATA_COLUMNS, len(fixed_fft_bin_indices))


# ── Step 4: global max ΔE (normalization constant) ───────────────────────────
def compute_global_max_delta_e(normalized_data_map, train_ids, amp_means, amp_stds,
                               average_baseline_energy_profile, propagation_pair_indices,
                               fixed_fft_bin_indices=FIXED_FFT_BIN_INDICES,
                               lookback=LOOKBACK_POINTS) -> float:
    amp_m, amp_s = torch.from_numpy(amp_means), torch.from_numpy(amp_stds)
    max_val = 0.0
    for sid in tqdm(train_ids, desc="Computing global max ΔE", leave=False):
        sig  = torch.from_numpy(normalized_data_map[sid]).float()
        fftc = torch.fft.rfft(sig[:lookback], n=lookback, dim=0)
        amps = torch.abs(fftc[fixed_fft_bin_indices]).T
        norm_amps = (amps - amp_m) / amp_s
        delta_e = (torch.abs(norm_amps) - average_baseline_energy_profile).mean(dim=-1).clamp(min=0)
        max_val = max(max_val, delta_e[propagation_pair_indices].max().item())
    return max(max_val, 1e-6)


# ── Graph-topology helpers ────────────────────────────────────────────────────
def get_all_paths_edge_index_and_col_idxs(num_transducers: int = NUM_TRANSDUCERS):
    """
    ALL C(num_transducers, 2) measured paths (132 directed edges for the
    full 12-sensor case) — used by the forward branch, which predicts ΔE
    for every measured path, not a subset.

    Returns (prop_edge_index [2, 2*n_pairs], col_idxs [list], unique_cols
    [tensor of length n_pairs]).
    """
    all_pairs   = sorted(itertools.combinations(range(num_transducers), 2))
    pair_to_col = {p: i for i, p in enumerate(all_pairs)}

    edges, col_idxs = [], []
    for (i, j) in all_pairs:
        for u, v in [(i, j), (j, i)]:
            edges.append([u, v])
            col_idxs.append(pair_to_col[(i, j)])

    prop_ei     = torch.tensor(edges, dtype=torch.long).t().contiguous()
    unique_cols = torch.tensor(sorted(set(col_idxs)), dtype=torch.long)
    return prop_ei, col_idxs, unique_cols


def get_reduced_sensor_mapping(removed_sensors: set, num_transducers: int = NUM_TRANSDUCERS):
    """
    Maps a set of REMOVED transducer IDs (1-indexed, matching
    data.splits.REMOVED_SENSORS) to the surviving 0-indexed sensor/path
    structure needed to build a reduced graph (used by the reduced-sensor
    split "C").

    Returns (keep_idxs, keep_cols):
      keep_idxs : sorted list of surviving 0-indexed transducer indices.
                  `coords_12[keep_idxs]` gives the reduced [N,2] coords.
      keep_cols : original 66-length column indices (sorted-combinations
                  ordering) for paths whose BOTH endpoints survive.
                  `delta_e_66[keep_cols]` lines up exactly with a fresh
                  `canonical_pairs(num_transducers=len(keep_idxs))` call
                  over the re-indexed 0..len(keep_idxs)-1 local sensor ids.
    """
    if not removed_sensors:
        all_pairs = sorted(itertools.combinations(range(num_transducers), 2))
        return list(range(num_transducers)), list(range(len(all_pairs)))
    removed_0idx = {s - 1 for s in removed_sensors}
    keep_idxs = sorted(set(range(num_transducers)) - removed_0idx)
    keep_set  = set(keep_idxs)
    all_pairs = sorted(itertools.combinations(range(num_transducers), 2))
    keep_cols = [idx for idx, (i, j) in enumerate(all_pairs) if i in keep_set and j in keep_set]
    return keep_idxs, keep_cols


def get_k_graph_edge_index(num_nodes, self_loops=False):
    """Fully-connected directed graph over `num_nodes` (both (u,v) and (v,u) for every pair)."""
    edges = list(itertools.combinations(range(num_nodes), 2))
    edge_list = []
    for u, v in edges:
        edge_list.append([u, v]); edge_list.append([v, u])
    if self_loops:
        edge_list += [[i, i] for i in range(num_nodes)]
    if not edge_list:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()


def get_inv_edge_feature_col_idxs(static_edge_index: torch.Tensor,
                                  num_transducers: int = NUM_TRANSDUCERS) -> np.ndarray:
    """Maps every directed edge (u,v) in `static_edge_index` -> data column index of pair {u,v}."""
    all_pairs   = list(itertools.combinations(range(num_transducers), 2))
    pair_to_col = {p: i for i, p in enumerate(all_pairs)}
    return np.array(
        [pair_to_col[tuple(sorted((static_edge_index[0, i].item(), static_edge_index[1, i].item())))]
         for i in range(static_edge_index.shape[1])],
        dtype=np.int64,
    )


# ── Master build function ────────────────────────────────────────────────────
def build_all_stats(raw_data_map: dict, train_ids: list) -> dict:
    """
    Build every statistic needed for a given split, from the training split
    only. Call once per (split, seed) at the top of each training/evaluation
    script. Always use `stats["normalized_data_map"]` (never raw signals
    directly) when constructing datasets.
    """
    print("- Step 0+1: Baseline subtraction + time-domain z-normalization")
    norm_data, _, _, _ = build_normalized_differential_data(raw_data_map, train_ids)

    print("- Step 2: Per-(path, frequency) amplitude statistics")
    amp_means, amp_stds = compute_amp_stats(norm_data, train_ids)

    print("- Step 3: Average baseline energy profile")
    avg_bl = compute_baseline_energy_profile(norm_data, train_ids, amp_means, amp_stds)

    print("- Building path edge index (all 66 measured paths)")
    prop_ei, prop_col_idxs, prop_unique = get_all_paths_edge_index_and_col_idxs()

    print("- Step 4: Global max ΔE (normalization constant)")
    g_max = compute_global_max_delta_e(norm_data, train_ids, amp_means, amp_stds,
                                       avg_bl, prop_unique)
    print(f"  global_max_delta_e = {g_max:.6f}")

    k12 = get_k_graph_edge_index(NUM_TRANSDUCERS, self_loops=False)
    inv_col_idxs = get_inv_edge_feature_col_idxs(k12)

    return dict(
        normalized_data_map             = norm_data,
        fixed_fft_bin_indices           = FIXED_FFT_BIN_INDICES,
        amp_means                       = amp_means,   # [66, 256]
        amp_stds                        = amp_stds,    # [66, 256]
        average_baseline_energy_profile = avg_bl,       # [66, 256]
        global_max_delta_e              = g_max,
        propagation_edge_index          = prop_ei,      # [2, 132]
        propagation_col_idxs            = prop_col_idxs,
        propagation_pair_indices        = prop_unique,   # [66]
        k12_edge_index                  = k12,           # [2, 132]
        inv_edge_feature_col_idxs       = inv_col_idxs,
        lookback_fft                    = LOOKBACK_POINTS,
        num_fft_bins                    = N_ATTENTION_FREQS,
    )
