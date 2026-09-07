# models/baselines/rapid.py
"""
RAPID: a training-free, classical elliptical-imaging localization baseline
(Reconstruction Algorithm for Probabilistic Inspection of Damage).

No parameters are learned from data. The only two free knobs -- the
ellipse half-width `beta` and a peak-intensity detection threshold -- are
selected by grid search on the validation split only, exactly like every
learned model's hyperparameters in this study. Included to answer the
central question a suite of only-learned baselines cannot: does the
localization error reported for the unseen zone reflect the intrinsic
difficulty of the extrapolation task, or a weakness specific to the
learned models?

Reuses the SAME path-wise energy-deviation quantity ΔE_ij every other
model in this repository consumes, computed over the full 66-path set
(see `compute_full_delta_e`) -- so the comparison against learned models
differs only in *how* ΔE is turned into a coordinate (hand-designed
elliptical accumulation here, learned message passing elsewhere), not in
the input representation itself.
"""

import itertools

import numpy as np
import scipy.fft

NUM_TRANSDUCERS = 12
ALL_PAIRS = list(itertools.combinations(range(NUM_TRANSDUCERS), 2))  # 66 pairs, same order as everywhere else


def compute_full_delta_e(sig, amp_means, amp_stds, avg_baseline, fft_bins, lookback):
    """
    Per-sample energy-deviation index over ALL 66 measured paths. Mirrors
    `data.datasets.CoupledModelDataset`'s ΔE formula exactly, minus the
    [0,1]-normalization by a global constant (not needed here -- RAPID
    only needs a consistent per-plate scale, not a normalized target).

    sig : np.ndarray [T, 66] normalized differential signal
    Returns delta_e : np.ndarray [66]
    """
    fft_full = scipy.fft.rfft(sig[:lookback, :], n=lookback, axis=0)
    amps = np.abs(fft_full[fft_bins, :]).T  # [66, 256]
    norm_amps = (amps - amp_means) / amp_stds
    cur_energy = np.abs(norm_amps)
    delta_e = (cur_energy - avg_baseline).mean(axis=-1)
    return np.clip(delta_e, a_min=0.0, a_max=None)  # [66]


def build_grid(grid_res: int):
    xs = np.linspace(0.0, 1.0, grid_res)
    ys = np.linspace(0.0, 1.0, grid_res)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    return gx, gy  # each [grid_res, grid_res]


def rapid_image(delta_e_66, transducer_coords, beta, gx, gy, pairs=None):
    """
    Accumulates weighted evidence over a [0,1]^2 grid:
        R_ij(x,y) = (||(x,y)-r_i|| + ||(x,y)-r_j||) / ||r_i-r_j||
        W_ij(x,y) = max(0, (beta - R_ij(x,y)) / (beta - 1))
        Image(x,y) = sum_ij DeltaE_ij * W_ij(x,y)

    transducer_coords : [N, 2] -- N=12 by default, fewer for a reduced-sensor
                        graph. Must be indexed consistently with `pairs`.
    pairs             : list of (i, j) local index tuples, defaults to
                        ALL_PAIRS. For a reduced sensor set, pass the
                        re-indexed 0..N-1 local pairs (see
                        `data.precompute.get_reduced_sensor_mapping`) and
                        pre-filter `delta_e_66` to match.
    """
    if pairs is None:
        pairs = ALL_PAIRS
    image = np.zeros_like(gx)
    for col, (i, j) in enumerate(pairs):
        ri, rj = transducer_coords[i], transducer_coords[j]
        dij = np.linalg.norm(ri - rj)
        d_pi = np.sqrt((gx - ri[0]) ** 2 + (gy - ri[1]) ** 2)
        d_pj = np.sqrt((gx - rj[0]) ** 2 + (gy - rj[1]) ** 2)
        r_ij = (d_pi + d_pj) / dij
        w_ij = np.clip((beta - r_ij) / (beta - 1.0), a_min=0.0, a_max=None)
        image += delta_e_66[col] * w_ij
    return image


def rapid_predict(delta_e_66, transducer_coords, beta, gx, gy, top_frac=0.05, pairs=None):
    """
    Returns (pred_xy, peak_intensity). `pred_xy` is the intensity-weighted
    centroid over grid cells with intensity >= (1-top_frac)*peak -- more
    stable than a raw argmax at finite grid resolution.
    """
    image = rapid_image(delta_e_66, transducer_coords, beta, gx, gy, pairs=pairs)
    peak = float(image.max())
    if peak <= 0:
        return np.array([0.5, 0.5]), 0.0
    mask = image >= (peak * (1.0 - top_frac))
    w, x, y = image[mask], gx[mask], gy[mask]
    return np.array([(w * x).sum() / w.sum(), (w * y).sum() / w.sum()]), peak
