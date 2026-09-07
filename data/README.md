# Data

This benchmark uses the **OGW-1 (Open Guided Waves) SHM Plate** dataset — a
public composite-plate guided-wave benchmark with 12 surface-bonded
piezoelectric transducers, 28 distinct single-defect damage locations (7
physical clusters of 4 points each), and pristine/undamaged baseline
recordings.

## Expected input file

Every script in this repository loads a single pickle file:

```
data/processed/ogw_data.pkl
```

This is a Python `dict` mapping a **sample id** (`str`) to a **raw signal
array** (`np.ndarray`, shape `[T, 66]`, one column per measured pitch-catch
path, `T` = number of time samples):

- Damaged samples: id format `"<D-label>_<excitation>"`, e.g. `"D13_100kHz"`.
  The D-label (`D1`-`D28`) identifies which of the 28 physical damage
  locations was used (see `datasets.DAMAGE_LABELS` for their normalized
  `[x,y] ∈ [0,1]²` coordinates).
- Undamaged/pristine samples: id contains the substring `"baseline"`, e.g.
  `"baseline_3"`.
- Two ids encode **multiple simultaneous defects** (e.g.
  `"D25_D28_100kHz"`) — these are automatically excluded by
  `splits._is_single_defect`, matching the paper's stated dataset
  construction.

Column order follows `sorted(itertools.combinations(range(12), 2))` over
0-indexed transducer ids — i.e. column `k` corresponds to the `k`-th pair in
that sorted list. Transducer coordinates (`datasets.TRANSDUCER_COORDS`) use
1-indexed ids 1-12.

If you don't already have `ogw_data.pkl`, build it from the raw OGW-1
release (time-synchronized per-path pitch-catch recordings) by assembling
one array per sample in the format above. This repository's pipeline starts
from that pickle — the raw-instrument-file-to-pickle conversion is
dataset-acquisition-specific and is not part of this repository.

## Everything downstream is derived, not hand-built

You never need to construct anything else by hand. `precompute.py` derives
every normalization statistic (baseline subtraction, per-path frequency
statistics, the global ΔE-normalization constant) from the **training split
only**, and `splits.py` deterministically assigns every sample id to
train/val/test. Both are called once per (split, seed) at the top of every
`train_*.py` / `evaluate_*.py` script — see the top-level README for the
exact call sequence.

## Splits at a glance

| Paper name | Code (`--split`) | What it tests |
|---|---|---|
| Split A | `A` | Single unseen damage cluster held out. |
| Split B | `B` | Two unseen clusters (both outer plate corners). |
| Split C | `B2` | Cross-region generalization: train/val/test are three disjoint plate regions. |

See the docstring at the top of `splits.py` for the full spatial-leakage
rationale (why splits assign *whole* damage clusters, never individual
labels, to a single role) and for the two auxiliary splits (`A2`, `C`) not
used in the paper's headline results.
