# WaveGraphNet

Reference implementation for **WaveGraphNet**: an inverse-forward graph
framework for sparse guided-wave damage localization on the OGW-1 SHM Plate
benchmark. This repository is a cleaned, reorganized packaging of the exact
code used to produce every number in the paper — same models, same
training procedure, same evaluation protocol, just laid out for readability.

## What's in this repository

- **The inverse localization branch** (`models/wavegraphnet_inverse.py`):
  predicts a defect coordinate directly from the measured per-path
  energy-deviation vector ΔE.
- **The forward consistency branch** (`models/wavegraphnet_forward.py`):
  predicts the path-wise ΔE a candidate coordinate would produce. Trained
  completely independently of the inverse branch; used only at inference,
  inside test-time refinement. Each measured path is represented by both of
  its directed edges, whose features are embedded by a shared encoder and
  averaged, so the edge embedding is invariant to which transducer is
  listed first.
- **Test-time refinement** (`evaluate_refinement.py`): given the inverse
  branch's one-shot prediction as a warm start, takes gradient steps
  directly on the coordinate against the frozen forward branch.
- **Four learned baselines** (`models/baselines/`: 1D-CNN, LSTM, GNN-MLP,
  GAT), each runnable on either the raw per-frequency spectral signal or
  the same ΔE representation WaveGraphNet uses (a fairness ablation).
- **RAPID** (`models/baselines/rapid.py`): a training-free classical
  elliptical-imaging baseline.
- **Data pipeline** (`data/`): spatial train/val/test splits, all
  preprocessing/normalization statistics, and every `Dataset` class.

## Repository layout

```
data/
  README.md          # OGW-1 dataset format and expected pickle layout
  splits.py          # train/val/test split definitions (see paper-name <-> code-key table below)
  precompute.py       # per-split normalization statistics + graph-topology helpers
  datasets.py         # PyTorch Dataset classes (raw-signal and ΔE-input variants) for every model

models/
  wavegraphnet_inverse.py   # DeltaEOnlyInverseGNN -- the proposed inverse branch
  wavegraphnet_forward.py   # DirectPathAttenuationGNN -- the proposed forward branch
  layers.py                 # shared building blocks for the graph baselines
  baselines/
    cnn1d.py                # 1D-CNN baseline
    lstm.py                 # LSTM baseline (+ ΔE-input variant)
    gnn.py                  # GNN-MLP / GAT baselines (FlexibleGNN)
    rapid.py                 # RAPID training-free baseline

utils/
  checkpointer.py     # checkpoint save/load conventions
  logger.py           # result logging + evaluation helpers
  forward_utils.py     # shared helpers for the forward branch's graph batches

train_inverse.py            # train the inverse branch standalone
train_forward.py            # train the forward branch standalone
evaluate_refinement.py       # standalone MAE, test-time refinement, and FPR for WaveGraphNet
evaluate_cluster_breakdown.py # per-cluster MAE breakdown for the held-out clusters
train_baseline_cnn.py         # 1D-CNN baseline
train_baseline_lstm.py        # LSTM baseline
train_baseline_gnn.py         # GNN-MLP / GAT baselines
run_rapid.py                  # RAPID baseline (no training)
```

## Setup

```bash
pip install -r requirements.txt
```

Place the dataset pickle at `data/processed/ogw_data.pkl` — see
[`data/README.md`](data/README.md) for the exact expected format.

## Pre-trained checkpoints

The trained checkpoints used to produce every number in the paper are archived
on Zenodo:

**https://doi.org/10.5281/zenodo.22639832**

Download and extract them into the root of this repository:

```bash
wget https://zenodo.org/records/22639832/files/wavegraphnet-checkpoints.tar.gz
tar xzf wavegraphnet-checkpoints.tar.gz
```

This creates `checkpoints/<split>/`, containing 18 files — the inverse branch and
the forward branch, for each of the 3 splits (`A`, `B`, `B2`) and each of the
3 seeds (`0`, `1`, `42`):

```
checkpoints/A/WaveGraphNet__DeltaE_Only_Inverse_seed0.pt
checkpoints/A/WaveGraphNet__Forward_seed0.pt
...
```

With these in place you can reproduce the reported results **without retraining** —
skip straight to the evaluation commands below. For example:

```bash
python evaluate_refinement.py --split A --seeds 0 1 42
```

should print a standalone MAE of `127.1mm`, a refined MAE of `40.8mm`, and a
false-positive rate of `0.0%`.

Note that evaluation still requires the dataset (`data/processed/ogw_data.pkl`);
the checkpoints only remove the need to train.

## Splits: paper name -> code

| Paper name | `--split` value | What it tests |
|---|---|---|
| Split A | `A`  | Single unseen damage cluster held out. |
| Split B | `B`  | Two unseen clusters (both outer plate corners). |
| Split C | `B2` | Cross-region generalization (train/val/test are three disjoint plate regions) — the hardest setting. |

(Two auxiliary splits, `A2` and `C`, also exist for robustness checks not
used in the paper's headline results — see `data/splits.py`'s module
docstring. Note internal split `"C"` is **not** the paper's "Split C" —
don't confuse the two.)

Every script accepts `--split` with one of these values and `--seed`. The
paper's results use seeds `{0, 1, 42}` for every learned model (RAPID has
no seeds — it is training-free and deterministic given its validation-selected
hyperparameters).

## Reproducibility

`train_inverse.py`, `train_forward.py`, and the three `train_baseline_*.py`
scripts all call `torch.use_deterministic_algorithms(True)` (plus set
`CUBLAS_WORKSPACE_CONFIG`) in `set_seed()`. This matters more than it looks:
the message-passing layers use `index_add_`-based scatter aggregation,
which on CUDA is **not** deterministic by default — the order floating-point
contributions get summed in varies run to run, even with an identical seed.
`torch.backends.cudnn.deterministic = True` does **not** cover this; it only
fixes cuDNN's convolution algorithm choice.

This was confirmed directly: two runs of `train_inverse.py`, same seed, same
GPU, same code, diverged starting at epoch 3 and compounded into visibly
different final checkpoints without the fix above; with it, two runs are
bit-identical. **Retraining without this fix can converge to a different
local optimum than any previously-reported checkpoint, including the ones
used for this paper's own numbers** — the original training scripts this
repository is based on did not set this flag either. RAPID is unaffected
(pure NumPy/CPU, no GPU nondeterminism to begin with) and reproduces the
paper's numbers exactly on every rerun.

If you only need to *evaluate* already-trained checkpoints (e.g. via
`evaluate_refinement.py`), none of this applies — evaluation is deterministic
given fixed weights regardless of how those weights were produced.

## Reproducing the paper's results

All commands below assume you are in this directory and `data/processed/ogw_data.pkl`
exists. Replace `--split A` with `B` or `B2` to reproduce the other splits.

### 1. WaveGraphNet (proposed method)

Train the two branches independently — order does not matter, they share
no gradients (skip this step if you downloaded the pre-trained checkpoints
above):

```bash
for seed in 0 1 42; do
  python train_inverse.py --split A --seed $seed
  python train_forward.py --split A --seed $seed --num_interaction_layers 3
done
```

Then evaluate standalone MAE, test-time refinement, and false-positive rate
in one pass (loads the checkpoints saved above):

```bash
python evaluate_refinement.py --split A --seeds 0 1 42 --fwd_num_interaction_layers 3
```

### 2. Baselines

```bash
for seed in 0 1 42; do
  python train_baseline_cnn.py  --split A --seed $seed --feature_type raw
  python train_baseline_cnn.py  --split A --seed $seed --feature_type delta_e
  python train_baseline_lstm.py --split A --seed $seed --feature_type raw
  python train_baseline_lstm.py --split A --seed $seed --feature_type delta_e
  python train_baseline_gnn.py  --split A --seed $seed --model simple_mlp --feature_type raw   # "GNN-MLP"
  python train_baseline_gnn.py  --split A --seed $seed --model attention  --feature_type raw   # "GAT"
  python train_baseline_gnn.py  --split A --seed $seed --feature_type delta_e                  # "GNN (delta_e)"
done
```

`--feature_type delta_e` is a fairness ablation available for every
baseline: it feeds the baseline the exact same 66-scalar ΔE representation
WaveGraphNet uses, instead of the baseline's original raw per-frequency
input, isolating whether an architectural gain reflects the representation
or the model.

Every baseline's val/test MAE, per seed, is printed at the end of training
and appended to `results_mae.json`; the best checkpoint (by validation MAE)
is saved to `checkpoints/<split>/<model>_seed<seed>.pt`.

### 3. RAPID (training-free baseline)

```bash
python run_rapid.py --split A
python run_rapid.py --split B2   # paper's "Split C"
```

No training — this runs in well under a minute per split. Writes
`rapid_evaluation_split<X>.json` with the selected hyperparameters (ellipse
half-width `beta`, detection threshold) and seen/unseen MAE + FPR.

Split C (`B2`) uses only RAPID and WaveGraphNet as its comparison point —
no baseline suite is run there, matching the paper's stated protocol (RAPID
is either the best baseline outright, or competitive with the best learned
baseline once its own hyperparameter-selection instability is accounted
for, on both easier splits, making a third full baseline sweep an
unnecessary added cost on the hardest split).

## Outputs

- `checkpoints/<split>/<model>_seed<seed>.pt` — best-by-validation-MAE
  checkpoint for every trained model.
- `logs/split<X>_<model>_seed<seed>.csv` — per-validation-step training
  curve (loss, val/test MAE, learning rate, wall-clock).
- `results_mae.json` — every model's per-seed val/test MAE, appended to
  automatically by every `train_*.py` script.
- `rapid_evaluation_split<X>.json` — RAPID's per-split hyperparameters and
  results.

## Notes on what's deliberately *not* in this repository

A handful of experimental directions were investigated during development
and did not make it into the reported method — they are intentionally
excluded here rather than carried along as unused code:

- **Training-time forward-consistency coupling** (a joint loss between the
  inverse and forward branches). Superseded entirely by test-time
  refinement — the two branches are trained completely independently in
  this repository.
- **`ConvexGraphDecoder`** (an alternative inverse-branch decoder
  anchoring to fixed plate corners) and the raw-signal-based inverse
  branch it was paired with. `DeltaEOnlyInverseGNN` (this repository's
  inverse branch) uses its own deep-set decoder instead — see the class
  docstring for the convex-hull limitation this implies.
- **Learnable/Fresnel-zone generalizations of RAPID's kernel**
  (`LearnableRapidKernel`, `FresnelRapidKernel`). The paper's RAPID
  baseline is the plain, fixed-`beta` version implemented here.
