# data/splits.py
"""
Spatial train / val / test splits for the OGW-1 SHM Plate benchmark.

The 28 damage labels form exactly 7 tight physical clusters of 4 points each
(D1-4, D5-8, D9-12, D13-16, D17-20, D21-24, D25-28) — within a cluster, points
are only 15-21mm apart on a 500mm plate (~3-4% of its size). Every split below
assigns WHOLE clusters to train/val/test — a cluster is never split across
more than one role. This matters for the paper's "unseen zone" claim: if a
held-out test label's near-duplicate sibling sits in train, the model can
trivially interpolate that location rather than genuinely extrapolate to it.

Paper name -> internal key
---------------------------------------------------------------------------
Split A  -> "A"   single unseen cluster (D21-24) held out as test.
Split B  -> "B"   two unseen clusters (D1-4, D21-24), both outer corners.
Split C  -> "B2"  cross-region generalization: train confined to a single
                   contiguous region (clusters 1,2,3,5), val is a cluster
                   spatially separate from train (cluster 4), test is two
                   further clusters (6,7). The hardest of the three.
---------------------------------------------------------------------------
Two auxiliary splits also exist, not used for the paper's headline numbers:
  "A2"             same test cluster as A, different val-label choice within
                   each cluster — a robustness check on whether results
                   depend on which specific label was picked for val.
  "C" (this file)  a DIFFERENT experiment from the paper's "Split C" (which is
                   "B2" above) — sensors 6 and 12 are removed entirely from
                   the graph (see REMOVED_SENSORS), testing localization with
                   a reduced, 10-transducer / 45-path sensing network. Not
                   currently reported in the paper. Do not confuse the two.
"""

import random
import re


def _is_single_defect(sample_id: str) -> bool:
    """
    Two raw OGW recordings encode more than one D-label in their sample id
    (e.g. 'D25_D28_100kHz' = a combined D25+D28 defect). A naive
    `sid.split("_")[0]` parse would silently relabel these as a single
    defect with a single-point ground truth, even though the recorded
    signal reflects multiple simultaneous defects. These must never enter
    train/val/test.
    """
    return len(re.findall(r"D\d+", sample_id)) <= 1


# ── Spatial split definitions ────────────────────────────────────────────────
_SPLITS = {
    "A": {
        "test": ["D21", "D22", "D23", "D24"],              # cluster 6, whole
        "val":  ["D4", "D7", "D9", "D13", "D17", "D27"],   # one label / other cluster
    },
    "A2": {
        # Same test cluster as A; diagonally-opposite val label within each
        # cluster's 2x2 mini-grid, to check result sensitivity to that choice.
        "test": ["D21", "D22", "D23", "D24"],
        "val":  ["D1", "D6", "D12", "D16", "D20", "D26"],
    },
    "B": {
        "test": ["D1", "D2", "D3", "D4", "D21", "D22", "D23", "D24"],  # clusters 1+6
        "val":  ["D7", "D9", "D13", "D17", "D27"],
    },
    "B2": {
        # Paper's "Split C". Single-region train, val spatially SEPARATE
        # from train (not mixed in) — every other split's val may share a
        # region with train (val only affects checkpoint selection, not the
        # generalization claim); B2 deliberately removes that leniency too.
        "test": ["D21", "D22", "D23", "D24", "D25", "D26", "D27", "D28"],  # clusters 6+7
        "val":  ["D13", "D14", "D15", "D16"],                               # cluster 4 only
        # everything else (clusters 1,2,3,5 = D1-D12, D17-D20) -> train
    },
    "C": {
        # Reduced-sensor experiment (see REMOVED_SENSORS) — NOT the paper's
        # Split C (that's "B2" above). Cluster 1 is the test region because
        # it sat closest to the two removed sensors (63mm pre-removal).
        "test": ["D1", "D2", "D3", "D4"],
        "val":  ["D7", "D9", "D13", "D17", "D22", "D27"],
    },
}

# Sensor IDs (1-indexed) entirely removed from the graph for a given split —
# node AND every incident path/edge dropped, not a train/val/test relabeling
# of the same 66-path graph. Simulates a hardware gap (failed/missing
# transducers). Only split "C" uses this; every other split has all 12
# sensors and 66 paths.
REMOVED_SENSORS = {
    "C": {6, 12},
}


def get_removed_sensors(split_name: str) -> set:
    """Sensor IDs (1-indexed) to drop entirely for this split's graph."""
    return REMOVED_SENSORS.get(split_name.upper(), set())


def get_train_val_test_ids(
    split_name: str,
    all_sample_ids: list,
    baseline_val_ratio: float = 0.10,
    baseline_test_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    Returns (train_ids, val_ids, test_ids).

    Damaged samples are assigned deterministically by label (see _SPLITS
    above). Baseline (undamaged) samples are split randomly by ratio,
    reproducibly via `seed`.

    Parameters
    ----------
    split_name : one of "A", "A2", "B", "B2", "C" (see module docstring for
                 the paper-name <-> internal-key mapping).
    all_sample_ids : full list of sample keys from the raw data map.
    seed        : RNG seed for the baseline shuffle only (damaged-sample
                 assignment is fully deterministic given split_name).
    """
    key = split_name.upper()
    if key not in _SPLITS:
        raise ValueError(f"Unknown split '{split_name}'. Choose one of: "
                         f"{sorted(_SPLITS.keys())}.")

    cfg = _SPLITS[key]
    test_labels = set(cfg["test"])
    val_labels  = set(cfg["val"])

    damaged_ids  = [s for s in all_sample_ids
                    if not s.startswith("baseline") and _is_single_defect(s)]
    baseline_ids = [s for s in all_sample_ids if s.startswith("baseline")]

    dmg_train, dmg_val, dmg_test = [], [], []
    for sid in damaged_ids:
        label = sid.split("_")[0]          # e.g. "D9" from "D9_100kHz"
        if label in test_labels:
            dmg_test.append(sid)
        elif label in val_labels:
            dmg_val.append(sid)
        else:
            dmg_train.append(sid)

    rng = random.Random(seed)
    bl = list(baseline_ids)
    rng.shuffle(bl)
    n = len(bl)
    n_test = max(1, int(n * baseline_test_ratio))
    n_val  = max(1, int(n * baseline_val_ratio))
    bl_test, bl_val, bl_train = bl[:n_test], bl[n_test:n_test + n_val], bl[n_test + n_val:]

    train_ids = dmg_train + bl_train
    val_ids   = dmg_val + bl_val
    test_ids  = dmg_test + bl_test
    rng.shuffle(train_ids); rng.shuffle(val_ids); rng.shuffle(test_ids)

    _print_split_summary(split_name, train_ids, val_ids, test_ids, val_labels, test_labels)
    return train_ids, val_ids, test_ids


def _print_split_summary(split, train_ids, val_ids, test_ids, val_labels, test_labels):
    n_dmg_tr  = sum(1 for s in train_ids if not s.startswith("baseline"))
    n_dmg_val = sum(1 for s in val_ids   if not s.startswith("baseline"))
    n_dmg_te  = sum(1 for s in test_ids  if not s.startswith("baseline"))
    n_bl_tr, n_bl_val, n_bl_te = (len(train_ids) - n_dmg_tr,
                                  len(val_ids) - n_dmg_val,
                                  len(test_ids) - n_dmg_te)
    print(f"\n  Split {split} | "
          f"Train: {n_dmg_tr}D+{n_bl_tr}B={len(train_ids)} | "
          f"Val ({sorted(val_labels)}): {n_dmg_val}D+{n_bl_val}B={len(val_ids)} | "
          f"Test ({sorted(test_labels)}): {n_dmg_te}D+{n_bl_te}B={len(test_ids)}",
          flush=True)
