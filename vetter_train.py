"""Train the vetter and compare it with the baseline (PREREG section 8).

Input: one row per detected group with the vetter_features.cluster_features
columns, `label` (1 injected, 0 shuffle-null or hypersphere fake), `block`
(level-0 block, splits.json) and `source` (injected, shuffle, hypersphere).

The CMD-scatter column is never a training feature (PREREG section 8): twin
stars give injected members field-like scatter.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from config import load_config

CFG = load_config()
SPLITS_PATH = Path(__file__).with_name("splits.json")
TEST_LOCK = Path(__file__).with_name("results") / "test_scored.lock"
FEATURES = ["member_count", "cst", "hdbscan_stability", "hdbscan_persistence",
            "mean_membership_probability", "5d_extent"]


def load_split(block):
    """Split name ("train", "val", "test" or "sealed") of a level-0 block,
    read from splits.json. Never random."""
    blocks = json.loads(SPLITS_PATH.read_text())["blocks"]
    names = [name for name, members in blocks.items() if block in members]
    if len(names) != 1:
        raise ValueError(f"block {block} is in {len(names)} splits: {names}")
    return names[0]


def _check_split(df, split):
    wrong = sorted({int(b) for b in df.block if load_split(int(b)) != split})
    if wrong:
        raise ValueError(f"rows from blocks {wrong} are not in the {split} split")


def baseline_scores(df):
    """Baseline = HDBSCAN + CST: the score is the CST value."""
    return df["cst"].to_numpy(dtype=float)


def precision_at_recall(y, score, recall=CFG["vetter"]["recall"]):
    """Highest precision among thresholds that reach at least `recall`."""
    precision, rec, _ = precision_recall_curve(y, score)
    return float(precision[rec >= recall].max())


def _threshold_at_recall(y, score, recall):
    """Largest score threshold whose recall is at least `recall`."""
    _, rec, thresholds = precision_recall_curve(y, score)
    return float(thresholds[rec[:-1] >= recall].max())


def train(df_train, df_val):
    """Fit LightGBM on the train blocks. Validation is used only for early
    stopping and for choosing the threshold that gives the target recall."""
    _check_split(df_train, "train")
    _check_split(df_val, "val")
    cfg = CFG["vetter"]
    model = lgb.LGBMClassifier(
        n_estimators=cfg["max_trees"], learning_rate=cfg["learning_rate"],
        num_leaves=cfg["num_leaves"], random_state=cfg["seed"],
        deterministic=True, force_row_wise=True, n_jobs=1, verbose=-1,
    )
    model.fit(df_train[FEATURES], df_train["label"],
              eval_X=(df_val[FEATURES],), eval_y=(df_val["label"],),
              callbacks=[lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False)])
    val_score = model.predict_proba(df_val[FEATURES])[:, 1]
    return {"model": model, "features": FEATURES,
            "threshold": _threshold_at_recall(df_val["label"], val_score, cfg["recall"])}


def vetter_scores(fitted, df):
    return fitted["model"].predict_proba(df[fitted["features"]])[:, 1]


def compare_to_baseline(y, vetter, baseline, rng, recall=CFG["vetter"]["recall"],
                        n_boot=CFG["analysis"]["bootstrap_resamples"]):
    """Precision difference (vetter - baseline) at `recall`, with a 95%
    bootstrap interval over groups. Passes when the difference is at least
    min_precision_gain and the interval excludes 0 (PREREG section 8)."""
    y, vetter, baseline = map(np.asarray, (y, vetter, baseline))
    diff = precision_at_recall(y, vetter, recall) - precision_at_recall(y, baseline, recall)
    boot = []
    for _ in range(n_boot):
        i = rng.integers(len(y), size=len(y))
        if y[i].min() == y[i].max():  # resample with one class: no precision defined
            continue
        boot.append(precision_at_recall(y[i], vetter[i], recall) - precision_at_recall(y[i], baseline[i], recall))
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"difference": diff, "ci95": (float(low), float(high)),
            "passes": bool(diff >= CFG["vetter"]["min_precision_gain"] and low > 0)}


def score_test(fitted, df_test, rng, lock_path=TEST_LOCK):
    """Score the test blocks ONCE. Writes a lock file the first time and
    refuses to run again, so nothing can be tuned against the test blocks."""
    lock_path = Path(lock_path)
    if lock_path.exists():
        raise RuntimeError(f"test blocks already scored ({lock_path}); refusing to score again")
    _check_split(df_test, "test")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps({"scored_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                     "n_groups": len(df_test)}) + "\n")
    vetter = vetter_scores(fitted, df_test)
    return compare_to_baseline(df_test["label"], vetter, baseline_scores(df_test), rng)
