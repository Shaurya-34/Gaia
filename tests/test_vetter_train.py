import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

import vetter_train as vt

BLOCKS = json.loads(vt.SPLITS_PATH.read_text())["blocks"]


def synthetic_groups(n, split, rng, separable=True):
    """Detected groups in the given split's blocks. When separable, injected
    groups are compact (small 5D extent) while CST barely separates them."""
    label = rng.integers(0, 2, n)
    extent = np.where(label == 1, rng.normal(0.3, 0.1, n), rng.normal(1.0, 0.3, n))
    if not separable:
        label = rng.integers(0, 2, n)  # labels unrelated to the features
    return pd.DataFrame({
        "member_count": rng.integers(10, 200, n).astype(float),
        "cst": rng.normal(4.0, 2.0, n) + 0.3 * label,
        "hdbscan_stability": rng.uniform(0, 1, n),
        "hdbscan_persistence": rng.uniform(0, 1, n),
        "mean_membership_probability": rng.uniform(0.5, 1, n),
        "5d_extent": extent,
        "cmr_scatter": rng.normal(0, 1, n) + 5 * label,  # would leak the label if used
        "label": label,
        "block": rng.choice(BLOCKS[split], n),
        "source": np.where(label == 1, "injected", rng.choice(["shuffle", "hypersphere"], n)),
    })


def datasets(separable, seed=0):
    rng = np.random.default_rng(seed)
    return (synthetic_groups(3000, "train", rng, separable), synthetic_groups(1000, "val", rng, separable),
            synthetic_groups(2000, "test", rng, separable))


def test_no_block_in_two_splits():
    seen = sum(BLOCKS.values(), [])
    assert sorted(seen) == list(range(12)) and len(seen) == len(set(seen))
    for split, blocks in BLOCKS.items():
        for b in blocks:
            assert vt.load_split(b) == split


def test_train_rejects_rows_from_wrong_split():
    df_train, df_val, _ = datasets(separable=True)
    with pytest.raises(ValueError):
        vt.train(df_val, df_val)
    with pytest.raises(ValueError):
        vt.train(df_train, df_train)


def test_never_trains_on_cmd_scatter_or_bookkeeping():
    assert not {"cmr_scatter", "label", "block", "source"} & set(vt.FEATURES)


def test_vetter_beats_baseline_when_separable():
    df_train, df_val, df_test = datasets(separable=True)
    fitted = vt.train(df_train, df_val)
    result = vt.compare_to_baseline(df_test.label, vt.vetter_scores(fitted, df_test),
                                    vt.baseline_scores(df_test), np.random.default_rng(1))
    assert result["passes"], result
    # the validation-chosen threshold gives roughly the target recall on test
    pred = vt.vetter_scores(fitted, df_test) >= fitted["threshold"]
    assert abs(pred[df_test.label == 1].mean() - 0.8) < 0.05


def test_random_labels_give_auc_near_half():
    df_train, df_val, df_test = datasets(separable=False)
    fitted = vt.train(df_train, df_val)
    auc = roc_auc_score(df_test.label, vt.vetter_scores(fitted, df_test))
    assert abs(auc - 0.5) < 0.05


def test_same_seed_same_output():
    df_train, df_val, df_test = datasets(separable=True)
    a, b = vt.train(df_train, df_val), vt.train(df_train, df_val)
    np.testing.assert_array_equal(vt.vetter_scores(a, df_test), vt.vetter_scores(b, df_test))
    assert a["threshold"] == b["threshold"]
    ra = vt.compare_to_baseline(df_test.label, vt.vetter_scores(a, df_test), vt.baseline_scores(df_test),
                                np.random.default_rng(2))
    rb = vt.compare_to_baseline(df_test.label, vt.vetter_scores(b, df_test), vt.baseline_scores(df_test),
                                np.random.default_rng(2))
    assert ra == rb


def test_precision_at_recall():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    perfect = np.arange(10)[::-1].astype(float)  # positives score highest
    assert vt.precision_at_recall(y, perfect, 0.8) == 1.0
    assert vt.precision_at_recall(y, -perfect, 0.8) == pytest.approx(4 / 10)


def test_score_test_runs_only_once(tmp_path):
    df_train, df_val, df_test = datasets(separable=True)
    fitted = vt.train(df_train, df_val)
    lock = tmp_path / "test_scored.lock"
    vt.score_test(fitted, df_test, np.random.default_rng(3), lock_path=lock)
    assert lock.exists()
    with pytest.raises(RuntimeError):
        vt.score_test(fitted, df_test, np.random.default_rng(3), lock_path=lock)


def test_split_by_trial_keeps_trials_whole():
    df = pd.DataFrame({"trial_key": np.repeat([f"t{i}" for i in range(50)], 4), "x": range(200)})
    out = vt.split_by_trial(df)
    assert out.groupby("trial_key").split.nunique().max() == 1
    assert set(out.split) == {"train", "val", "test"}
    pd.testing.assert_frame_equal(out, vt.split_by_trial(df))  # seeded


def test_score_test_rejects_non_test_blocks(tmp_path):
    df_train, df_val, _ = datasets(separable=True)
    fitted = vt.train(df_train, df_val)
    with pytest.raises(ValueError):
        vt.score_test(fitted, df_val, np.random.default_rng(4), lock_path=tmp_path / "x.lock")
