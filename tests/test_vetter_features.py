"""Tests for vetter_features.cluster_features (written by Shaurya).

Frames use the project's column conventions (README.md): l, b,
parallax_corrected, pm_l, pm_b, phot_g_mean_mag, bp_rp, source_id.
"""
import math

import numpy as np
import pandas as pd
import pytest

from vetter_features import cluster_features

TRUTH_COLUMNS = ["is_injected", "injection_id", "twin_source_id"]


def synthetic_field(n, rng):
    """Broad field: 20 x 20 deg, 100-500 pc, broad proper motions and CMD."""
    return pd.DataFrame({
        "source_id": rng.integers(1, 2**60, n),
        "l": rng.uniform(100, 120, n), "b": rng.uniform(10, 30, n),
        "parallax_corrected": rng.uniform(2, 10, n),
        "pm_l": rng.normal(0, 20, n), "pm_b": rng.normal(0, 20, n),
        "phot_g_mean_mag": rng.uniform(8, 19, n), "bp_rp": rng.uniform(0, 3, n),
    })


def synthetic_blob(n, rng):
    """Tight in sky, parallax and proper motion, on a narrow CMD sequence."""
    plx = rng.normal(5.0, 0.05, n)
    bp_rp = rng.uniform(0.5, 2.5, n)
    abs_g = 2.0 + 3.0 * bp_rp + rng.normal(0, 0.05, n)
    return pd.DataFrame({
        "source_id": rng.integers(1, 2**60, n),
        "l": rng.normal(110, 0.2, n), "b": rng.normal(20, 0.2, n),
        "parallax_corrected": plx,
        "pm_l": rng.normal(15, 0.3, n), "pm_b": rng.normal(-5, 0.3, n),
        "phot_g_mean_mag": abs_g + 5 * np.log10(100.0 / plx), "bp_rp": bp_rp,
    })


def assert_features_equal(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], float) and math.isnan(a[k]):
            assert isinstance(b[k], float) and math.isnan(b[k]), k
        else:
            assert a[k] == pytest.approx(b[k]), k


def test_blob_is_tighter_than_random_field_draw():
    rng = np.random.default_rng(0)
    field = synthetic_field(20_000, rng)
    blob = synthetic_blob(50, rng)
    draw = field.sample(50, random_state=1)
    f_blob = cluster_features(blob, field)
    f_draw = cluster_features(draw, field)
    assert f_blob["5d_extent"] < f_draw["5d_extent"]
    assert f_blob["cmr_scatter"] < f_draw["cmr_scatter"]


def with_garbage_truth(df, rng):
    """Same frame with garbage truth columns put FIRST, where a column-order
    fallback would find them."""
    garbage = pd.DataFrame({
        "is_injected": rng.integers(0, 2, len(df)).astype(bool),
        "injection_id": rng.normal(0, 1e6, len(df)),
        "twin_source_id": rng.integers(-2**60, 2**60, len(df)),
    }, index=df.index)
    return pd.concat([garbage, df], axis=1)


@pytest.mark.parametrize("make_group", [synthetic_blob, None], ids=["blob", "field_draw"])
def test_ignores_truth_columns(make_group):
    rng = np.random.default_rng(2)
    field = synthetic_field(5_000, rng)
    members = make_group(40, rng) if make_group else field.sample(40, random_state=3)
    clean = cluster_features(members, field)
    dirty = cluster_features(with_garbage_truth(members, rng), with_garbage_truth(field, rng))
    assert not set(TRUTH_COLUMNS) & set(members.columns)
    assert_features_equal(clean, dirty)


def test_extent_is_the_same_across_l_0_360():
    """A field straddling l = 0/360 must give the same 5-D extent as the same
    field rotated to l = 180. The old code scaled l by the wrapped field's
    ~350 degree spread, so a group's size on the sky stopped counting; a group
    spread over a few degrees (like a nearby cluster) shows it."""
    rng = np.random.default_rng(5)
    field = synthetic_field(20_000, rng)
    blob = synthetic_blob(60, rng)
    blob["l"] = 110.0 + rng.normal(0, 2.0, len(blob))  # extended on the sky
    for df in (field, blob):  # centre both on l = 180 first (field 100-120 -> 170-190)
        df["l"] = df.l + 70.0
    at_180 = cluster_features(blob, field)["5d_extent"]
    wrapped_field, wrapped_blob = field.copy(), blob.copy()
    for df in (wrapped_field, wrapped_blob):
        df["l"] = (df.l - 180.0) % 360  # same geometry, now straddling l = 0/360
    at_0 = cluster_features(wrapped_blob, wrapped_field)["5d_extent"]
    assert at_0 == pytest.approx(at_180, rel=0.05)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_tiny_group_gives_nan_scatter(n):
    rng = np.random.default_rng(4)
    field = synthetic_field(1_000, rng)
    f = cluster_features(synthetic_blob(n, rng), field)
    assert math.isnan(f["cmr_scatter"])
