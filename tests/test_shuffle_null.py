import numpy as np
import pandas as pd

from shuffle_null import CFG, shuffle_cells, shuffle_proper_motions

CLUMP_PM = np.array([30.0, -15.0])
CLUMP_LB = np.array([105.0, 25.0])


def field_with_clump(rng, n_field=200_000, n_clump=30):
    """Broad field over a 10 x 10 deg patch plus a cluster that is tight on the
    sky (0.2 deg), in parallax and in proper motion. The field is dense enough
    that the cluster is a few percent of its cell, as for a real distant group."""
    n = n_field + n_clump
    df = pd.DataFrame({
        "l": rng.uniform(100, 110, n), "b": rng.uniform(20, 30, n),
        "parallax_corrected": rng.uniform(2, 10, n),
        "pm_l": rng.normal(0, 20, n), "pm_b": rng.normal(0, 20, n),
        "pm_l_error": rng.uniform(0.01, 0.1, n), "pm_b_error": rng.uniform(0.01, 0.1, n),
    })
    df.loc[n_field:, ["l", "b"]] = CLUMP_LB + rng.normal(0, 0.1, (n_clump, 2))
    df.loc[n_field:, "parallax_corrected"] = rng.normal(5.0, 0.05, n_clump)
    df.loc[n_field:, ["pm_l", "pm_b"]] = CLUMP_PM + rng.normal(0, 0.2, (n_clump, 2))
    return df


def co_moving_near_clump(d):
    """Stars within 0.3 deg of the clump centre AND within 1 mas/yr of its PM."""
    on_sky = np.hypot(d.l - CLUMP_LB[0], d.b - CLUMP_LB[1]) < 0.3
    in_pm = np.hypot(d.pm_l - CLUMP_PM[0], d.pm_b - CLUMP_PM[1]) < 1.0
    return int((on_sky & in_pm).sum())


def test_positions_parallaxes_unchanged_and_pms_permuted_within_cells():
    df = field_with_clump(np.random.default_rng(0), n_field=20_000)
    out = shuffle_proper_motions(df, np.random.default_rng(1))
    pd.testing.assert_frame_equal(out[["l", "b", "parallax_corrected"]], df[["l", "b", "parallax_corrected"]])
    assert not np.array_equal(out.pm_l, df.pm_l)
    cols = ["pm_l", "pm_b", "pm_l_error", "pm_b_error"]
    cells = shuffle_cells(df)
    for c in np.unique(cells)[:200]:
        m = cells == c
        # same multiset of (pm_l, pm_b, errors) inside each cell, kept together per star
        assert sorted(map(tuple, df.loc[m, cols].to_numpy())) == sorted(map(tuple, out.loc[m, cols].to_numpy()))


def test_cells_are_pixel_by_parallax_quantile():
    df = field_with_clump(np.random.default_rng(2), n_field=20_000)
    n_bins = CFG["parallax_quantile_bins"]
    assert set(np.unique(shuffle_cells(df) % n_bins)) == set(range(n_bins))


def test_clump_disappears():
    df = field_with_clump(np.random.default_rng(3))
    out = shuffle_proper_motions(df, np.random.default_rng(4))
    assert co_moving_near_clump(df) >= 30
    assert co_moving_near_clump(out) < 8  # clump PMs scatter over a ~3.4 deg^2 cell


def test_deterministic():
    df = field_with_clump(np.random.default_rng(5), n_field=20_000)
    a = shuffle_proper_motions(df, np.random.default_rng(6))
    b = shuffle_proper_motions(df, np.random.default_rng(6))
    pd.testing.assert_frame_equal(a, b)
