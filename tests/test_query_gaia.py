"""Offline checks of query_gaia.py (no archive access)."""
import json

import numpy as np
import pandas as pd

import query_gaia
import splits


def test_query_has_all_prereg_cuts():
    q = query_gaia.build_query(3, 0)
    for clause in ["parallax >= 1.9", "parallax_over_error > 10", "ruwe < 1.4",
                   "visibility_periods_used > 6", "astrometric_excess_noise < 1",
                   "phot_g_mean_flux_over_error > 10", "phot_bp_mean_flux_over_error > 10",
                   "phot_rp_mean_flux_over_error > 10"]:
        assert clause in q, clause


def test_pixel_range_is_level3_width():
    assert query_gaia.pixel_source_id_range(3, 0) == (0, 2 ** 53)
    assert query_gaia.pixel_source_id_range(3, 5) == (5 * 2 ** 53, 6 * 2 ** 53)


def test_drop_sealed_removes_only_sealed_block():
    sealed = json.loads(query_gaia.SPLITS_PATH.read_text())["blocks"]["sealed"]
    rng = np.random.default_rng(0)
    l = rng.uniform(0, 360, 20_000)
    b = np.rad2deg(np.arcsin(rng.uniform(-1, 1, 20_000)))
    df = pd.DataFrame({"l": l, "b": b})
    kept = query_gaia.drop_sealed(df)
    in_sealed = np.isin(splits.galactic_block(l, b), sealed)
    assert in_sealed.any()
    assert len(kept) == (~in_sealed).sum()
    assert not np.isin(splits.galactic_block(kept.l, kept.b), sealed).any()
