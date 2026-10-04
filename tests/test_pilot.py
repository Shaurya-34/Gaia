import json

import numpy as np
import pandas as pd
import pytest

import baseline as bl
import injection
import pilot
from splits import galactic_block

SPLITS = json.loads(injection.SPLITS_PATH.read_text())
BLOCK = injection.CFG["injection"]["pilot_block"]


def test_sample_params_restricted_to_block():
    allowed = {c["pixel"] for c in SPLITS["eligible_centres"][str(BLOCK)]}
    ps = injection.sample_params(200, np.random.default_rng(0), split="train", blocks=[BLOCK])
    assert {p["field_pixel"] for p in ps} <= allowed
    with pytest.raises(ValueError):
        injection.sample_params(5, np.random.default_rng(0), split="train", blocks=SPLITS["blocks"]["val"])


@pytest.fixture(scope="module")
def field():
    """Synthetic real-format field around the first eligible centre of the pilot block."""
    centre = SPLITS["eligible_centres"][str(BLOCK)][0]["pixel"]
    rng = np.random.default_rng(0)
    n = 300_000
    l = rng.uniform(0, 360, n)
    b = np.rad2deg(np.arcsin(rng.uniform(-1, 1, n)))
    keep = np.isin(galactic_block(l, b, level=SPLITS["field_level"]), bl.field_pixels(centre))
    m = int(keep.sum())
    rv = rng.normal(0, 30, m)
    rv[rng.uniform(size=m) < 0.6] = np.nan
    df = pd.DataFrame({
        "source_id": np.arange(1, m + 1), "l": l[keep], "b": b[keep],
        "parallax_corrected": rng.uniform(2, 10, m), "pm_l": rng.normal(0, 20, m), "pm_b": rng.normal(0, 20, m),
        "radial_velocity": rv, "phot_g_mean_mag": rng.uniform(8, 19, m), "bp_rp": rng.uniform(0, 3, m),
        "parallax_error": 0.05, "pmra_error": 0.05, "pmdec_error": 0.05, "ruwe": 1.0,
        "visibility_periods_used": 20, "astrometric_excess_noise": 0.1,
        "phot_g_mean_flux_over_error": 500.0, "phot_bp_mean_flux_over_error": 100.0,
        "phot_rp_mean_flux_over_error": 100.0,
    })
    return centre, df


def trial_params(centre, **over):
    l, b = bl.pixel_centre(centre)
    p = {"injection_id": 0, "field_pixel": centre, "dist_pc": 150.0, "l_deg": l, "b_deg": b,
         "n_members": 150, "r_half_pc": 2.0, "sigma_v_kms": 0.5}
    p.update(over)
    return p


def test_run_trial_detects_a_rich_nearby_cluster(field):
    centre, df = field
    row = pilot.run_trial(df, trial_params(centre), [1, 0])
    assert row["n_surviving"] > 100
    assert row["recovered"] and row["detected_tau3"]
    assert row["runtime_s"] > 0 and row["peak_memory_gb"] > 0
    assert {f"detected_tau{t}" for t in pilot.TAUS} <= row.keys()


def test_run_trial_reports_known_overlap(field):
    centre, df = field
    # pretend 50 real field stars near the injection site are a catalogue cluster
    l, b = bl.pixel_centre(centre)
    near = df.iloc[np.argsort(np.hypot(df.l - l, df.b - b))[:50]]
    known = pd.DataFrame({"Name": "Fake_1", "GaiaDR3": near.source_id})
    row = pilot.run_trial(df, trial_params(centre), [1, 0], known)
    assert {"known_overlap_discard", "known_overlap_cluster", "known_overlap_n"} <= row.keys()
    assert row["known_overlap_discard"] in (True, False)


def test_run_trial_is_deterministic(field):
    centre, df = field
    a = pilot.run_trial(df, trial_params(centre, n_members=40), [1, 1])
    b = pilot.run_trial(df, trial_params(centre, n_members=40), [1, 1])
    for k in ["runtime_s", "peak_memory_gb"]:
        a.pop(k), b.pop(k)
    assert a == b
