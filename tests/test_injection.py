"""Tests that define correct behaviour for injection.py (written by Shaurya).

Skips cleanly while injection.py is missing, does not import yet, or a
function is not written yet.
"""
import numpy as np
import pandas as pd
import pytest

try:
    import injection
except Exception as exc:  # missing file, or a half-written file that does not parse
    pytest.skip(f"injection.py not importable yet: {exc!r}", allow_module_level=True)

K = 4.74047  # km/s per (mas/yr * kpc)


def fn(name):
    f = getattr(injection, name, None)
    if f is None:
        pytest.skip(f"injection.{name} not written yet")
    return f


def random_stars(n, rng):
    """Heliocentric positions 50-500 pc in random directions, and velocities."""
    d = rng.uniform(50, 500, n)
    u = rng.normal(size=(n, 3))
    pos = u / np.linalg.norm(u, axis=1, keepdims=True) * d[:, None]
    vel = rng.normal(0, 30, (n, 3))
    return pos, vel


# --- sample_plummer_positions ---------------------------------------------

def test_plummer_shape_and_median_radius():
    r_half = 5.0
    pos = fn("sample_plummer_positions")(100_000, r_half, np.random.default_rng(1))
    assert pos.shape == (100_000, 3)
    median_r = np.median(np.linalg.norm(pos, axis=1))
    assert median_r == pytest.approx(r_half, rel=0.03)


def test_plummer_is_centred():
    pos = fn("sample_plummer_positions")(100_000, 5.0, np.random.default_rng(2))
    # isotropic about the origin; the Plummer tail is heavy, so use the median
    np.testing.assert_allclose(np.median(pos, axis=0), 0.0, atol=0.1)


# --- sample_member_velocities ---------------------------------------------

def test_member_velocity_std():
    sigma = 0.8
    vel = fn("sample_member_velocities")(100_000, sigma, np.random.default_rng(3))
    assert vel.shape == (100_000, 3)
    np.testing.assert_allclose(vel.std(axis=0), sigma, rtol=0.02)
    np.testing.assert_allclose(vel.mean(axis=0), 0.0, atol=0.02)


# --- cluster_center --------------------------------------------------------

def test_cluster_center_distance_and_frame():
    cc = fn("cluster_center")
    c = cc(300.0, 123.0, -40.0)
    assert c.shape == (3,)
    assert np.linalg.norm(c) == pytest.approx(300.0)
    # x points to the Galactic centre (l=0, b=0), y to l=90, z to b=90
    np.testing.assert_allclose(cc(100.0, 0.0, 0.0), [100, 0, 0], atol=1e-9)
    np.testing.assert_allclose(cc(100.0, 90.0, 0.0), [0, 100, 0], atol=1e-9)
    np.testing.assert_allclose(cc(100.0, 0.0, 90.0), [0, 0, 100], atol=1e-9)


# --- cartesian_to_observables ----------------------------------------------

@pytest.fixture
def stars():
    return random_stars(5_000, np.random.default_rng(4))


def test_observables_columns_and_ranges(stars):
    pos, vel = stars
    obs = fn("cartesian_to_observables")(pos, vel)
    assert isinstance(obs, pd.DataFrame)
    assert list(obs.columns) == ["l", "b", "parallax", "pm_l", "pm_b"]
    assert len(obs) == len(pos)
    assert ((obs.l >= 0) & (obs.l < 360)).all()
    assert ((obs.b >= -90) & (obs.b <= 90)).all()
    np.testing.assert_allclose(obs.parallax, 1000.0 / np.linalg.norm(pos, axis=1))


def test_observables_tangential_speed(stars):
    pos, vel = stars
    obs = fn("cartesian_to_observables")(pos, vel)
    r_hat = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    vt2_true = (vel ** 2).sum(axis=1) - (vel * r_hat).sum(axis=1) ** 2
    v_l = K * obs.pm_l.to_numpy() / obs.parallax.to_numpy()
    v_b = K * obs.pm_b.to_numpy() / obs.parallax.to_numpy()
    np.testing.assert_allclose(v_l ** 2 + v_b ** 2, vt2_true, rtol=1e-6)


def test_observables_match_astropy(stars):
    """Signs and directions, not just the speed: compare with astropy's
    Galactic frame (pm_l is pm_l * cos b, as in features.add_galactic_pm)."""
    import astropy.units as u
    from astropy.coordinates import (CartesianDifferential,
                                     CartesianRepresentation, Galactic)

    pos, vel = stars
    obs = fn("cartesian_to_observables")(pos, vel)
    rep = CartesianRepresentation(pos.T * u.pc, differentials=CartesianDifferential(vel.T * u.km / u.s))
    g = Galactic(rep)
    sph = g.spherical
    np.testing.assert_allclose(obs.l, sph.lon.deg, atol=1e-8)
    np.testing.assert_allclose(obs.b, sph.lat.deg, atol=1e-8)
    np.testing.assert_allclose(obs.pm_l, g.pm_l_cosb.to_value(u.mas / u.yr), rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(obs.pm_b, g.pm_b.to_value(u.mas / u.yr), rtol=1e-5, atol=1e-8)


# --- round trip through features.py (noise off) ------------------------------

def test_observables_round_trip_through_features(stars):
    from features import to_features_cartesian

    pos, vel = stars
    obs = fn("cartesian_to_observables")(pos, vel).rename(columns={"parallax": "parallax_corrected"})
    feats = to_features_cartesian(obs, tau_myr=1.0)
    np.testing.assert_allclose(feats[:, :3], pos, rtol=1e-9, atol=1e-9)


# --- synthetic field for the data-dependent functions -------------------------

def synthetic_field(n, rng, plx_error=0.1):
    """A fake real pull: columns as query_gaia.py + features.add_galactic_pm."""
    pos, vel = random_stars(n, rng)
    obs = fn("cartesian_to_observables")(pos, vel)
    r_hat = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    rv = (vel * r_hat).sum(axis=1)
    rv[rng.uniform(size=n) < 0.7] = np.nan  # most stars lack an RV
    field = pd.DataFrame({
        "source_id": np.arange(1, n + 1),
        "l": obs.l, "b": obs.b, "parallax_corrected": obs.parallax,
        "pm_l": obs.pm_l, "pm_b": obs.pm_b, "radial_velocity": rv,
        "parallax_error": plx_error, "pmra_error": 0.1, "pmdec_error": 0.1,
        "ruwe": 1.0, "phot_g_mean_mag": rng.uniform(8, 19, n), "bp_rp": rng.uniform(0, 3, n),
        "visibility_periods_used": 20, "astrometric_excess_noise": 0.1,
        "phot_g_mean_flux_over_error": 500.0, "phot_bp_mean_flux_over_error": 100.0,
        "phot_rp_mean_flux_over_error": 100.0,
    })
    return field, vel


@pytest.fixture
def field():
    return synthetic_field(3_000, np.random.default_rng(5))


def params(**over):
    p = {"injection_id": 7, "dist_pc": 150.0, "l_deg": 30.0, "b_deg": 10.0,
         "n_members": 200, "r_half_pc": 3.0, "sigma_v_kms": 0.5}
    p.update(over)
    return p


def test_bulk_velocity_is_a_real_star_with_rv(field):
    df, vel = field
    v = fn("bulk_velocity")(df, np.random.default_rng(6))
    assert v.shape == (3,)
    has_rv = df.radial_velocity.notna().to_numpy()
    dist = np.linalg.norm(vel[has_rv] - v, axis=1)
    assert dist.min() < 1e-6


def test_assign_errors_and_noise(field):
    df, _ = field
    rng = np.random.default_rng(7)
    obs = fn("cartesian_to_observables")(*random_stars(100, rng))
    obs = fn("assign_errors")(obs, df, rng)
    for col in ["parallax_error", "pm_l_error", "pm_b_error", "ruwe", "phot_g_mean_mag", "bp_rp",
                "visibility_periods_used", "astrometric_excess_noise", "phot_bp_mean_flux_over_error"]:
        assert obs[col].notna().all()
    noisy = fn("add_noise")(obs, np.random.default_rng(8))
    assert not np.allclose(noisy.parallax, obs.parallax)
    zero = obs.assign(parallax_error=0.0, pm_l_error=0.0, pm_b_error=0.0)
    np.testing.assert_array_equal(fn("add_noise")(zero, rng)[["parallax", "pm_l", "pm_b"]],
                                  zero[["parallax", "pm_l", "pm_b"]])


def test_twins_are_near_and_copied(field):
    df, _ = field
    rng = np.random.default_rng(14)
    pos, vel = random_stars(200, rng)
    obs = fn("assign_errors")(fn("cartesian_to_observables")(pos, vel), df, rng)
    twin = df.set_index("source_id").loc[obs.twin_source_id]
    np.testing.assert_array_equal(obs.phot_g_mean_mag, twin.phot_g_mean_mag)
    np.testing.assert_array_equal(obs.bp_rp, twin.bp_rp)
    # each twin is among the twin_k real stars nearest the member
    k = injection.CFG["injection"]["twin_k"]
    field_xyz = injection._positions(df.l, df.b, df.parallax_corrected)
    twin_xyz = injection._positions(twin.l, twin.b, twin.parallax_corrected)
    member_xyz = injection._positions(obs.l, obs.b, obs.parallax)
    d_twin = np.linalg.norm(twin_xyz - member_xyz, axis=1)
    d_all = np.linalg.norm(field_xyz[None, :, :] - member_xyz[:, None, :], axis=2)
    kth = np.sort(d_all, axis=1)[:, k - 1]
    assert (d_twin <= kth + 1e-9).all()
    # twins are drawn at random, not always the nearest star
    assert obs.twin_source_id.nunique() > 0.5 * len(obs)


def test_apply_selection_cuts():
    good = {"parallax_corrected": 5.0, "parallax_error": 0.1, "ruwe": 1.0,
            "visibility_periods_used": 20, "astrometric_excess_noise": 0.1,
            "phot_g_mean_flux_over_error": 500.0, "phot_bp_mean_flux_over_error": 100.0,
            "phot_rp_mean_flux_over_error": 100.0, "phot_g_mean_mag": 15.0}
    bad = [("parallax_corrected", 1.9), ("parallax_error", 0.5),  # plx/err = 10 is not > 10
           ("ruwe", 1.4), ("visibility_periods_used", 6), ("astrometric_excess_noise", 1.0),
           ("phot_g_mean_flux_over_error", 10.0), ("phot_bp_mean_flux_over_error", 10.0),
           ("phot_rp_mean_flux_over_error", 10.0), ("phot_g_mean_mag", 21.5)]
    obs = pd.DataFrame([good] + [{**good, col: val} for col, val in bad])
    kept = fn("apply_selection")(obs)
    assert list(kept.index) == [0]


def test_inject_truth_matches_output(field):
    df, _ = field
    out, truth = fn("inject")(df, params(), np.random.default_rng(9))
    injected = out[out.is_injected]
    assert len(out) == len(df) + truth["n_surviving"]
    assert len(injected) == truth["n_surviving"] > 0
    assert set(injected.source_id) == set(truth["member_source_ids"])
    assert (injected.source_id < 0).all() and injected.source_id.is_unique
    assert not out.is_injected.iloc[:len(df)].any()


def test_surviving_fraction_falls_with_distance(field):
    df, _ = field
    near = fn("inject")(df, params(dist_pc=100.0), np.random.default_rng(10))[1]
    far = fn("inject")(df, params(dist_pc=490.0), np.random.default_rng(10))[1]
    assert far["n_surviving"] < near["n_surviving"]


def test_sample_params_ranges():
    import json

    from splits import galactic_block

    rng_cfg = injection.CFG["injection"]
    splits = json.loads(injection.SPLITS_PATH.read_text())
    for split in ["train", "val", "test"]:
        ps = fn("sample_params")(300, np.random.default_rng(11), split=split)
        assert len(ps) == 300 and len({p["injection_id"] for p in ps}) == 300
        allowed = {c["pixel"] for b in splits["blocks"][split] for c in splits["eligible_centres"][str(b)]}
        for p in ps:
            assert p["field_pixel"] in allowed
            assert galactic_block(p["l_deg"], p["b_deg"], level=splits["field_level"]) == p["field_pixel"]
            assert rng_cfg["dist_pc"][0] <= p["dist_pc"] <= rng_cfg["dist_pc"][1]
            assert rng_cfg["n_members"][0] <= p["n_members"] <= rng_cfg["n_members"][1]
            assert rng_cfg["r_half_pc"][0] <= p["r_half_pc"] <= rng_cfg["r_half_pc"][1]
            assert rng_cfg["sigma_v_kms"][0] <= p["sigma_v_kms"] <= rng_cfg["sigma_v_kms"][1]


def test_sample_params_never_uses_sealed_block():
    import json

    sealed = json.loads(injection.SPLITS_PATH.read_text())["blocks"]["sealed"]
    for split in ["train", "val", "test"]:
        for p in fn("sample_params")(200, np.random.default_rng(15), split=split):
            assert p["field_pixel"] // 16 not in sealed


# --- determinism -----------------------------------------------------------

def test_same_seed_same_injection(field):
    df, _ = field
    a_out, a_truth = fn("inject")(df, params(), np.random.default_rng(12))
    b_out, b_truth = fn("inject")(df, params(), np.random.default_rng(12))
    pd.testing.assert_frame_equal(a_out, b_out)
    np.testing.assert_array_equal(a_truth["member_source_ids"], b_truth["member_source_ids"])
    assert fn("sample_params")(5, np.random.default_rng(13)) == fn("sample_params")(5, np.random.default_rng(13))


def test_same_seed_same_output():
    plummer = fn("sample_plummer_positions")
    vels = fn("sample_member_velocities")
    a = plummer(1000, 5.0, np.random.default_rng(42))
    b = plummer(1000, 5.0, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)
    a = vels(1000, 1.0, np.random.default_rng(42))
    b = vels(1000, 1.0, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)
