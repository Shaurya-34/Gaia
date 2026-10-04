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


# --- determinism -----------------------------------------------------------

def test_same_seed_same_output():
    plummer = fn("sample_plummer_positions")
    vels = fn("sample_member_velocities")
    a = plummer(1000, 5.0, np.random.default_rng(42))
    b = plummer(1000, 5.0, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)
    a = vels(1000, 1.0, np.random.default_rng(42))
    b = vels(1000, 1.0, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)
