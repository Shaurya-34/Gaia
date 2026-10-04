"""Injection simulator: plant synthetic co-moving groups into real Gaia data.

Conventions (README.md): pc, km/s, mas, mas/yr; heliocentric Galactic
Cartesian with x toward the Galactic centre; every function takes an
np.random.Generator.

`field` is a real pull from query_gaia.py after features.add_galactic_pm, so it
has l, b, parallax_corrected, pm_l, pm_b, radial_velocity, parallax_error,
pmra_error, pmdec_error, ruwe and phot_g_mean_mag.

Known simplifications (README.md): errors are copied from nearby real stars;
RA/Dec proper-motion errors stand in for l/b ones; members have no CMD, so
faint members that Gaia would never have measured are not removed.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from query_gaia import MAX_RUWE, MIN_PLX_OVER_ERROR

K = 4.740470446          # km/s per (mas/yr) at 1 kpc, as in features.py
MIN_PLX_CORRECTED = 2.0  # mas; same local cut as query_gaia.apply_zero_point

# PLACEHOLDER ranges until PREREG.md fixes them from the 2024 first-kiloparsec
# paper and Table 1 of the 2025 selection-function paper.
PARAM_RANGES = {
    "dist_pc": (50.0, 500.0),       # uniform
    "n_members": (10, 200),         # log-uniform
    "r_half_pc": (1.0, 10.0),       # log-uniform
    "sigma_v_kms": (0.3, 2.0),      # log-uniform
}


# 1-2: pure math, no data needed
def sample_plummer_positions(n, r_half_pc, rng):
    if rng is None:
        rng = np.random.default_rng()

    a = r_half_pc / 1.305
    u = rng.uniform(0, 1, n)
    r = a / np.sqrt(u**(-2/3) - 1)
    normals = rng.normal(0, 1, (n, 3))
    norms = np.linalg.norm(normals, axis = 1, keepdims= True)
    directions = normals / norms
    positions = directions * r[:, np.newaxis]
    return positions

def sample_member_velocities(n, sigma_v_kms, rng):    # isotropic Gaussian
    if rng is None:
        rng = np.random.default_rng()
    velocities = rng.normal(0, sigma_v_kms, (n, 3))
    return velocities

# 3-4: place the cluster
def cluster_center(dist_pc, l_deg, b_deg):
    l_rad = l_deg * np.pi / 180
    b_rad = b_deg * np.pi / 180
    return np.array([dist_pc * np.cos(b_rad) * np.cos(l_rad), dist_pc * np.cos(b_rad) * np.sin(l_rad), dist_pc * np.sin(b_rad)])

def _unit_vectors(l_rad, b_rad):
    """Line-of-sight, l and b unit vectors, each (n, 3)."""
    zero = np.zeros_like(l_rad)
    r_hat = np.column_stack([np.cos(b_rad) * np.cos(l_rad), np.cos(b_rad) * np.sin(l_rad), np.sin(b_rad)])
    e_l = np.column_stack([-np.sin(l_rad), np.cos(l_rad), zero])
    e_b = np.column_stack([-np.sin(b_rad) * np.cos(l_rad), -np.sin(b_rad) * np.sin(l_rad), np.cos(b_rad)])
    return r_hat, e_l, e_b

def bulk_velocity(field, rng):
    # U,V,W of the whole cluster: the heliocentric 3-D velocity of one random
    # real star that has a Gaia radial velocity.
    rv = field[field.radial_velocity.notna()]
    star = rv.iloc[rng.integers(len(rv))]
    r_hat, e_l, e_b = _unit_vectors(np.deg2rad([star.l]), np.deg2rad([star.b]))
    v_l = K * star.pm_l / star.parallax_corrected
    v_b = K * star.pm_b / star.parallax_corrected
    return (star.radial_velocity * r_hat + v_l * e_l + v_b * e_b)[0]


# 5: the physics step (inverse of features.to_features_cartesian)
def cartesian_to_observables(pos, vel):
    # l, b, parallax, pm_l, pm_b  (pm_l already includes cos b)
    d = np.linalg.norm(pos, axis=1)
    l = np.arctan2(pos[:, 1], pos[:, 0])
    b = np.arcsin(pos[:, 2] / d)
    _, e_l, e_b = _unit_vectors(l, b)
    parallax = 1000 / d
    v_l = (vel * e_l).sum(axis=1)
    v_b = (vel * e_b).sum(axis=1)
    return pd.DataFrame({
        "l": np.rad2deg(l) % 360,
        "b": np.rad2deg(b),
        "parallax": parallax,
        "pm_l": v_l * parallax / K,
        "pm_b": v_b * parallax / K,
    })


# 6-7: make it look like real Gaia
def assign_errors(obs, field, rng):
    """Copy errors, ruwe and G from the nearest real star in 3-D position.
    rng is unused (nearest neighbour is deterministic); kept for the spec."""
    tree = cKDTree(_positions(field.l, field.b, field.parallax_corrected))
    _, idx = tree.query(_positions(obs.l, obs.b, obs.parallax))
    donor = field.iloc[idx]
    out = obs.copy()
    out["parallax_error"] = donor.parallax_error.to_numpy()
    out["pm_l_error"] = donor.pmra_error.to_numpy()    # RA/Dec errors stand in for l/b
    out["pm_b_error"] = donor.pmdec_error.to_numpy()
    out["ruwe"] = donor.ruwe.to_numpy()
    out["phot_g_mean_mag"] = donor.phot_g_mean_mag.to_numpy()
    return out

def _positions(l_deg, b_deg, parallax):
    d = 1000.0 / np.asarray(parallax)
    r_hat, _, _ = _unit_vectors(np.deg2rad(np.asarray(l_deg)), np.deg2rad(np.asarray(b_deg)))
    return r_hat * d[:, None]

def add_noise(obs, rng):
    """Perturb parallax and proper motions by their errors."""
    out = obs.copy()
    for col in ["parallax", "pm_l", "pm_b"]:
        out[col] = out[col] + rng.normal(0, out[f"{col}_error"])
    return out


# 8: realism filter
def apply_selection(obs):
    # same cuts as the real pull (on parallax_corrected, as the local cut is)
    keep = (
        (obs.parallax_corrected >= MIN_PLX_CORRECTED)
        & (obs.parallax_corrected / obs.parallax_error >= MIN_PLX_OVER_ERROR)
        & (obs.ruwe < MAX_RUWE)
    )
    return obs[keep]


# 9: glue
def inject(field, params, rng):
    """Plant one cluster. Injected source_ids are negative, so they never clash
    with real ones. Returns (field + surviving members, truth dict)."""
    n = params["n_members"]
    center = cluster_center(params["dist_pc"], params["l_deg"], params["b_deg"])
    v_bulk = bulk_velocity(field, rng)
    pos = center + sample_plummer_positions(n, params["r_half_pc"], rng)
    vel = v_bulk + sample_member_velocities(n, params["sigma_v_kms"], rng)

    obs = cartesian_to_observables(pos, vel)
    obs = add_noise(assign_errors(obs, field, rng), rng)
    obs = obs.rename(columns={"parallax": "parallax_corrected"})
    obs["source_id"] = -(params["injection_id"] * 1_000_000 + np.arange(n) + 1)
    obs["is_injected"] = True
    survivors = apply_selection(obs)

    field_with_cluster = pd.concat([field.assign(is_injected=False), survivors], ignore_index=True)
    truth = {
        **params,
        "bulk_velocity_kms": v_bulk,
        "n_surviving": len(survivors),
        "member_source_ids": survivors.source_id.to_numpy(),
    }
    return field_with_cluster, truth


# 10: the experiment driver
def sample_params(n_trials, rng):
    def log_uniform(lo, hi):
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

    params = []
    for i in range(n_trials):
        params.append({
            "injection_id": i,
            "dist_pc": float(rng.uniform(*PARAM_RANGES["dist_pc"])),
            "l_deg": float(rng.uniform(0, 360)),
            "b_deg": float(np.rad2deg(np.arcsin(rng.uniform(-1, 1)))),  # uniform on the sky
            "n_members": int(round(log_uniform(*PARAM_RANGES["n_members"]))),
            "r_half_pc": log_uniform(*PARAM_RANGES["r_half_pc"]),
            "sigma_v_kms": log_uniform(*PARAM_RANGES["sigma_v_kms"]),
        })
    return params
