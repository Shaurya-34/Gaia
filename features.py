"""Turn Gaia astrometry into the 5-D search space.

Default (to_features): Hunt & Reffert 2021 space (l, b, parallax_corrected,
pm_l, pm_b), each column robust-scaled (sklearn RobustScaler). Working in
observables avoids smearing clusters along the line of sight by parallax noise.

Ablation (to_features_cartesian): Galactic Cartesian position (pc) plus
tangential velocity components (v_l, v_b) scaled by tau.

Radial velocity is deliberately NOT used. It is kept in the table as an
independent check on candidate groups later.

Known limitation of the Cartesian version: v_l, v_b are tangential components
along each star's own line of sight, so two stars with the same 3-D space
velocity but widely separated on the sky have different (v_l, v_b). This is
negligible for compact groups (a few tens of pc at >=100 pc) and gets worse for
nearby, extended ones.
"""
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from sklearn.preprocessing import RobustScaler

HR21_COLUMNS = ["l", "b", "parallax_corrected", "pm_l", "pm_b"]

KMS_PER_MASYR_KPC = 4.740470446   # v[km/s] = 4.74047 * pm[mas/yr] / parallax[mas]
PC_PER_MYR_PER_KMS = 1.0227121650537077


def add_galactic_pm(df: pd.DataFrame) -> pd.DataFrame:
    """Add pm_l (already x cos b) and pm_b in mas/yr from ICRS proper motions."""
    c = SkyCoord(ra=df.ra.to_numpy() * u.deg, dec=df.dec.to_numpy() * u.deg,
                 pm_ra_cosdec=df.pmra.to_numpy() * u.mas / u.yr,
                 pm_dec=df.pmdec.to_numpy() * u.mas / u.yr, frame="icrs")
    g = c.galactic
    out = df.copy()
    out["pm_l"] = g.pm_l_cosb.to_value(u.mas / u.yr)
    out["pm_b"] = g.pm_b.to_value(u.mas / u.yr)
    return out


def to_features(df: pd.DataFrame) -> np.ndarray:
    """Return an (N, 5) array of robust-scaled [l, b, parallax_corrected,
    pm_l, pm_b] (Hunt & Reffert 2021). The scaler is fit on df itself, so
    call it once per tile/field. l is used as-is: a tile that crosses
    l = 0/360 must be re-wrapped before calling this.
    """
    return RobustScaler().fit_transform(df[HR21_COLUMNS].to_numpy())


def to_features_cartesian(df: pd.DataFrame, plx_col: str = "parallax_corrected",
                          tau_myr: float = 10.0) -> np.ndarray:
    """Ablation only. Return an (N, 5) array [x, y, z, tau*v_l, tau*v_b] in parsecs.

    tau_myr converts velocity (pc/Myr) into a length so position and velocity
    share one scale: a velocity spread of 1 km/s ~ 1.02 pc/Myr becomes
    ~10 pc at tau=10 Myr. This is the most important tunable in the pipeline;
    it is ablated, not guessed.
    Requires columns: l, b (deg), plx_col (mas), pm_l, pm_b (mas/yr).
    """
    d = 1000.0 / df[plx_col].to_numpy()
    l = np.deg2rad(df.l.to_numpy())
    b = np.deg2rad(df.b.to_numpy())
    x = d * np.cos(b) * np.cos(l)
    y = d * np.cos(b) * np.sin(l)
    z = d * np.sin(b)
    kms = KMS_PER_MASYR_KPC / df[plx_col].to_numpy()   # km/s per (mas/yr)
    vl = df.pm_l.to_numpy() * kms * PC_PER_MYR_PER_KMS  # pc/Myr
    vb = df.pm_b.to_numpy() * kms * PC_PER_MYR_PER_KMS
    return np.column_stack([x, y, z, tau_myr * vl, tau_myr * vb])
