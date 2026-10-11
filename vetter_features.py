import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from baseline import recentre


def _field_centre(field):
    """Mean direction of the field's stars (wrap-safe), as (l, b) in degrees."""
    l, b = np.deg2rad(field.l.to_numpy()), np.deg2rad(field.b.to_numpy())
    v = np.array([np.mean(np.cos(b) * np.cos(l)), np.mean(np.cos(b) * np.sin(l)), np.mean(np.sin(b))])
    return float(np.rad2deg(np.arctan2(v[1], v[0])) % 360), float(np.rad2deg(np.arcsin(v[2] / np.linalg.norm(v))))

def cluster_features(members: pd.DataFrame, field: pd.DataFrame, hdbscan_info: dict = None) -> dict[str, float]:
    """
    Extracts vetting features for a detected stellar group to evaluate its authenticity.

    Parameters:
    -----------
    members : pd.DataFrame
        DataFrame containing the stars belonging to the detected cluster.
        Required columns: 'l', 'b', 'parallax_corrected', 'pm_l', 'pm_b',
        'bp_rp', 'phot_g_mean_mag'
    field : pd.DataFrame
        DataFrame of the surrounding field stars; the 5D scaler is fit on it.
    hdbscan_info : dict, optional
        A dictionary containing clustering metrics like 'cst', 'stability',
        'persistence', and 'membership_probs'.

    Returns:
    --------
    dict[str, float]
        Dictionary filled with computed validation features.
    """
    if hdbscan_info is None:
        hdbscan_info = {}

    features = {}

    # 1. Member Count
    features["member_count"] = float(len(members))

    # 2-5. HDBSCAN / Cluster metrics with safe NaN fallbacks
    features["cst"] = float(hdbscan_info.get("cst", np.nan))
    features["hdbscan_stability"] = float(hdbscan_info.get("stability", np.nan))
    features["hdbscan_persistence"] = float(hdbscan_info.get("persistence", np.nan))

    # Extract membership probabilities mean
    probs = hdbscan_info.get("membership_probs", None)
    if probs is not None and len(probs) > 0:
        features["mean_membership_probability"] = float(np.mean(probs))
    else:
        features["mean_membership_probability"] = np.nan

    # 6. 5D Extent in robust-scaled feature space (median distance to the centroid)
    # Hunt & Reffert 2021 feature space, project column names (README.md).
    spatial_cols = ['l', 'b', 'parallax_corrected', 'pm_l', 'pm_b']

    if len(members) > 0:
        # Re-centre field and members on the field's centre, as the detector does
        # (baseline.recentre), so a field crossing l = 0/360 does not split in l.
        l0, b0 = _field_centre(field)
        field_rc, members_rc = recentre(field, l0, b0), recentre(members, l0, b0)
        # Fit scaler on the field so a tight group keeps a small extent
        scaler = RobustScaler().fit(field_rc[spatial_cols])
        scaled_coords = scaler.transform(members_rc[spatial_cols])

        # Calculate the centroid (median coordinates in the robust space)
        centroid = np.median(scaled_coords, axis=0)

        # Calculate Euclidean distance of every star to that centroid
        distances = np.linalg.norm(scaled_coords - centroid, axis=1)
        features["5d_extent"] = float(np.median(distances))
    else:
        features["5d_extent"] = np.nan

    # 7. Scatter of absolute G magnitude about a smooth Colour-Magnitude Relation (CMR)
    required_cmr_cols = ['phot_g_mean_mag', 'bp_rp', 'parallax_corrected']
    # A cubic fit needs at least 5 stars to leave any residual scatter
    if all(col in members.columns for col in required_cmr_cols) and len(members) >= 5:
        # Calculate absolute G magnitude: M_G = g - 5 * log10(100 / parallax)
        # Note: parallax is expected to be in milliarcseconds (mas)
        # Handle non-positive parallaxes safely to avoid log10 errors
        valid_parallax_mask = members['parallax_corrected'] > 0
        valid_members = members[valid_parallax_mask]

        if len(valid_members) >= 5:
            abs_g = valid_members['phot_g_mean_mag'] - 5 * np.log10(100.0 / valid_members['parallax_corrected'])
            color = valid_members['bp_rp']

            # Fit a degree-3 polynomial curve: absolute_G = f(color)
            poly_coeffs = np.polyfit(color, abs_g, deg=3)
            predicted_abs_g = np.polyval(poly_coeffs, color)

            # Compute residual standard deviation
            residuals = abs_g - predicted_abs_g
            # Using degrees of freedom (N - deg - 1) = N - 4
            ddof = 4 if len(residuals) > 4 else 0
            features["cmr_scatter"] = float(np.std(residuals, ddof=ddof))
        else:
            features["cmr_scatter"] = np.nan
    else:
        features["cmr_scatter"] = np.nan

    return features
