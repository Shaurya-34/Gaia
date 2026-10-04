"""HDBSCAN + CST baseline detector (PREREG section 3).

One field = a level-2 Galactic HEALPix pixel plus its 8 neighbours. The field
is re-centred on its central pixel and rotated so that pixel sits at (0, 0)
(positions and proper motions), robust-scaled in the Hunt & Reffert 2021 space,
and clustered with HDBSCAN at each min_cluster_size. Every group gets a CST
(ocelot's Mann-Whitney test). Groups whose centroid is outside the central
pixel (with a tolerance in pc) are dropped, and duplicates across runs are
merged, keeping the highest CST.

`df` columns: source_id, l, b, parallax_corrected, pm_l, pm_b (README.md).
"""
import astropy.units as u
import hdbscan
import numpy as np
from astropy.coordinates import SkyCoord
from astropy_healpix import healpix_to_lonlat, neighbours
from hdbscan._hdbscan_tree import compute_stability

from config import load_config
from features import to_features
from ocelot_cst import cluster_significance_test
from scoring import jaccard
from splits import galactic_block

CFG = load_config()["search"]


def field_pixels(centre, level=CFG["field_level"]):
    """The central pixel and its (up to 8) neighbours."""
    nb = neighbours(np.array([centre]), 2 ** level, order="nested").ravel()
    return np.r_[centre, nb[nb >= 0]]


def select_field(df, centre, level=CFG["field_level"]):
    pix = galactic_block(df.l.to_numpy(), df.b.to_numpy(), level=level)
    return df[np.isin(pix, field_pixels(centre, level))]


def pixel_centre(centre, level=CFG["field_level"]):
    lon, lat = healpix_to_lonlat(centre, 2 ** level, order="nested")
    return float(lon.to_value(u.deg)), float(lat.to_value(u.deg))


def recentre(df, l0, b0):
    """Copy of df with l, b, pm_l, pm_b in a frame rotated so (l0, b0) is at
    (0, 0). l is wrapped to (-180, 180]."""
    c = SkyCoord(l=df.l.to_numpy() * u.deg, b=df.b.to_numpy() * u.deg,
                 pm_l_cosb=df.pm_l.to_numpy() * u.mas / u.yr, pm_b=df.pm_b.to_numpy() * u.mas / u.yr,
                 frame="galactic")
    origin = SkyCoord(l=l0 * u.deg, b=b0 * u.deg, frame="galactic")
    r = c.transform_to(origin.skyoffset_frame())
    out = df.copy()
    out["l"] = r.lon.wrap_at(180 * u.deg).deg
    out["b"] = r.lat.deg
    out["pm_l"] = r.pm_lon_coslat.to_value(u.mas / u.yr)
    out["pm_b"] = r.pm_lat.to_value(u.mas / u.yr)
    return out


def cluster_stats(clusterer):
    """Per-label (stability, persistence) from an hdbscan.HDBSCAN fit. Label i
    is the i-th selected node of the condensed tree in sorted order (checked
    in tests/test_baseline.py, since these are private hdbscan helpers)."""
    tree = clusterer.condensed_tree_
    selected = sorted(tree._select_clusters())
    stability = compute_stability(tree._raw_tree)
    return {i: (float(stability[node]), float(clusterer.cluster_persistence_[i]))
            for i, node in enumerate(selected)}


def run_hdbscan(df_field, centre, level=CFG["field_level"]):
    """All groups from every min_cluster_size run on one field, with CSTs."""
    l0, b0 = pixel_centre(centre, level)
    X = to_features(recentre(df_field, l0, b0))
    ids = df_field.source_id.to_numpy()
    groups = []
    for mcs in CFG["min_cluster_size"]:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=CFG["min_samples"],
                                    cluster_selection_method=CFG["cluster_selection_method"]).fit(X)
        labels = clusterer.labels_
        if labels.max() < 0:
            continue
        stats = cluster_stats(clusterer)
        cst, _ = cluster_significance_test(X, labels, min_samples=CFG["min_samples"], knn_kwargs=CFG["cst_knn"])
        for lab in np.unique(labels[labels >= 0]):
            m = labels == lab
            members = df_field[m]
            groups.append({
                "field_pixel": int(centre), "min_cluster_size": mcs,
                "source_ids": ids[m], "n_members": int(m.sum()), "cst": float(cst[lab]),
                "stability": stats[lab][0], "persistence": stats[lab][1],
                "mean_membership_probability": float(clusterer.probabilities_[m].mean()),
                "l": float(np.median(members.l)), "b": float(np.median(members.b)),
                "parallax_corrected": float(np.median(members.parallax_corrected)),
                "pm_l": float(np.median(members.pm_l)), "pm_b": float(np.median(members.pm_b)),
            })
    return groups


def in_central_pixel(group, centre, level=CFG["field_level"], tol_pc=CFG["centroid_tolerance_pc"]):
    """True if the centroid, or any point tol_pc from it on the sky, is in the
    central pixel. The tolerance is checked at 8 points on a circle."""
    tol_deg = np.rad2deg(tol_pc / (1000.0 / group["parallax_corrected"]))
    theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    b = np.r_[group["b"], np.clip(group["b"] + tol_deg * np.sin(theta), -90, 90)]
    l = np.r_[group["l"], group["l"] + tol_deg * np.cos(theta) / max(np.cos(np.deg2rad(group["b"])), 1e-6)] % 360
    return bool((galactic_block(l, b, level=level) == centre).any())


def merge_duplicates(groups, min_jaccard=CFG["duplicate_jaccard"]):
    """Greedy by CST: keep a group unless it overlaps (Jaccard >= min_jaccard)
    a group already kept."""
    kept = []
    for g in sorted(groups, key=lambda g: -g["cst"]):
        if all(jaccard(g["source_ids"], k["source_ids"]) < min_jaccard for k in kept):
            kept.append(g)
    return kept


def detect(df, centre, level=CFG["field_level"]):
    """Run the baseline on the field around `centre`; returns a list of groups.
    Detection at a CST threshold tau is applied downstream."""
    df_field = select_field(df, centre, level)
    groups = [g for g in run_hdbscan(df_field, centre, level) if in_central_pixel(g, centre, level)]
    return merge_duplicates(groups)
