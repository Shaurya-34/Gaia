"""Offline tests of catalogue.py on synthetic catalogue tables."""
import json

import numpy as np
import pandas as pd

import catalogue as cat
from splits import healpix_to_lonlat

SPLITS = json.loads(cat.SPLITS_PATH.read_text())["blocks"]


def block_centre(block):
    lon, lat = healpix_to_lonlat(block, 1, order="nested")
    return float(lon.deg), float(lat.deg)


def members_table(spec):
    """spec: {cluster name: source_ids}"""
    return pd.DataFrame([{"Name": n, "GaiaDR3": s} for n, ids in spec.items() for s in ids])


def test_known_overlap_discards_when_known_cluster_dominates():
    known = members_table({"Pleiades": range(100, 140), "Other": range(500, 505)})
    injected = np.arange(-20, 0)
    group = np.r_[np.arange(-10, 0), np.arange(100, 130)]  # 10 injected, 30 Pleiades
    assert cat.known_overlap(group, injected, known) == (True, "Pleiades", 30)
    group = np.r_[np.arange(-20, 0), np.arange(100, 105)]  # 20 injected, 5 Pleiades
    assert cat.known_overlap(group, injected, known) == (False, "Pleiades", 5)
    assert cat.known_overlap(np.arange(-20, 0), injected, known) == (False, None, 0)


def test_rediscovery_targets_apply_distance_members_and_sealed_rules():
    train_b, sealed_b = SPLITS["train"][0], SPLITS["sealed"][0]
    (lt, bt), (ls, bs) = block_centre(train_b), block_centre(sealed_b)
    clusters = pd.DataFrame({
        "Name": ["near_rich", "near_poor", "far_rich", "sealed_rich"],
        "dist50": [150.0, 150.0, 700.0, 150.0],
        "GLON": [lt, lt, lt, ls], "GLAT": [bt, bt, bt, bs],
    })
    known = members_table({"near_rich": range(0, 40), "near_poor": range(100, 120),
                           "far_rich": range(200, 240), "sealed_rich": range(300, 340)})
    ours = np.r_[np.arange(0, 35), np.arange(100, 120), np.arange(200, 240), np.arange(300, 340)]
    t = cat.rediscovery_targets(clusters, known, ours, min_members=30)
    assert t.Name.tolist() == ["near_rich"]
    assert t.n_surviving.tolist() == [35]  # only members present in our data count
