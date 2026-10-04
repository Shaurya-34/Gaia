import json

import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
import astropy.units as u

import baseline as bl
from scoring import jaccard
from splits import galactic_block

SPLITS = json.loads(open(bl.__file__.replace("baseline.py", "splits.json")).read())
CENTRE = SPLITS["eligible_centres"][str(SPLITS["blocks"]["train"][0])][0]["pixel"]
LEVEL = SPLITS["field_level"]


def planted(rng, l, b, n, start_id, plx=5.0, pm=(10.0, -5.0)):
    """A compact co-moving group at (l, b), about 200 pc away."""
    return pd.DataFrame({
        "source_id": np.arange(start_id, start_id + n),
        "l": (l + rng.normal(0, 0.4, n)) % 360, "b": b + rng.normal(0, 0.4, n),
        "parallax_corrected": rng.normal(plx, 0.05, n),
        "pm_l": rng.normal(pm[0], 0.3, n), "pm_b": rng.normal(pm[1], 0.3, n),
    })


@pytest.fixture(scope="module")
def sky():
    """Uniform field over the 9 field pixels, a cluster at the central pixel's
    centre and one at a neighbouring pixel's centre."""
    rng = np.random.default_rng(0)
    n = 400_000
    l = rng.uniform(0, 360, n)
    b = np.rad2deg(np.arcsin(rng.uniform(-1, 1, n)))
    keep = np.isin(galactic_block(l, b, level=LEVEL), bl.field_pixels(CENTRE))
    m = int(keep.sum())
    field = pd.DataFrame({
        "source_id": np.arange(1, m + 1), "l": l[keep], "b": b[keep],
        "parallax_corrected": rng.uniform(2, 10, m),
        "pm_l": rng.normal(0, 20, m), "pm_b": rng.normal(0, 20, m),
    })
    lc, bc = bl.pixel_centre(CENTRE)
    neighbour = int(bl.field_pixels(CENTRE)[1])
    ln, bn = bl.pixel_centre(neighbour)
    a = planted(rng, lc, bc, 80, 10_000_000)
    c = planted(rng, ln, bn, 80, 20_000_000, pm=(-12.0, 8.0))
    return pd.concat([field, a, c], ignore_index=True), a, c


def test_field_is_centre_plus_8_neighbours():
    pix = bl.field_pixels(CENTRE)
    assert len(pix) == 9 and pix[0] == CENTRE and len(set(pix)) == 9


def test_select_field_keeps_only_field_pixels(sky):
    df, _, _ = sky
    shifted = df.assign(l=(df.l + 180) % 360)  # move everything to the other side of the sky
    assert len(bl.select_field(shifted, CENTRE)) < 0.01 * len(df)
    assert len(bl.select_field(df, CENTRE)) == len(df)


def test_recentre_moves_centre_to_origin_and_keeps_geometry():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"l": rng.uniform(100, 140, 500), "b": rng.uniform(-20, 20, 500),
                       "pm_l": rng.normal(0, 10, 500), "pm_b": rng.normal(0, 10, 500)})
    r = bl.recentre(df, 120.0, 0.0)
    c0 = bl.recentre(pd.DataFrame({"l": [120.0], "b": [0.0], "pm_l": [0.0], "pm_b": [0.0]}), 120.0, 0.0)
    assert abs(c0.l[0]) < 1e-9 and abs(c0.b[0]) < 1e-9
    np.testing.assert_allclose(np.hypot(r.pm_l, r.pm_b), np.hypot(df.pm_l, df.pm_b), rtol=1e-9)
    before = SkyCoord(l=df.l.to_numpy() * u.deg, b=df.b.to_numpy() * u.deg, frame="galactic")
    after = SkyCoord(r.l.to_numpy() * u.deg, r.b.to_numpy() * u.deg)
    np.testing.assert_allclose(before[0].separation(before).deg, after[0].separation(after).deg, atol=1e-9)


def test_in_central_pixel_with_tolerance():
    lc, bc = bl.pixel_centre(CENTRE)
    l_edge = lc
    while galactic_block(l_edge, bc, level=LEVEL) == CENTRE:
        l_edge += 0.01
    g = {"l": lc, "b": bc, "parallax_corrected": 5.0}  # 200 pc: 5 pc tolerance is ~1.4 deg
    assert bl.in_central_pixel(g, CENTRE)
    assert bl.in_central_pixel({**g, "l": l_edge + 0.5}, CENTRE)
    assert not bl.in_central_pixel({**g, "l": l_edge + 5.0}, CENTRE)
    assert not bl.in_central_pixel({**g, "l": l_edge + 0.5, "parallax_corrected": 0.5}, CENTRE)  # 2 kpc: 0.14 deg


def test_merge_duplicates_keeps_highest_cst():
    groups = [{"source_ids": np.arange(10), "cst": 4.0},
              {"source_ids": np.arange(2, 12), "cst": 9.0},
              {"source_ids": np.arange(100, 110), "cst": 3.0}]
    kept = bl.merge_duplicates(groups)
    assert [g["cst"] for g in kept] == [9.0, 3.0]


def blobs_in_noise(seed):
    rng = np.random.default_rng(seed)
    return np.vstack([rng.uniform(0, 1, (3000, 5)), rng.normal(0.5, 0.01, (60, 5)), rng.normal(0.2, 0.01, (40, 5))])


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("mcs", [10, 20, 40])
def test_hdbscan_package_matches_sklearn(seed, mcs):
    """Same groups as sklearn's HDBSCAN once min_samples is aligned: sklearn
    counts the point itself, so hdbscan's 10 is sklearn's 11."""
    import hdbscan
    from sklearn.cluster import HDBSCAN as SkHDBSCAN
    from sklearn.metrics import adjusted_rand_score

    X = blobs_in_noise(seed)
    h = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=10, cluster_selection_method="leaf").fit(X).labels_
    s = SkHDBSCAN(min_cluster_size=mcs, min_samples=11, cluster_selection_method="leaf", copy=True).fit(X).labels_
    assert adjusted_rand_score(h, s) > 0.95


def test_cluster_stats_map_labels_to_condensed_tree_nodes():
    import hdbscan

    X = blobs_in_noise(0)
    c = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=10, cluster_selection_method="leaf").fit(X)
    raw = c.condensed_tree_._raw_tree
    children = {}
    for p, ch in zip(raw["parent"], raw["child"]):
        children.setdefault(p, []).append(ch)

    def points(node):
        out, stack = set(), [node]
        while stack:
            for ch in children.get(stack.pop(), []):
                out.add(ch) if ch < len(X) else stack.append(ch)
        return out

    stats = bl.cluster_stats(c)
    assert sorted(stats) == list(range(c.labels_.max() + 1))
    for i, node in enumerate(sorted(c.condensed_tree_._select_clusters())):
        assert points(node) == set(np.flatnonzero(c.labels_ == i))
    assert all(np.isfinite(s) and s >= 0 and np.isfinite(p) and p >= 0 for s, p in stats.values())


def test_detects_central_cluster_and_drops_neighbour(sky):
    df, a, c = sky
    groups = bl.detect(df, CENTRE)
    best_a = max(jaccard(g["source_ids"], a.source_id) for g in groups)
    match = max(groups, key=lambda g: jaccard(g["source_ids"], a.source_id))
    assert best_a >= 0.5 and match["cst"] > 3
    assert all(jaccard(g["source_ids"], c.source_id) < 0.1 for g in groups)
    for g in groups:
        assert {"field_pixel", "min_cluster_size", "n_members", "cst", "mean_membership_probability",
                "stability", "persistence"} <= g.keys()
