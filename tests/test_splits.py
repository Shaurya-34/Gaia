import json

import numpy as np
from astropy_healpix import healpix_to_lonlat, neighbours

import splits
from config import load_config

CFG = load_config()


def test_block_counts_and_coverage():
    blocks = splits.assign_blocks(CFG["splits"])
    assert {k: len(v) for k, v in blocks.items()} == {"train": 7, "val": 2, "test": 2, "sealed": 1}
    assert sorted(sum(blocks.values(), [])) == list(range(12))


def test_plane_in_every_split_and_sealed_equatorial():
    blocks = splits.assign_blocks(CFG["splits"])
    for split, members in blocks.items():
        assert set(members) & set(splits.EQUATORIAL_BLOCKS), split
    assert blocks["sealed"][0] in splits.EQUATORIAL_BLOCKS


def test_assignment_is_deterministic():
    assert splits.assign_blocks(CFG["splits"]) == splits.assign_blocks(CFG["splits"])


def test_equatorial_blocks_hold_the_galactic_plane():
    l = np.arange(0.5, 360, 1.0)  # avoid the block vertices at multiples of 45 deg
    assert set(splits.galactic_block(l, np.zeros_like(l)).tolist()) == set(splits.EQUATORIAL_BLOCKS)


def test_galactic_block_of_block_centres():
    lon, lat = healpix_to_lonlat(np.arange(12), 1, order="nested")
    np.testing.assert_array_equal(splits.galactic_block(lon.deg, lat.deg), np.arange(12))


def test_eligible_centres_stay_inside_their_block():
    level = CFG["search"]["field_level"]
    centres = splits.eligible_centres(level)
    for block, pixels in centres.items():
        assert len(pixels) == 4  # the inner 2x2 of each block's 4x4 level-2 pixels
        nb = neighbours(np.array(pixels), 2 ** level, order="nested")
        assert (nb >= 0).all()
        assert (nb // 4 ** level == block).all()


def test_committed_output_matches_code():
    committed = json.loads(splits.OUT_PATH.read_text())
    assert committed == json.loads(json.dumps(splits.build(CFG)))
