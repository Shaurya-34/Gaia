"""Sky splits (PREREG section 7): level-0 HEALPix blocks in the Galactic frame,
eligible level-2 injection centres, and the train/val/test/sealed assignment.

Run:  python splits.py   -> writes splits.json (committed with PREREG v1)
"""
import json
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy_healpix import healpix_to_lonlat, lonlat_to_healpix, neighbours

from config import load_config

EQUATORIAL_BLOCKS = [4, 5, 6, 7]  # level-0 pixels straddling b = 0 (Galactic frame)
OUT_PATH = Path(__file__).with_name("splits.json")


def galactic_block(l_deg, b_deg, level=0):
    """Nested HEALPix index (Galactic frame) of each (l, b) at `level`."""
    nside = 2 ** level
    return lonlat_to_healpix(np.asarray(l_deg) * u.deg, np.asarray(b_deg) * u.deg, nside, order="nested")


def eligible_centres(field_level):
    """Level-`field_level` pixels whose 8 neighbours all share their level-0
    parent, so the whole field lies inside one block. Returns {block: [pixels]}."""
    nside = 2 ** field_level
    per_block = 4 ** field_level
    pix = np.arange(12 * per_block)
    nb = neighbours(pix, nside, order="nested")  # (8, n); -1 where a pixel has only 7
    ok = ((nb >= 0) & (nb // per_block == pix // per_block)).all(axis=0)
    out = {b: [] for b in range(12)}
    for p in pix[ok]:
        out[int(p // per_block)].append(int(p))
    return out


def assign_blocks(cfg_splits):
    """One equatorial block per split (so the plane is in every split and the
    sealed block is equatorial); the remaining blocks fill the counts."""
    counts = {"train": cfg_splits["n_train"], "val": cfg_splits["n_val"],
              "test": cfg_splits["n_test"], "sealed": cfg_splits["n_sealed"]}
    assert sum(counts.values()) == 12 and len(counts) == len(EQUATORIAL_BLOCKS)
    rng = np.random.default_rng(cfg_splits["seed"])
    equatorial = rng.permutation(EQUATORIAL_BLOCKS)
    others = list(rng.permutation([b for b in range(12) if b not in EQUATORIAL_BLOCKS]))
    assignment = {}
    for split, eq in zip(counts, equatorial):
        assignment[split] = [int(eq)] + [int(others.pop()) for _ in range(counts[split] - 1)]
    return {split: sorted(blocks) for split, blocks in assignment.items()}


def build(cfg):
    level = cfg["search"]["field_level"]
    centres = eligible_centres(level)
    lon, lat = healpix_to_lonlat(np.arange(12 * 4 ** level), 2 ** level, order="nested")
    return {
        "frame": cfg["healpix_frame"],
        "seed": cfg["splits"]["seed"],
        "blocks": assign_blocks(cfg["splits"]),
        "field_level": level,
        "eligible_centres": {
            str(b): [{"pixel": p, "l_deg": round(float(lon[p].deg), 6), "b_deg": round(float(lat[p].deg), 6)}
                     for p in pixels]
            for b, pixels in centres.items()
        },
    }


def main():
    cfg = load_config()
    assert cfg["healpix_frame"] == "galactic"
    splits = build(cfg)
    OUT_PATH.write_text(json.dumps(splits, indent=2) + "\n")
    print(json.dumps(splits["blocks"]))
    print({b: len(c) for b, c in splits["eligible_centres"].items()})


if __name__ == "__main__":
    main()
