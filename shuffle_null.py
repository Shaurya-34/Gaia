"""Shuffle-decoy null (PREREG section 8): permute proper motions among stars
in the same local cell (level-5 Galactic HEALPix pixel x parallax-quantile bin
within that pixel). Positions and parallaxes stay put, so density is preserved;
co-moving structure is broken.

A group that fills most of its own cells survives shuffling, so very nearby
clusters can leak into the null; that makes the false-positive count
conservative, not optimistic.
"""
import numpy as np
import pandas as pd

from config import load_config
from splits import galactic_block

CFG = load_config()["shuffle_null"]
PM_COLUMNS = ["pm_l", "pm_b", "pm_l_error", "pm_b_error", "pmra_error", "pmdec_error"]


def shuffle_cells(df: pd.DataFrame) -> np.ndarray:
    """Integer cell id per star: level-5 pixel x parallax-quantile bin."""
    pix = galactic_block(df.l.to_numpy(), df.b.to_numpy(), level=CFG["cell_level"])
    n_bins = CFG["parallax_quantile_bins"]
    rank = df.groupby(pix).parallax_corrected.rank(method="first", pct=True).to_numpy()
    plx_bin = np.minimum((rank * n_bins).astype(int), n_bins - 1)
    return pix * n_bins + plx_bin


def shuffle_proper_motions(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Copy of df with each star's proper motion (and its errors) swapped with
    another star's from the same cell."""
    cols = [c for c in PM_COLUMNS if c in df.columns]
    cells = shuffle_cells(df)
    order = np.argsort(cells, kind="stable")
    sorted_cells = cells[order]
    starts = np.flatnonzero(np.r_[True, sorted_cells[1:] != sorted_cells[:-1]])
    perm = order.copy()
    for s, e in zip(starts, np.r_[starts[1:], len(order)]):
        perm[s:e] = rng.permutation(order[s:e])
    values = df[cols].to_numpy()
    shuffled = np.empty_like(values)
    shuffled[order] = values[perm]  # star order[k] takes star perm[k]'s values
    out = df.copy()
    out[cols] = shuffled
    return out
