"""Figure 1: star density of block 6 in Galactic l, b, with the four pilot
fields' central pixels outlined and the Pleiades, alpha Per and Hyades marked."""
import glob

import astropy.units as u
import numpy as np
import pandas as pd
from astropy_healpix import boundaries_lonlat

from figstyle import AQUA, BLUE, INK, INK2, ROOT, VIOLET, BLUE_RAMP, plt, save

import catalogue
from splits import galactic_block

FIELDS = [99, 102, 105, 108]
CLUSTERS = {"Melotte_22": "Pleiades", "Melotte_20": "alpha Per", "Melotte_25": "Hyades"}


def main():
    stars = pd.concat([pd.read_parquet(f, columns=["l", "b"])
                       for f in glob.glob(str(ROOT / "data" / "bulk" / "GaiaSource_*.parquet"))])
    stars = stars[galactic_block(stars.l.to_numpy(), stars.b.to_numpy()) == 6]
    l_edges, b_edges = np.arange(135, 225.01, 0.5), np.arange(-42, 42.01, 0.5)
    counts, _, _ = np.histogram2d(stars.l, stars.b, bins=[l_edges, b_edges])

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    density = counts.T / 0.25  # stars per square degree (0.5 x 0.5 deg bins, ignoring cos b)
    density[density == 0] = np.nan
    lo, hi = np.nanpercentile(density, [1, 99])
    mesh = ax.pcolormesh(l_edges, b_edges, density, cmap=BLUE_RAMP, vmin=lo, vmax=hi, rasterized=True)
    cb = fig.colorbar(mesh, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("stars per deg² (after cuts, < 500 pc)", color=INK2)
    cb.outline.set_visible(False)

    lon, lat = boundaries_lonlat([6], step=8, nside=1, order="nested")
    ax.plot(np.r_[lon.to_value(u.deg)[0], lon.to_value(u.deg)[0][0]],
            np.r_[lat.to_value(u.deg)[0], lat.to_value(u.deg)[0][0]], color=INK2, lw=1, label="block 6")
    lon, lat = boundaries_lonlat(FIELDS, step=8, nside=4, order="nested")
    for i, pix in enumerate(FIELDS):
        x, y = lon.to_value(u.deg)[i], lat.to_value(u.deg)[i]
        ax.plot(np.r_[x, x[0]], np.r_[y, y[0]], color=VIOLET, lw=1.5, label="pilot field (central pixel)" if i == 0 else None)
        ax.text(x.mean(), y.mean(), f"field {pix}", color=VIOLET, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

    clusters, _ = catalogue.load()
    rows = []
    # label offsets (points) chosen so the three labels do not collide or leave the axes
    offsets = {"Melotte_22": (8, -14, "left"), "Melotte_20": (-8, 10, "right"), "Melotte_25": (8, 8, "left")}
    for name, label in CLUSTERS.items():
        c = clusters[clusters.Name == name].iloc[0]
        text = f"{label} ({c.dist50:.0f} pc)"
        if name == "Melotte_25":
            text = f"Hyades ({c.dist50:.0f} pc,\njust outside the 50 pc edge)"
        dx, dy, ha = offsets[name]
        ax.scatter(c.GLON, c.GLAT, s=60, facecolor="none", edgecolor=INK, lw=1.5, zorder=5)
        ax.annotate(text, (c.GLON, c.GLAT), xytext=(dx, dy), textcoords="offset points", color=INK, fontsize=8,
                    ha=ha, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        rows.append({"name": label, "l": c.GLON, "b": c.GLAT, "dist50_pc": c.dist50})

    ax.set_xlim(225, 135)  # Galactic longitude increases to the left
    ax.set_ylim(-42, 42)
    ax.set_xlabel("Galactic longitude l (deg)")
    ax.set_ylabel("Galactic latitude b (deg)")
    ax.set_title("Block 6: Gaia DR3 stars within 500 pc after quality cuts")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(False)
    data = pd.DataFrame({"l_lo": np.repeat(l_edges[:-1], len(b_edges) - 1),
                         "b_lo": np.tile(b_edges[:-1], len(l_edges) - 1), "count": counts.ravel()})
    data = pd.concat([data, pd.DataFrame(rows)], ignore_index=True)
    save(fig, "fig1_sky",
         f"Star density of block 6 ({len(stars):,} stars after the PREREG cuts, parallax >= 2 mas) in Galactic "
         "coordinates, with the central pixels of the four pilot fields outlined and three known clusters marked. "
         "Limitation: one region of the sky only; the Hyades (47 pc) lies just outside the 50-500 pc search range "
         "and is shown for reference.", data)


if __name__ == "__main__":
    main()
