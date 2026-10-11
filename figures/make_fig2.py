"""Figure 2: the Pleiades and alpha Per in proper-motion space and parallax,
against field stars in the same sky patch; HDBSCAN-recovered members vs
Hunt & Reffert 2023 members (after our cuts)."""
import json

import numpy as np
import pandas as pd

from figstyle import BLUE, FIELD, INK2, ROOT, VIOLET, plt, save

import catalogue
from pilot import load_stars
from scoring import jaccard

CLUSTERS = {"Melotte_22": "Pleiades", "Melotte_20": "alpha Per"}


def best_group(name, truth):
    groups = json.loads((ROOT / "results" / "rediscovery" / f"{name}_groups.json").read_text())
    return max(groups, key=lambda g: jaccard(g["source_ids"], truth))


def sep_deg(l1, b1, l2, b2):
    l1, b1, l2, b2 = map(np.deg2rad, (l1, b1, l2, b2))
    return np.rad2deg(np.arccos(np.clip(np.sin(b1) * np.sin(b2) + np.cos(b1) * np.cos(b2) * np.cos(l1 - l2), -1, 1)))


def main():
    stars = load_stars(ROOT / "data" / "bulk")
    clusters, members = catalogue.load()
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.5))
    rows = []
    for col, (name, label) in enumerate(CLUSTERS.items()):
        c = clusters[clusters.Name == name].iloc[0]
        truth = members.loc[(members.Name == name) & members.GaiaDR3.isin(stars.source_id), "GaiaDR3"].to_numpy()
        g = best_group(name, truth)
        found = np.asarray(g["source_ids"])
        patch = stars[sep_deg(stars.l, stars.b, c.GLON, c.GLAT) < 3 * c.r50]
        rec = patch[patch.source_id.isin(found)]
        cat = patch[patch.source_id.isin(truth)]
        field = patch[~patch.source_id.isin(found) & ~patch.source_id.isin(truth)]
        pm0 = rec[["pm_l", "pm_b"]].median()
        ax = axes[0, col]
        ax.scatter(field.pm_l, field.pm_b, s=2, color=FIELD, lw=0, label=f"field stars in the patch ({len(field):,})",
                   rasterized=True)
        ax.scatter(cat.pm_l, cat.pm_b, s=16, facecolor="none", edgecolor=VIOLET, lw=0.5,
                   label=f"Hunt & Reffert 2023 members ({len(cat):,})")
        ax.scatter(rec.pm_l, rec.pm_b, s=4, color=BLUE, lw=0, label=f"recovered by HDBSCAN ({len(rec):,})")
        ax.set_xlim(pm0.pm_l - 12, pm0.pm_l + 12)
        ax.set_ylim(pm0.pm_b - 12, pm0.pm_b + 12)
        ax.set_xlabel("pm_l (mas/yr)")
        ax.set_ylabel("pm_b (mas/yr)")
        ax.set_title(f"{label}: Jaccard {jaccard(found, truth):.2f}, CST {g['cst']:.1f}")
        ax.legend(loc="lower left", fontsize=7, markerscale=2)
        ax = axes[1, col]
        bins = np.linspace(2, 12, 101)
        ax.hist(field.parallax_corrected, bins=bins, histtype="step", color=FIELD, lw=1.5, label="field stars in the patch")
        ax.hist(rec.parallax_corrected, bins=bins, histtype="step", color=BLUE, lw=2, label="recovered by HDBSCAN")
        ax.hist(cat.parallax_corrected, bins=bins, histtype="step", color=VIOLET, lw=1.2, ls="--",
                label="Hunt & Reffert 2023 members")
        ax.set_xlabel("corrected parallax (mas)")
        ax.set_ylabel("stars per 0.1 mas bin")
        ax.legend(loc="upper right", fontsize=7)
        sample = pd.concat([field.sample(min(len(field), 20_000), random_state=0).assign(kind="field"),
                            rec.assign(kind="recovered"), cat.assign(kind="catalogue")])
        rows.append(sample[["source_id", "l", "b", "parallax_corrected", "pm_l", "pm_b", "kind"]].assign(cluster=label))
    fig.tight_layout()
    save(fig, "fig2_pm_parallax",
         "The Pleiades and alpha Per as the baseline finds them: proper motions (top) and corrected parallaxes "
         "(bottom) of the HDBSCAN-recovered group, Hunt & Reffert 2023 members surviving our cuts, and field stars "
         "within three half-number radii. Rings without a blue dot are catalogue members the detector missed. "
         "Limitation: Jaccard is measured against one catalogue's membership, which is itself a model.",
         pd.concat(rows, ignore_index=True))


if __name__ == "__main__":
    main()
