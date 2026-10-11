"""Figure 3: colour-magnitude diagrams of the recovered Pleiades and alpha Per
members, beside an injected cluster, whose twin-star magnitudes and colours
look field-like (a known limitation of the injection, not a result)."""
import json

import numpy as np
import pandas as pd

from figstyle import AQUA, BLUE, FIELD, ROOT, VIOLET, plt, save

import baseline
import catalogue
import injection
from pilot import load_stars
from scoring import jaccard

CLUSTERS = {"Melotte_22": "Pleiades", "Melotte_20": "alpha Per"}
INJECTED = {"injection_id": 0, "field_pixel": 102, "dist_pc": 150.0, "n_members": 300,
            "r_half_pc": 2.0, "sigma_v_kms": 0.5}
SEED = [20261011, 3]  # Code choice for this illustration


def abs_g(df):
    return df.phot_g_mean_mag + 5 * np.log10(df.parallax_corrected / 100.0)


def main():
    stars = load_stars(ROOT / "data" / "bulk")
    clusters, members = catalogue.load()
    fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True, sharex=True)
    rows = []
    for ax, (name, label) in zip(axes[:2], CLUSTERS.items()):
        truth = members.loc[(members.Name == name) & members.GaiaDR3.isin(stars.source_id), "GaiaDR3"].to_numpy()
        groups = json.loads((ROOT / "results" / "rediscovery" / f"{name}_groups.json").read_text())
        best = max(groups, key=lambda g: jaccard(g["source_ids"], truth))
        rec = stars[stars.source_id.isin(best["source_ids"])]
        ax.scatter(rec.bp_rp, abs_g(rec), s=5, color=BLUE, lw=0)
        ax.set_title(f"{label}: recovered members ({len(rec):,})")
        rows.append(rec[["source_id", "bp_rp", "phot_g_mean_mag", "parallax_corrected"]].assign(group=label))

    field = pd.read_parquet(ROOT / "data" / "pilot_fields" / "field_102.parquet")
    l0, b0 = baseline.pixel_centre(102)
    p = dict(INJECTED, l_deg=l0, b_deg=b0)
    with_cluster, _ = injection.inject(field, p, np.random.default_rng(SEED))
    inj = with_cluster[with_cluster.is_injected]
    ax = axes[2]
    bg = field.sample(20_000, random_state=0)
    ax.scatter(bg.bp_rp, abs_g(bg), s=2, color=FIELD, lw=0, label="field stars (sample)", rasterized=True)
    ax.scatter(inj.bp_rp, abs_g(inj), s=6, color=VIOLET, lw=0, label="injected members")
    ax.set_title(f"Injected cluster ({len(inj)} members, 150 pc)")
    ax.legend(loc="lower left", fontsize=7, markerscale=2)
    rows.append(inj[["source_id", "bp_rp", "phot_g_mean_mag", "parallax_corrected"]].assign(group="injected"))
    for ax in axes:
        ax.set_xlabel("BP - RP (mag)")
    axes[0].set_ylabel("absolute G (mag)")
    axes[0].invert_yaxis()
    axes[0].set_xlim(-0.3, 4.0)
    fig.tight_layout()
    save(fig, "fig3_cmd",
         "Colour-magnitude diagrams (absolute G from the corrected parallax, no extinction correction). The "
         "Pleiades and alpha Per members recovered by the baseline trace tight single-age sequences; the injected "
         "cluster (right) does not, because each injected member copies the magnitude and colour of a random nearby "
         "real 'twin' star. Limitation: injected clusters have field-like CMDs, so CMD tightness is not used to "
         "train the vetter and survival against distance is not realistic.",
         pd.concat(rows, ignore_index=True))


if __name__ == "__main__":
    main()
