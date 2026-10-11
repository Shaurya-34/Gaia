"""Figure 4: one field, four injected clusters (5, 12, 25 and 100 members),
searched one at a time; injected members found vs missed, with Jaccard and CST."""
import numpy as np
import pandas as pd

from figstyle import AQUA, BLUE, FIELD, ROOT, VIOLET, plt, save

import baseline
import injection
from scoring import match_injection

PIXEL = 102
N_VALUES = [5, 12, 25, 100]
SEED = 20261012  # Code choice for this illustration


def main():
    field = pd.read_parquet(ROOT / "data" / "pilot_fields" / f"field_{PIXEL}.parquet")
    l0, b0 = baseline.pixel_centre(PIXEL)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6))
    rows = []
    for ax, n in zip(axes, N_VALUES):
        p = {"injection_id": n, "field_pixel": PIXEL, "dist_pc": 250.0, "l_deg": l0, "b_deg": b0,
             "n_members": n, "r_half_pc": 2.0, "sigma_v_kms": 0.5}
        with_cluster, truth = injection.inject(field, p, np.random.default_rng([SEED, n]))
        groups = baseline.detect(with_cluster, PIXEL)
        members = truth["member_source_ids"]
        idx, jac = match_injection(members, [{"members": g["source_ids"], "cst": g["cst"]} for g in groups])
        group = set(groups[idx]["source_ids"]) if idx is not None else set()
        cst = groups[idx]["cst"] if idx is not None else float("nan")
        inj = with_cluster[with_cluster.is_injected]
        found, missed = inj[inj.source_id.isin(group)], inj[~inj.source_id.isin(group)]
        extra = with_cluster[with_cluster.source_id.isin(group) & ~with_cluster.is_injected]
        near = with_cluster[(~with_cluster.is_injected) & (np.hypot(with_cluster.l - l0, with_cluster.b - b0) < 3)]
        ax.scatter(near.l, near.b, s=1, color=FIELD, lw=0, rasterized=True)
        ax.scatter(extra.l, extra.b, s=10, color=AQUA, lw=0, label="field stars in the group")
        ax.scatter(found.l, found.b, s=16, color=BLUE, lw=0, label="injected, found")
        ax.scatter(missed.l, missed.b, s=16, facecolor="none", edgecolor=VIOLET, lw=1, label="injected, missed")
        detected = idx is not None and cst > 3
        ax.set_title(f"n* = {truth['n_surviving']}: {'found' if detected else 'missed'}\n"
                     f"Jaccard {jac:.2f}, CST {cst:.1f}" if idx is not None else
                     f"n* = {truth['n_surviving']}: missed\nno matching group", fontsize=9)
        ax.set_xlim(l0 + 1.5, l0 - 1.5)
        ax.set_ylim(b0 - 1.5, b0 + 1.5)
        ax.set_aspect("equal")
        ax.set_xlabel("l (deg)")
        rows.append({"n_injected": n, "n_surviving": truth["n_surviving"], "jaccard": jac, "cst": cst,
                     "detected_tau3": detected, "group_field_stars": len(extra)})
    axes[0].set_ylabel("b (deg)")
    axes[-1].legend(loc="lower right", fontsize=7, markerscale=1)
    fig.tight_layout()
    save(fig, "fig4_injection_demo",
         f"Four clusters injected one at a time at 250 pc into field {PIXEL} (r_half 2 pc), shown on the sky. "
         "Blue: injected members the detector put in the matched group; violet rings: injected members it missed; "
         "aqua: real field stars that joined the group. Titles give the surviving member count, the Jaccard overlap "
         "with the injected members and the group's CST (detected means CST > 3). Limitation: single draws "
         "illustrate the method; detection rates come from the main run (figure 5).",
         pd.DataFrame(rows))


if __name__ == "__main__":
    main()
