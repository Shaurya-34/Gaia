"""Figure 6: CST distributions of real clusters, injected clusters and shuffled
data, with the thresholds tau = 3, 5, 7 and the chosen operating tau."""
import json

import numpy as np
import pandas as pd

from figstyle import BLUE, INK, INK2, MUTED, ROOT, VIOLET, plt, save

REAL_MATCH_J = 0.5  # a real-data group counts as a known cluster at Jaccard >= 0.5 with a Hunt & Reffert 2023 cluster


def main():
    real = pd.read_csv(ROOT / "results" / "shuffle" / "real_groups.csv")
    shuf = pd.read_csv(ROOT / "results" / "shuffle" / "shuffle_groups.csv")
    summary = json.loads((ROOT / "results" / "shuffle" / "summary.json").read_text())
    trials = pd.read_csv(ROOT / "results" / "main" / "trials.csv")
    known = real[real.hr23_jaccard >= REAL_MATCH_J].cst
    injected = trials.matched_cst.dropna()
    bins = np.linspace(0, max(40, np.nanmax([known.max() if len(known) else 0, injected.max(), shuf.cst.max()])), 41)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(shuf.cst, bins=bins, histtype="stepfilled", color="#d9d8d2", label=f"shuffled data, all groups ({len(shuf)})")
    ax.hist(injected, bins=bins, histtype="step", color=VIOLET, lw=2, label=f"injected clusters, matched group ({len(injected)})")
    ax.hist(known, bins=bins, histtype="step", color=BLUE, lw=2,
            label=f"real data, groups matching a known cluster ({len(known)})")
    ax.set_yscale("log")
    ymax = ax.get_ylim()[1]
    op = summary.get("operating_tau")
    for t in (3, 5, 7):
        ax.axvline(t, color=INK if t == op else MUTED, lw=1.6 if t == op else 0.8, ls="-" if t == op else "--")
        ax.text(t + 0.3, ymax * 0.6, f"tau = {t}" + (" (operating)" if t == op else ""), rotation=90, va="top",
                fontsize=7, color=INK if t == op else INK2)
    ax.set_xlabel("cluster significance test (CST, sigma)")
    ax.set_ylabel("groups per bin (log scale)")
    ax.set_title("CST of real, injected and shuffled groups in block 6")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    data = pd.concat([pd.DataFrame({"kind": "shuffle", "cst": shuf.cst}),
                      pd.DataFrame({"kind": "injected", "cst": injected}),
                      pd.DataFrame({"kind": "real_known", "cst": known})], ignore_index=True)
    op_text = f"tau = {op} was chosen as the operating threshold by the PREREG section 8 rule" if op else \
        "no tau in {3, 5, 7} met the false-positive budget, so no operating threshold is chosen"
    save(fig, "fig6_cst_hist",
         "Cluster significance test values for groups found in shuffled data (all groups: the null), for the "
         "matched groups of injected clusters (main run), and for real-data groups that match a Hunt & Reffert 2023 "
         f"cluster (Jaccard >= {REAL_MATCH_J}) in block 6's four fields. {op_text}. Limitation: shuffling within "
         "level-5 pixel x parallax cells cannot fully erase a group that fills its own cells, so the null is "
         "conservative; four overlapping fields only.", data)


if __name__ == "__main__":
    main()
