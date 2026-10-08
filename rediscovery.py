"""Rediscovery check before the main run (PREREG section 8, as amended 2026-10-08):
the baseline must recover the Pleiades and alpha Per (Jaccard >= 0.5 against
Hunt & Reffert 2023 members that survive our cuts); the Hyades is informational.

Each cluster is searched in the field centred on the level-2 pixel that holds
its catalogue centre, since the detector keeps only groups centred in the
field's central pixel. Groups are saved to results/rediscovery/.

Run:  python rediscovery.py
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import baseline
import catalogue
from config import load_config
from pilot import load_stars
from scoring import jaccard
from splits import galactic_block

CLUSTERS = {"Melotte_22": "Pleiades", "Melotte_20": "alpha Per", "Melotte_25": "Hyades"}
MUST_PASS = {"Melotte_22", "Melotte_20"}
OUT = Path(__file__).with_name("results") / "rediscovery"
LEVEL = load_config()["search"]["field_level"]


def check(stars, clusters, members, name):
    row = clusters[clusters.Name == name].iloc[0]
    pixel = int(galactic_block(row.GLON, row.GLAT, level=LEVEL))
    field = baseline.select_field(stars, pixel).reset_index(drop=True)
    per_pixel = pd.Series(galactic_block(field.l.to_numpy(), field.b.to_numpy(), level=LEVEL)).value_counts()
    t0 = time.perf_counter()
    groups = baseline.detect(field, pixel)
    runtime = time.perf_counter() - t0
    truth = members.loc[(members.Name == name) & members.GaiaDR3.isin(field.source_id), "GaiaDR3"].to_numpy()
    scores = [jaccard(g["source_ids"], truth) for g in groups]
    best = int(np.argmax(scores)) if groups else None
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**{k: v for k, v in g.items() if k != "source_ids"}, "group": i,
                   "source_ids": list(map(int, g["source_ids"]))} for i, g in enumerate(groups)]) \
        .to_json(OUT / f"{name}_groups.json", orient="records")
    return {
        "cluster": name, "common_name": CLUSTERS[name], "field_pixel": pixel, "field_stars": len(field),
        "stars_per_field_pixel": {int(k): int(v) for k, v in per_pixel.sort_index().items()},
        "catalogue_members": int((members.Name == name).sum()), "members_after_cuts": len(truth),
        "groups": len(groups), "best_jaccard": round(scores[best], 3) if groups else 0.0,
        "best_cst": round(groups[best]["cst"], 2) if groups else None,
        "best_n_members": groups[best]["n_members"] if groups else 0,
        "best_min_cluster_size": groups[best]["min_cluster_size"] if groups else None,
        "passes": bool(groups and scores[best] >= 0.5), "must_pass": name in MUST_PASS,
        "runtime_s": round(runtime),
    }


def main():
    stars = load_stars(Path("data/bulk"))
    clusters, members = catalogue.load()
    results = []
    for name in CLUSTERS:
        r = check(stars, clusters, members, name)
        results.append(r)
        print(json.dumps(r), flush=True)
    (OUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    failed = [r["common_name"] for r in results if r["must_pass"] and not r["passes"]]
    print("GATE " + ("FAILED: " + ", ".join(failed) if failed else "PASSED (Pleiades and alpha Per)"))


if __name__ == "__main__":
    main()
