"""E3: shuffle null on real data (PREREG section 8, closure scope).

Each of block 6's four fields is searched unshuffled (real-data groups, for
figure 6) and with proper motions shuffled under 3 seeds. Every group is saved
with its cluster_features row; shuffled groups are the vetter's negatives.
False positives are counted per 10^6 central-pixel stars searched (the
detector keeps only groups centred in the central pixel). The operating tau is
the smallest tau in {3, 5, 7} meeting the false-positive budget.

Run:  python shuffle_run.py --workers 3
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import baseline
import catalogue
from config import load_config
from scoring import jaccard
from shuffle_null import shuffle_proper_motions
from splits import galactic_block
from vetter_features import cluster_features

CFG = load_config()
FIELDS = [99, 102, 105, 108]
SEEDS = list(range(CFG["shuffle_null"]["repeats_per_field"]))
OUT = Path("results/shuffle")
TAUS = CFG["search"]["cst_tau"]


def search(fields_dir, pixel, seed):
    """seed None = unshuffled real data. Returns one row per group."""
    field = pd.read_parquet(Path(fields_dir) / f"field_{pixel}.parquet")
    if seed is not None:
        field = shuffle_proper_motions(field, np.random.default_rng([20261011, pixel, seed]))
    groups = baseline.detect(field, pixel)
    rows = []
    for i, g in enumerate(groups):
        members = field[field.source_id.isin(g["source_ids"])]
        feats = cluster_features(members, field, {"cst": g["cst"], "stability": g["stability"],
                                                  "persistence": g["persistence"],
                                                  "membership_probs": [g["mean_membership_probability"]]})
        rows.append({**feats, "field_pixel": pixel, "seed": -1 if seed is None else seed, "group": i,
                     "source": "real" if seed is None else "shuffle",
                     "min_cluster_size": g["min_cluster_size"], "l": g["l"], "b": g["b"],
                     "parallax_corrected": g["parallax_corrected"],
                     "source_ids": json.dumps(list(map(int, g["source_ids"])))})
    central = int((galactic_block(field.l.to_numpy(), field.b.to_numpy(), level=2) == pixel).sum())
    return pixel, seed, rows, central


def tag_known(real, members):
    """Best Jaccard of each real-data group against Hunt & Reffert 2023 clusters."""
    by_name = members.groupby("Name").GaiaDR3.apply(set)
    best_name, best_j = [], []
    for ids in real.source_ids.map(json.loads):
        ids = set(ids)
        cand = members.loc[members.GaiaDR3.isin(ids), "Name"].unique()
        scores = {n: jaccard(list(ids), list(by_name[n])) for n in cand}
        n = max(scores, key=scores.get) if scores else None
        best_name.append(n)
        best_j.append(scores[n] if n else 0.0)
    return real.assign(hr23_match=best_name, hr23_jaccard=best_j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--fields", type=Path, default=Path("data/pilot_fields"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [(p, None) for p in FIELDS] + [(p, s) for p in FIELDS for s in SEEDS]
    rows, central = [], {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(search, str(args.fields), p, s) for p, s in tasks]
        for fut in as_completed(futs):
            pixel, seed, r, c = fut.result()
            rows += r
            central[pixel] = c
            print(f"field {pixel} {'real' if seed is None else f'shuffle seed {seed}'}: {len(r)} groups", flush=True)
    groups = pd.DataFrame(rows)
    real = tag_known(groups[groups.source == "real"], catalogue.load()[1])
    shuf = groups[groups.source == "shuffle"]
    real.to_csv(OUT / "real_groups.csv", index=False)
    shuf.to_csv(OUT / "shuffle_groups.csv", index=False)

    stars_searched = sum(central[p] for p in FIELDS) * len(SEEDS)  # central-pixel stars x seeds
    budget = CFG["shuffle_null"]["max_false_positives_per_million"]
    per_tau = {t: {"detections": int((shuf.cst > t).sum()),
                   "per_million_central_stars": float((shuf.cst > t).sum() / stars_searched * 1e6)} for t in TAUS}
    passing = [t for t in TAUS if per_tau[t]["per_million_central_stars"] <= budget]
    summary = {"fields": FIELDS, "seeds": SEEDS, "central_pixel_stars": central,
               "central_star_searches": stars_searched, "shuffle_groups": len(shuf), "real_groups": len(real),
               "false_positives_by_tau": per_tau, "budget_per_million": budget,
               "operating_tau": min(passing) if passing else None,
               "real_groups_above_tau": {t: int((real.cst > t).sum()) for t in TAUS},
               "real_groups_matching_hr23_j_ge_0.5": int((real.hr23_jaccard >= 0.5).sum())}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
