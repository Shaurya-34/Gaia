"""Check that the HDBSCAN tree cache changes no result (brief task P): run the
same field and injection with the cache off and on, and compare every group's
members, CST, stability and persistence, and the injection's Jaccard.

Run:  python verify_speedup.py --trial 1
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import baseline as bl
import injection
from config import load_config
from scoring import match_injection

INJ = load_config()["injection"]


def run(field, pixel, cache):
    bl.USE_TREE_CACHE = cache
    t0 = time.perf_counter()
    groups = bl.detect(field, pixel)
    return groups, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", type=int, default=1)
    args = ap.parse_args()
    row = pd.read_csv(Path("results/pilot/trials.csv")).set_index("injection_id").loc[args.trial]
    p = {"injection_id": args.trial, "field_pixel": int(row.field_pixel), "dist_pc": row.dist_pc,
         "l_deg": row.l_deg, "b_deg": row.b_deg, "n_members": int(row.n_members),
         "r_half_pc": row.r_half_pc, "sigma_v_kms": row.sigma_v_kms}
    field = pd.read_parquet(Path("data/pilot_fields") / f"field_{p['field_pixel']}.parquet")
    field, truth = injection.inject(field, p, np.random.default_rng([INJ["pilot_seed"], args.trial]))
    before, t_before = run(field, p["field_pixel"], cache=False)
    after, t_after = run(field, p["field_pixel"], cache=True)
    key = lambda g: (g["min_cluster_size"], tuple(sorted(g["source_ids"])))
    a, b = sorted(before, key=key), sorted(after, key=key)
    same_sets = [key(x) for x in a] == [key(y) for y in b]
    same_vals = same_sets and all(
        x["cst"] == y["cst"] and x["stability"] == y["stability"] and x["persistence"] == y["persistence"]
        for x, y in zip(a, b))
    ja = match_injection(truth["member_source_ids"], [{"members": g["source_ids"], "cst": g["cst"]} for g in before])
    jb = match_injection(truth["member_source_ids"], [{"members": g["source_ids"], "cst": g["cst"]} for g in after])
    print(f"trial {args.trial}, field {p['field_pixel']}: {len(before)} vs {len(after)} groups | "
          f"identical members: {same_sets} | identical CST/stability/persistence: {same_vals} | "
          f"Jaccard {ja[1]:.4f} vs {jb[1]:.4f} | time {t_before:.0f} s without cache, {t_after:.0f} s with")
    if not (same_sets and same_vals and ja == jb):
        raise SystemExit("RESULTS DIFFER")


if __name__ == "__main__":
    main()
