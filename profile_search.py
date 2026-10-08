"""Profile one detector search stage by stage (brief task P).

Times feature building, HDBSCAN per min_cluster_size, cluster stats, the CST
per min_cluster_size and the edge/merge step, with peak memory after each.

Run:  python profile_search.py                      field 102, no injection
      python profile_search.py --trial 12           replay a pilot trial (same seed)
"""
import argparse
import time
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd

import baseline as bl
import injection
from config import load_config
from features import to_features
from ocelot_cst import cluster_significance_test
from pilot import peak_memory_gb

CFG = load_config()


def profile(field, pixel):
    t = {}
    t0 = time.perf_counter()
    l0, b0 = bl.pixel_centre(pixel)
    X = to_features(bl.recentre(field, l0, b0))
    t["features"] = (time.perf_counter() - t0, peak_memory_gb())
    groups = 0
    for mcs in CFG["search"]["min_cluster_size"]:
        t0 = time.perf_counter()
        c = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=CFG["search"]["min_samples"],
                            cluster_selection_method=CFG["search"]["cluster_selection_method"]).fit(X)
        t[f"hdbscan_mcs{mcs}"] = (time.perf_counter() - t0, peak_memory_gb())
        if c.labels_.max() < 0:
            continue
        t0 = time.perf_counter()
        bl.cluster_stats(c)
        t[f"stats_mcs{mcs}"] = (time.perf_counter() - t0, peak_memory_gb())
        t0 = time.perf_counter()
        cluster_significance_test(X, c.labels_, min_samples=CFG["search"]["min_samples"],
                                  knn_kwargs=CFG["search"]["cst_knn"])
        t[f"cst_mcs{mcs}"] = (time.perf_counter() - t0, peak_memory_gb())
        groups += c.labels_.max() + 1
    return t, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", type=Path, default=Path("data/pilot_fields"))
    ap.add_argument("--pixel", type=int, default=102)
    ap.add_argument("--trial", type=int, help="replay this pilot trial's injection")
    args = ap.parse_args()

    inj = CFG["injection"]
    if args.trial is not None:
        # exact parameters as run in the pilot (re-drawing would use today's config ranges)
        row = pd.read_csv(Path("results/pilot/trials.csv")).set_index("injection_id").loc[args.trial]
        p = {"injection_id": args.trial, "field_pixel": int(row.field_pixel), "dist_pc": row.dist_pc,
             "l_deg": row.l_deg, "b_deg": row.b_deg, "n_members": int(row.n_members),
             "r_half_pc": row.r_half_pc, "sigma_v_kms": row.sigma_v_kms}
        field = pd.read_parquet(args.fields / f"field_{p['field_pixel']}.parquet")
        field, truth = injection.inject(field, p, np.random.default_rng([inj["pilot_seed"], args.trial]))
        pixel = p["field_pixel"]
        print(f"trial {args.trial}: field {pixel}, N {p['n_members']}, r_half {p['r_half_pc']:.2f} pc, "
              f"d {p['dist_pc']:.0f} pc, survivors {truth['n_surviving']}")
    else:
        pixel = args.pixel
        field = pd.read_parquet(args.fields / f"field_{pixel}.parquet")
    field = bl.select_field(field, pixel).reset_index(drop=True)
    print(f"field {pixel}: {len(field):,} stars")
    timings, groups = profile(field, pixel)
    total = sum(s for s, _ in timings.values())
    for stage, (s, mem) in timings.items():
        print(f"{stage:16s} {s:8.1f} s  {100 * s / total:5.1f}%  peak {mem:.2f} GB")
    print(f"{'total':16s} {total:8.1f} s  ({groups} groups before edge/merge)")


if __name__ == "__main__":
    main()
