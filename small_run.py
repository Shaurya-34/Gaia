"""Small run to test the optimisations before the main run (brief task S).
Rows are not used in any result.

12 injected clusters in the pilot block's fields, 4 of them with N from 3 to 6
so that groups with 0 to 4 survivors go through the detector and scorer. Each
cluster is searched alone; the grouped (several per field) part is separate.

Run:  python small_run.py --workers 3
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import injection
import pilot
from config import load_config

INJ = load_config()["injection"]
N_VALUES = [3, 4, 5, 6, 12, 20, 40, 80, 150, 300, 600, 1200]
DISTANCES = [60, 120, 200, 280, 360, 440, 480, 90, 170, 250, 330, 410]
SEED = 20261009  # Code choice for this run
OUT = Path("results/small_run")


def small_params():
    base = injection.sample_params(len(N_VALUES), np.random.default_rng(SEED), split="train",
                                   blocks=[INJ["pilot_block"]])
    for i, p in enumerate(base):
        p.update(injection_id=i, n_members=N_VALUES[i], dist_pc=float(DISTANCES[i]))
    return base


def _run(fields_dir, params, seed):
    pilot._init_worker(fields_dir, None)
    return pilot.run_trial_full(pd.read_parquet(Path(fields_dir) / f"field_{params['field_pixel']}.parquet"),
                                params, seed)["row"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--fields", type=Path, default=Path("data/pilot_fields"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "alone.csv"
    done = set(pd.read_csv(path).injection_id) if path.exists() else set()
    todo = [p for p in small_params() if p["injection_id"] not in done]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run, str(args.fields), p, [SEED, p["injection_id"]]) for p in todo]
        for fut in as_completed(futures):
            row = fut.result()
            pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
            print(f"trial {row['injection_id']}: N {row['n_members']}, n_* {row['n_surviving']}, "
                  f"low_n {row['low_n_star']}, J {row['matched_jaccard']:.2f}, "
                  f"det3 {row['detected_tau3']}, {row['runtime_s']:.0f} s, {row['peak_memory_gb']:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
