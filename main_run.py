"""Main injection-recovery run (PREREG section 4, as amended in section 11).

Cells are distance bins x n_* bins. Targets: `trials_per_cell_trainval` where
the n_* bin is below `rich_n_star`, `trials_per_cell_trainval_rich` above, and
`below5_per_distance_bin` trials with N < 5 per distance bin (reported
separately). Top-up: parameters are drawn from the PREREG ranges; a draw is
accepted only if the cell it is expected to land in (n_* ~ N, since twin
stars rarely lose members) still has room and has had fewer than
`max_attempts_per_cell` attempts. Each trial is credited to the cell of its
actual n_*. Centres are eligible level-2 pixels of the given splits.

Each trial writes one row to trials.csv, its matched group's source_ids to
groups.jsonl and its vetter feature row to vetter_rows.csv (for vetter
training later). Resume-safe: a restart rebuilds the counts from trials.csv
and continues the seeded proposal sequence.

Run:  python main_run.py --splits train val --workers 3
"""
import argparse
import json
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd

import injection
import pilot
from config import load_config

CFG = load_config()
INJ = CFG["injection"]


def cells():
    """{cell: target}; a cell is (distance bin index, n_* bin index), with
    n_* bin index -1 for the separately reported N < 5 trials."""
    d_bins, n_bins = INJ["distance_bins_pc"], INJ["n_star_bins"]
    out = {}
    for i in range(len(d_bins) - 1):
        out[(i, -1)] = INJ["below5_per_distance_bin"]
        for j in range(len(n_bins) - 1):
            rich = n_bins[j] >= INJ["rich_n_star"]
            out[(i, j)] = INJ["trials_per_cell_trainval_rich" if rich else "trials_per_cell_trainval"]
    return out


def cell_of(dist_pc, n_star):
    d_bins, n_bins = INJ["distance_bins_pc"], INJ["n_star_bins"]
    i = min(int(np.searchsorted(d_bins, dist_pc, side="right")) - 1, len(d_bins) - 2)
    if n_star < n_bins[0]:
        return (i, -1)
    j = min(int(np.searchsorted(n_bins, n_star, side="right")) - 1, len(n_bins) - 2)
    return (i, j)


def proposals(splits, seed, blocks=None):
    """Endless seeded stream of (proposal index, params); `blocks` optionally
    restricts centres to some blocks of those splits."""
    rng = np.random.default_rng([seed, 0])
    k = 0
    s = json.loads(injection.SPLITS_PATH.read_text())
    use = {sp: [b for b in s["blocks"][sp] if blocks is None or b in blocks] for sp in splits}
    use = {sp: bl for sp, bl in use.items() if bl}
    names = list(use)
    centres = np.array([sum(len(s["eligible_centres"][str(b)]) for b in use[sp]) for sp in names], dtype=float)
    while True:
        split = names[int(rng.choice(len(names), p=centres / centres.sum()))]  # in proportion to centres
        p = injection.sample_params(1, rng, split=split, blocks=use[split])[0]
        p["injection_id"] = k
        yield k, p
        k += 1


def enough_memory():
    """Memory guard: a new trial starts only with memory_guard_gb free. A compact
    rich injection can push one search to about 8 GB (pilot trial 12, in the
    CST); the guard stops trials piling on, but cannot foresee such a spike."""
    import psutil

    return psutil.virtual_memory().available / 1e9 >= INJ["memory_guard_gb"]


def _init(fields_dir):
    pilot._init_worker(fields_dir, None)


def _run(params, seed):
    pixel = params["field_pixel"]
    if pixel not in pilot._FIELDS:
        pilot._FIELDS[pixel] = pd.read_parquet(pilot._FIELDS_DIR / f"field_{pixel}.parquet")
    return pilot.run_trial_full(pilot._FIELDS[pixel], params, seed)


def shard_targets(targets, shard, n_shards):
    """This shard's share of each cell target (remainders go to the lowest shards)."""
    return {c: t // n_shards + (1 if shard < t % n_shards else 0) for c, t in targets.items()}


def run_info(shard, n_shards):
    import platform
    import sys
    from importlib.metadata import version

    import psutil

    return {"shard": shard, "n_shards": n_shards, "python": sys.version.split()[0],
            "platform": platform.platform(), "cpu": platform.processor(),
            "physical_cores": psutil.cpu_count(logical=False), "logical_cores": psutil.cpu_count(),
            "libraries": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "hdbscan", "astropy",
                                                  "astropy-healpix", "scipy", "lightgbm"]}}


def run(splits, out, fields_dir, workers, seed=INJ["main_seed"], max_trials=None, shard=0, n_shards=1,
        blocks=None, trial_fn=None, executor=None):
    """trial_fn/executor are for tests (default: _run on a process pool)."""
    out.mkdir(parents=True, exist_ok=True)
    trials_path, groups_path, vetter_path = out / "trials.csv", out / "groups.jsonl", out / "vetter_rows.csv"
    launched_path = out / "launched.jsonl"
    info_path = out / "run_info.json"
    if not info_path.exists():
        info_path.write_text(json.dumps(run_info(shard, n_shards), indent=2) + "\n")
    targets = shard_targets(cells(), shard, n_shards)
    done = pd.read_csv(trials_path) if trials_path.exists() else pd.DataFrame()
    filled = {c: 0 for c in targets}
    attempts = {c: 0 for c in targets}
    for _, r in done.iterrows():
        filled[cell_of(r.dist_pc, r.n_surviving)] += 1
        attempts[tuple(json.loads(r.expected_cell))] += 1
    last = int(done.injection_id.max()) if len(done) else -1
    in_flight = {}  # future -> expected cell
    stream = proposals(splits, seed, blocks)
    launched = 0

    def has_room(c):
        pending = sum(1 for cell, _ in in_flight.values() if cell == c)
        return filled[c] + pending < targets[c] and attempts[c] < INJ["max_attempts_per_cell"]

    def next_proposal():
        for k, p in stream:
            if k <= last or k % n_shards != shard:
                continue
            c = cell_of(p["dist_pc"], p["n_members"])
            if has_room(c):
                return k, p, c
            if not any(has_room(x) for x in targets):
                return None

    trial_fn = trial_fn or _run
    pool_cm = executor or ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(str(fields_dir),))
    with pool_cm as pool:
        while True:
            while len(in_flight) < workers and (max_trials is None or launched < max_trials):
                if in_flight and not enough_memory():
                    break  # memory guard: wait for a running trial to finish first
                nxt = next_proposal()
                if nxt is None:
                    break
                k, p, c = nxt
                attempts[c] += 1
                with open(launched_path, "a") as f:
                    f.write(json.dumps({"injection_id": k}) + "\n")
                fut = pool.submit(trial_fn, p, [seed, k])
                in_flight[fut] = (c, p["split"])
                launched += 1
            if not in_flight:
                break
            finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in finished:
                c, split = in_flight.pop(fut)
                res = fut.result()
                row = {**res["row"], "split": split, "expected_cell": json.dumps(list(c))}
                filled[cell_of(row["dist_pc"], row["n_surviving"])] += 1
                pd.DataFrame([row]).to_csv(trials_path, mode="a", header=not trials_path.exists(), index=False)
                with open(groups_path, "a") as f:
                    f.write(json.dumps({"injection_id": row["injection_id"],
                                        "source_ids": None if res["group_ids"] is None
                                        else list(map(int, res["group_ids"]))}) + "\n")
                if res["vetter"] is not None:
                    v = {**res["vetter"], "injection_id": row["injection_id"], "label": 1,
                         "source": "injected", "block": row["field_pixel"] // 16, "split": split}
                    pd.DataFrame([v]).to_csv(vetter_path, mode="a", header=not vetter_path.exists(), index=False)
                print(f"trial {row['injection_id']} [{split}] cell {list(cell_of(row['dist_pc'], row['n_surviving']))}: "
                      f"d {row['dist_pc']:.0f} pc, N {row['n_members']}, n_* {row['n_surviving']}, "
                      f"J {row['matched_jaccard']:.2f}, {row['runtime_s']:.0f} s, {row['peak_memory_gb']:.1f} GB",
                      flush=True)
    return filled, attempts, targets


def report(filled, attempts, targets):
    short = {c: (filled[c], targets[c], attempts[c]) for c in targets if filled[c] < targets[c]}
    print(f"cells filled: {sum(filled[c] >= targets[c] for c in targets)} of {len(targets)}")
    for c, (f, t, a) in sorted(short.items()):
        print(f"  cell {list(c)}: {f}/{t} after {a} attempts"
              + (" (capped)" if a >= INJ["max_attempts_per_cell"] else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--out", type=Path, default=Path("results/main"))
    ap.add_argument("--fields", type=Path, default=Path("data/fields"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-trials", type=int, help="stop after launching this many (for dry runs)")
    ap.add_argument("--shard", default="0/1", help="i/N: this machine takes trials with index k %% N == i")
    ap.add_argument("--blocks", type=int, nargs="+", help="restrict centres to these blocks (e.g. 6 for tests)")
    args = ap.parse_args()
    shard, n_shards = map(int, args.shard.split("/"))
    if not 0 <= shard < n_shards:
        raise SystemExit("--shard must be i/N with 0 <= i < N")
    out = args.out if n_shards == 1 else args.out / f"shard_{shard}_of_{n_shards}"
    report(*run(args.splits, out, args.fields, args.workers, max_trials=args.max_trials,
                shard=shard, n_shards=n_shards, blocks=args.blocks))


if __name__ == "__main__":
    main()
