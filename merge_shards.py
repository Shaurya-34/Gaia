"""Merge main-run shards (main_run.py --shard i/N) into one result set.

Checks: all N shards present; no trial id duplicated; every id belongs to its
shard (id % N == i); every launched trial has a result (otherwise it is listed
as missing, e.g. in flight when a machine stopped); all shards used the same
library versions. Writes trials.csv, groups.jsonl and vetter_rows.csv to
<out>/merged/ and prints the cell fill against the full targets.

Run:  python merge_shards.py results/main 2
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import main_run


def merge(out, n_shards):
    dirs = [out / f"shard_{i}_of_{n_shards}" for i in range(n_shards)]
    absent = [str(d) for d in dirs if not (d / "trials.csv").exists()]
    if absent:
        raise SystemExit(f"missing shard results: {absent}")
    infos = [json.loads((d / "run_info.json").read_text()) for d in dirs]
    trials, groups, vetter, problems = [], [], [], []
    for i, d in enumerate(dirs):
        t = pd.read_csv(d / "trials.csv")
        wrong = t.injection_id[t.injection_id % n_shards != i].tolist()
        if wrong:
            problems.append(f"shard {i} holds ids of other shards: {wrong}")
        launched = {json.loads(l)["injection_id"] for l in (d / "launched.jsonl").read_text().splitlines()}
        missing = sorted(launched - set(t.injection_id))
        if missing:
            problems.append(f"shard {i}: launched but no result (rerun the shard to finish them): {missing}")
        trials.append(t.assign(shard=i))
        groups += (d / "groups.jsonl").read_text().splitlines()
        if (d / "vetter_rows.csv").exists():
            vetter.append(pd.read_csv(d / "vetter_rows.csv"))
    merged = pd.concat(trials, ignore_index=True)
    dup = merged.injection_id[merged.injection_id.duplicated()].tolist()
    if dup:
        problems.append(f"duplicated trial ids: {dup}")
    libs = {json.dumps(x["libraries"], sort_keys=True) for x in infos}
    if len(libs) > 1:
        problems.append("shards used different library versions: " + "; ".join(libs))
    dest = out / "merged"
    dest.mkdir(exist_ok=True)
    merged.sort_values("injection_id").to_csv(dest / "trials.csv", index=False)
    (dest / "groups.jsonl").write_text("\n".join(groups) + "\n")
    if vetter:
        pd.concat(vetter, ignore_index=True).sort_values("injection_id").to_csv(dest / "vetter_rows.csv", index=False)
    (dest / "run_info.json").write_text(json.dumps(infos, indent=2) + "\n")
    return merged, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("n_shards", type=int)
    args = ap.parse_args()
    merged, problems = merge(args.out, args.n_shards)
    targets = main_run.cells()
    filled = {c: 0 for c in targets}
    for _, r in merged.iterrows():
        filled[main_run.cell_of(r.dist_pc, r.n_surviving)] += 1
    print(f"{len(merged)} trials from {args.n_shards} shards; "
          f"cells filled {sum(filled[c] >= targets[c] for c in targets)} of {len(targets)}")
    for p in problems:
        print("PROBLEM:", p)
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
