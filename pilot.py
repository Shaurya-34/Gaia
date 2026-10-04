"""Pilot stage (PREREG section 4): 100 injection trials in one training block,
not used in any result. Measures runtime per detector run, checks that cut
survival against distance looks sane, and exercises the Jaccard scorer.

Run:  python pilot.py --baseline-only     one real field, no injection: runtime and memory
      python pilot.py                     the trials (resume-safe, one CSV row per trial)
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

import baseline
import catalogue
import injection
from config import load_config
from features import add_galactic_pm
from scoring import is_recovered, match_injection

CFG = load_config()
TAUS = CFG["search"]["cst_tau"]
OUT = Path(__file__).with_name("results") / "pilot"


def load_stars(bulk_dir):
    """All bulk parquet files, with Galactic proper motions added."""
    files = sorted(Path(bulk_dir).glob("GaiaSource_*.parquet"))
    if not files:
        raise SystemExit(f"no bulk parquet files in {bulk_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"loaded {len(df):,} stars from {len(files)} files")
    return add_galactic_pm(df)


def peak_memory_gb():
    m = psutil.Process().memory_info()
    return getattr(m, "peak_wset", m.rss) / 1e9


def run_trial(field, params, seed, known_members=None):
    """Inject one cluster into `field` (stars of its level-2 field), run the
    baseline, match and score. Returns one result row. With `known_members`
    (Hunt & Reffert 2023), the PREREG section 5 overlap rule is reported."""
    rng = np.random.default_rng(seed)
    field_with, truth = injection.inject(field, params, rng)
    t0 = time.perf_counter()
    groups = baseline.detect(field_with, params["field_pixel"])
    runtime = time.perf_counter() - t0
    members = truth["member_source_ids"]
    idx, jac = match_injection(members, [{"members": g["source_ids"], "cst": g["cst"]} for g in groups])
    match = groups[idx] if idx is not None else None
    row = {k: params[k] for k in ["injection_id", "field_pixel", "dist_pc", "l_deg", "b_deg",
                                  "n_members", "r_half_pc", "sigma_v_kms"]}
    row.update({
        "seed": str(seed), "n_field_stars": len(field), "n_surviving": truth["n_surviving"],
        "survival": truth["n_surviving"] / params["n_members"],
        "median_parallax_error": float(np.median(field_with.loc[field_with.is_injected, "parallax_error"]))
        if truth["n_surviving"] else np.nan,
        "n_groups": len(groups), "matched_jaccard": jac,
        "matched_cst": match["cst"] if match else np.nan,
        "matched_n_members": match["n_members"] if match else 0,
        "recovered": is_recovered(jac),
        "runtime_s": round(runtime, 2), "peak_memory_gb": round(peak_memory_gb(), 2),
    })
    for tau in TAUS:
        row[f"detected_tau{tau}"] = bool(match and match["cst"] > tau)
    if known_members is not None:
        discard, name, n_known = (catalogue.known_overlap(match["source_ids"], members, known_members)
                                  if match else (False, None, 0))
        row.update({"known_overlap_discard": discard, "known_overlap_cluster": name, "known_overlap_n": n_known})
    return row


def summarise(trials):
    n = len(trials)
    rt = trials.runtime_s
    main_runs = CFG["injection"]["trials_per_cell_trainval"] * 25 + CFG["injection"]["trials_per_cell_test"] * 25
    hours = main_runs * rt.mean() / 3600
    print(f"\n{n} trials | detector run (4 min_cluster_size values): median {rt.median():.0f} s, "
          f"mean {rt.mean():.0f} s, max {rt.max():.0f} s | peak memory {trials.peak_memory_gb.max():.1f} GB")
    print(f"projected main run: {main_runs} trials x {rt.mean():.0f} s = {hours:.0f} CPU-hours "
          f"(ceiling {CFG['injection']['cpu_hours_ceiling']})")
    bins = CFG["injection"]["distance_bins_pc"]
    by_d = trials.groupby(pd.cut(trials.dist_pc, bins), observed=False)
    print("\nby distance: trials, median survival, detected (tau 3), recovered (J >= 0.5)")
    print(pd.DataFrame({"trials": by_d.size(), "survival": by_d.survival.median().round(2),
                        "detected_tau3": by_d.detected_tau3.mean().round(2),
                        "recovered": by_d.recovered.mean().round(2)}).to_string())
    print(f"\nmatched Jaccard quartiles: {trials.matched_jaccard.quantile([.25, .5, .75]).round(2).tolist()}; "
          f"trials with n_* < {CFG['definitions']['min_n_star_fit']}: "
          f"{(trials.n_surviving < CFG['definitions']['min_n_star_fit']).sum()}")
    if "known_overlap_discard" in trials:
        print(f"known-cluster overlap rule (PREREG section 5) would discard {int(trials.known_overlap_discard.sum())} "
              f"trial(s): {trials.loc[trials.known_overlap_discard, 'known_overlap_cluster'].value_counts().to_dict()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bulk", type=Path, default=Path("data/bulk"))
    ap.add_argument("--baseline-only", action="store_true", help="one real field, no injection")
    ap.add_argument("--trials", type=int, default=CFG["injection"]["pilot_trials"])
    args = ap.parse_args()

    inj = CFG["injection"]
    params = injection.sample_params(args.trials, np.random.default_rng(inj["pilot_seed"]),
                                     split="train", blocks=[inj["pilot_block"]])
    stars = load_stars(args.bulk)
    fields = {}

    def field_for(pixel):
        if pixel not in fields:
            fields[pixel] = baseline.select_field(stars, pixel).reset_index(drop=True)
        return fields[pixel]

    if args.baseline_only:
        pixel = params[0]["field_pixel"]
        field = field_for(pixel)
        t0 = time.perf_counter()
        groups = baseline.detect(field, pixel)
        runtime = time.perf_counter() - t0
        csts = np.array([g["cst"] for g in groups])
        print(f"field {pixel}: {len(field):,} stars, {len(groups)} groups, "
              + ", ".join(f"{(csts > t).sum()} with CST > {t}" for t in TAUS)
              + f" | {runtime:.0f} s, peak memory {peak_memory_gb():.1f} GB")
        return

    known = catalogue.load()[1] if (catalogue.OUT / "hr23_members.parquet").exists() else None
    if known is None:
        print("no Hunt & Reffert 2023 catalogue (run catalogue.py); overlap rule not reported")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "trials.csv"
    done = set(pd.read_csv(path).injection_id) if path.exists() else set()
    for p in params:
        if p["injection_id"] in done:
            continue
        row = run_trial(field_for(p["field_pixel"]), p, [inj["pilot_seed"], p["injection_id"]], known)
        pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
        print(f"trial {row['injection_id']}: d {row['dist_pc']:.0f} pc, n_* {row['n_surviving']}, "
              f"J {row['matched_jaccard']:.2f}, CST {row['matched_cst']:.1f}, {row['runtime_s']:.0f} s")
    summarise(pd.read_csv(path))


if __name__ == "__main__":
    main()
