"""E6: vetter vs CST experiment on block 6 (closure scope). Shaurya owns the
experiment and its analysis; this script builds the dataset and runs the
pre-registered comparison.

Positives: main-run matched groups that recover their injection (Jaccard >=
0.5); trials the PREREG section 5 overlap rule discards are excluded.
Negatives: every group found in shuffled data (shuffle_run.py). The unmatched
groups of injection trials are NOT used (they can contain real clusters).
All features are recomputed with the current vetter_features.py; for positives
the injection is regenerated from its saved seed, so no search is rerun.
Split by trial (60/20/20); the test side is scored once (lock file).

Run:  python run_vetter.py --val-only     validation report, test untouched
      python run_vetter.py                 the one test scoring
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import fit_detectability
import injection
import vetter_train as vt
from vetter_features import cluster_features

MAIN, SHUFFLE, OUT = Path("results/main"), Path("results/shuffle"), Path("results/vetter")
FIELDS = Path("data/pilot_fields")
PARAM_COLS = ["injection_id", "field_pixel", "dist_pc", "l_deg", "b_deg", "n_members", "r_half_pc", "sigma_v_kms"]


def positives():
    trials = pd.read_csv(MAIN / "trials.csv")
    trials["overlap"] = fit_detectability.overlap_discards(trials)
    trials = trials[(trials.matched_jaccard >= 0.5) & trials.overlap.isna()]
    det = pd.read_csv(MAIN / "vetter_rows.csv").set_index("injection_id")  # detector outputs per matched group
    ids = {json.loads(l)["injection_id"]: json.loads(l)["source_ids"]
           for l in (MAIN / "groups.jsonl").read_text().splitlines() if l.strip()}
    fields, rows = {}, []
    for _, t in trials.iterrows():
        k = int(t.injection_id)
        p = {c: t[c] for c in PARAM_COLS}
        p["field_pixel"], p["n_members"], p["injection_id"] = int(p["field_pixel"]), int(p["n_members"]), k
        if p["field_pixel"] not in fields:
            fields[p["field_pixel"]] = pd.read_parquet(FIELDS / f"field_{p['field_pixel']}.parquet")
        field_with, _ = injection.inject(fields[p["field_pixel"]], p, np.random.default_rng(json.loads(t.seed)))
        group = field_with[field_with.source_id.isin(ids[k])]
        d = det.loc[k]
        f = cluster_features(group, field_with, {"cst": d.cst, "stability": d.hdbscan_stability,
                                                 "persistence": d.hdbscan_persistence,
                                                 "membership_probs": [d.mean_membership_probability]})
        rows.append({**f, "label": 1, "source": "injected", "trial_key": f"inj_{k}"})
    return pd.DataFrame(rows)


def negatives():
    s = pd.read_csv(SHUFFLE / "shuffle_groups.csv")
    return s.assign(label=0, source="shuffle", trial_key="shuffle_" + s.field_pixel.astype(str) + "_" + s.seed.astype(str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.concat([positives(), negatives()], ignore_index=True)[vt.FEATURES + ["label", "source", "trial_key"]]
    data = vt.split_by_trial(data)
    data.to_csv(OUT / "dataset.csv", index=False)
    tr, va, te = (data[data.split == s] for s in ("train", "val", "test"))
    fitted = vt.train(tr, va)
    rng = np.random.default_rng(vt.CFG["vetter"]["seed"])
    val = vt.compare_to_baseline(va.label, vt.vetter_scores(fitted, va), vt.baseline_scores(va), rng)
    imp = dict(zip(vt.FEATURES, fitted["model"].booster_.feature_importance("gain").round(1).tolist()))
    summary = {"counts": {s: {"positives": int((d.label == 1).sum()), "negatives": int((d.label == 0).sum())}
                          for s, d in (("train", tr), ("val", va), ("test", te))},
               "features": vt.FEATURES, "feature_importance_gain": imp, "validation": val,
               "threshold_from_validation": fitted["threshold"],
               "caveat": "Positives are injected clusters (twin-star photometry) and negatives are shuffled real "
                         "data, so the vetter learns injected-vs-shuffled, not real clusters vs real noise."}
    if not args.val_only:
        summary["test"] = vt.score_test(fitted, te, rng, lock_path=OUT / "test_scored.lock")
        summary["success_rule"] = "beat CST by >= 5 precision points at 80% recall, bootstrap CI excluding 0"
        summary["result"] = "passes" if summary["test"]["passes"] else "null result (reported as such)"
        plot_pr(te, vt.vetter_scores(fitted, te))
    (OUT / ("summary_val.json" if args.val_only else "summary.json")).write_text(json.dumps(summary, indent=2,
                                                                                            default=float) + "\n")
    print(json.dumps(summary, indent=2, default=float))


def plot_pr(te, vetter):
    import sys
    sys.path.insert(0, str(Path(__file__).parent / "figures"))
    from figstyle import BLUE, MUTED, VIOLET, plt
    from sklearn.metrics import precision_recall_curve

    fig, ax = plt.subplots(figsize=(5, 4))
    for score, label, color in ((vetter, "LightGBM vetter", BLUE), (vt.baseline_scores(te), "CST alone", VIOLET)):
        p, r, _ = precision_recall_curve(te.label, score)
        ax.plot(r, p, color=color, label=label)
    ax.axvline(vt.CFG["vetter"]["recall"], color=MUTED, ls="--", lw=1)
    ax.set_xlabel("recall (injected groups)")
    ax.set_ylabel("precision")
    ax.set_title("Vetter vs CST on held-out trials (block 6)")
    ax.legend(loc="lower left")
    fig.savefig(OUT / "pr_curve.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
