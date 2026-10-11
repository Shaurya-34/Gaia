"""E4: fit the PREREG section 6 detectability model to the main run.

P(detected) = sigmoid((log10 n_* - a) / sigma),
a = a0 + a1 (d / 100 pc) + a2 log10(median parallax error / mas) + a3 log10(rho_data)

Closure scope: fit on a fixed seed-based 75% of trials and evaluate on the other
25% (held-out trials, not held-out sky). Trials with n_* < 5 are reported
separately; trials the PREREG section 5 overlap rule discards are dropped.
Outputs results/detectability/fit.json (read by figures 5 and 7).

Run:  python fit_detectability.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, roc_auc_score

import catalogue
from config import load_config

CFG = load_config()
MAIN = Path("results/main")
OUT = Path("results/detectability")
TAU = 3  # primary detection threshold (PREREG section 5); 5 and 7 also reported
SPLIT_SEED = 20261011  # Code choice: fixed held-out split
N_BOOT = CFG["analysis"]["bootstrap_resamples"]
PARAMS = ["a0", "a1", "a2", "a3", "sigma"]


def design(df):
    return np.column_stack([np.ones(len(df)), df.dist_pc / 100.0,
                            np.log10(df.median_parallax_error), np.log10(df.rho_data)])


def prob(theta, X, logn):
    *a, log_sigma = theta
    z = (logn - X @ np.asarray(a)) / np.exp(log_sigma)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def nll(theta, X, logn, y, k=None):
    p = prob(theta, X, logn) if k is None else gen_prob(theta, X, logn, k)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def gen_prob(theta, X, logn, k):
    *a, log_sigma = theta
    z = (logn - X @ np.asarray(a)) / np.exp(log_sigma)
    return 1.0 - (1.0 + np.exp(np.clip(z, -50, 50))) ** (-k)


def fit(X, logn, y, start=None):
    start = np.r_[np.log10(10.0), np.zeros(X.shape[1] - 1), np.log(0.2)] if start is None else start
    res = minimize(nll, start, args=(X, logn, y), method="Nelder-Mead",
                   options={"maxiter": 20000, "maxfev": 20000, "xatol": 1e-6, "fatol": 1e-8})
    return res.x, res.fun


def calibration(y, p):
    """Slope and intercept of a logistic regression of y on logit(p)."""
    lp = np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1))

    def f(t):
        q = np.clip(1 / (1 + np.exp(-(t[0] + t[1] * lp))), 1e-9, 1 - 1e-9)
        return -np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))
    t = minimize(f, [0.0, 1.0], method="Nelder-Mead").x
    return {"intercept": float(t[0]), "slope": float(t[1])}


def metrics(y, p):
    return {"n": int(len(y)), "brier": float(brier_score_loss(y, p)),
            "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None, **calibration(y, p)}


def overlap_discards(trials):
    """PREREG section 5, applied after the run from the saved matched groups:
    discard when the group holds more stars of one Hunt & Reffert 2023 cluster
    than injected stars (injected source_ids are negative)."""
    members = catalogue.load()[1]
    groups = {}
    for line in (MAIN / "groups.jsonl").read_text().splitlines():
        if line.strip():
            g = json.loads(line)
            groups[g["injection_id"]] = g["source_ids"]
    discard = {}
    for i in trials.injection_id:
        ids = groups.get(i)
        if not ids:
            discard[i] = None
            continue
        ids = np.asarray(ids)
        known = members[members.GaiaDR3.isin(ids)].Name.value_counts()
        discard[i] = known.index[0] if len(known) and known.iloc[0] > (ids < 0).sum() else None
    return trials.injection_id.map(discard)


def main():
    t = pd.read_csv(MAIN / "trials.csv")
    t["overlap_cluster"] = overlap_discards(t)
    low = t[t.low_n_star]
    kept = t[~t.low_n_star & t.overlap_cluster.isna()].copy()
    kept["y"] = kept[f"detected_tau{TAU}"].astype(int)
    rng = np.random.default_rng(SPLIT_SEED)
    held = rng.permutation(kept.injection_id.to_numpy())[: len(kept) // 4]
    train, test = kept[~kept.injection_id.isin(held)], kept[kept.injection_id.isin(held)]

    X, logn, y = design(train), np.log10(train.n_surviving), train.y.to_numpy()
    theta, f = fit(X, logn, y)
    boot = []
    for _ in range(N_BOOT):
        i = rng.integers(len(train), size=len(train))
        boot.append(fit(X[i], logn.to_numpy()[i], y[i], start=theta)[0])
    boot = np.array(boot)
    ci = {p: [float(np.percentile(boot[:, j] if p != "sigma" else np.exp(boot[:, j]), q)) for q in (2.5, 97.5)]
          for j, p in enumerate(PARAMS)}
    est = dict(zip(PARAMS, [*theta[:-1], np.exp(theta[-1])]))

    # distance-only comparison model: a = b0 + b1 d/100 (PREREG pass criterion)
    Xd, Xd_test = design(train)[:, :2], design(test)[:, :2]
    theta_d, _ = fit(Xd, logn, y)
    p_full = prob(theta, design(test), np.log10(test.n_surviving))
    p_dist = prob(theta_d, Xd_test, np.log10(test.n_surviving))

    # generalised logistic sensitivity (2025 paper eq. 1), free k
    def nll_gen(t):
        return nll(t[:-1], X, logn, y, k=np.exp(t[-1]))
    g = minimize(nll_gen, np.r_[theta, 0.0], method="Nelder-Mead", options={"maxiter": 40000, "maxfev": 40000}).x

    # size check: indicator for r_half >= 3 pc added to a
    Xs = np.column_stack([X, (train.r_half_pc >= 3).astype(float)])
    theta_s, _ = fit(Xs, logn, y, start=np.r_[theta[:-1], 0.0, theta[-1]])

    bins = CFG["injection"]["distance_bins_pc"]
    n50 = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sub = kept[(kept.dist_pc >= lo) & (kept.dist_pc < hi)]
        row = {"bin_pc": [lo, hi], "trials": int(len(sub))}
        if len(sub):
            ref = pd.DataFrame({"dist_pc": [(lo + hi) / 2], "median_parallax_error": [sub.median_parallax_error.median()],
                                "rho_data": [sub.rho_data.median()]})
            row["model_n50"] = float(10 ** (design(ref) @ theta[:-1])[0])
        if len(sub) >= 5 and 0 < sub.y.sum() < len(sub):  # empirical: 1-D logistic in log10 n_* per bin
            t1, _ = fit(np.ones((len(sub), 1)), np.log10(sub.n_surviving), sub.y.to_numpy(),
                        start=np.r_[1.0, np.log(0.2)])
            row["empirical_n50"] = float(10 ** t1[0])
        n50.append(row)

    result = {
        "tau": TAU, "trials_total": int(len(t)), "trials_fit": int(len(train)), "trials_held_out": int(len(test)),
        "n_star_below_5": {"trials": int(len(low)), "detected": int(low[f"detected_tau{TAU}"].sum())},
        "overlap_discards": t.overlap_cluster.dropna().value_counts().to_dict(),
        "detected_rate_by_tau": {k: float(kept[f"detected_tau{k}"].mean()) for k in CFG["search"]["cst_tau"]},
        "params": est, "params_ci95": ci, "neg_log_likelihood": float(f),
        "held_out": {"full_model": metrics(test.y.to_numpy(), p_full),
                     "distance_only": metrics(test.y.to_numpy(), p_dist),
                     "full_beats_distance_only_brier": bool(brier_score_loss(test.y, p_full)
                                                             < brier_score_loss(test.y, p_dist))},
        "generalised_logistic": {"params": dict(zip(PARAMS, [*g[:-2], float(np.exp(g[-2]))])),
                                 "k": float(np.exp(g[-1])), "neg_log_likelihood": float(nll_gen(g))},
        "size_check": {"a4_rhalf_ge_3pc": float(theta_s[-2]),
                       "detected_rate_rhalf_lt_3": float(kept[kept.r_half_pc < 3].y.mean()),
                       "detected_rate_rhalf_ge_3": float(kept[kept.r_half_pc >= 3].y.mean())},
        "n50_by_distance": n50,
        "covariate_reference": {"median_parallax_error": float(kept.median_parallax_error.median()),
                                "rho_data": float(kept.rho_data.median())},
        "note": "Held-out trials, not held-out sky (closure scope). All trials from block 6's four fields.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fit.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    np.save(OUT / "bootstrap_params.npy", boot)  # a0..a3, log sigma per resample (figure 5 bands)
    kept.assign(held_out=kept.injection_id.isin(held)).to_csv(OUT / "trials_used.csv", index=False)
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
