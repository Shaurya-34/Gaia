"""Figure 5: detection probability against surviving members n_*, one curve
per distance bin (fitted model with bootstrap band, binned detection rates as
points, trial counts as ticks), and a distance x n_* heatmap with trial counts."""
import json

import numpy as np
import pandas as pd

from figstyle import BLUE_RAMP, FIELD, INK, INK2, MUTED, ROOT, plt, save

DIST_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]  # ordinal, validated (single hue)
FIT = ROOT / "results" / "detectability"


def model_p(theta, d, perr, rho, logn):
    a0, a1, a2, a3, log_sigma = theta
    a = a0 + a1 * d / 100 + a2 * np.log10(perr) + a3 * np.log10(rho)
    return 1 / (1 + np.exp(-np.clip((logn - a) / np.exp(log_sigma), -50, 50)))


def main():
    fit = json.loads((FIT / "fit.json").read_text())
    boot = np.load(FIT / "bootstrap_params.npy")
    est = fit["params"]
    theta = np.array([est["a0"], est["a1"], est["a2"], est["a3"], np.log(est["sigma"])])
    trials = pd.read_csv(FIT / "trials_used.csv")
    tau = fit["tau"]
    trials["y"] = trials[f"detected_tau{tau}"].astype(int)
    d_bins = [b["bin_pc"] for b in fit["n50_by_distance"]]
    n_bins = [5, 10, 25, 60, 150, 2000]
    logn = np.linspace(np.log10(5), np.log10(2000), 200)

    fig, ax = plt.subplots(figsize=(7, 4.6))
    rows = []
    for k, (lo, hi) in enumerate(d_bins):
        sub = trials[(trials.dist_pc >= lo) & (trials.dist_pc < hi)]
        color = DIST_RAMP[k]
        if len(sub):
            perr, rho = sub.median_parallax_error.median(), sub.rho_data.median()
            d = (lo + hi) / 2
            band = np.array([model_p(b, d, perr, rho, logn) for b in boot])
            ax.fill_between(10 ** logn, *np.percentile(band, [2.5, 97.5], axis=0), color=color, alpha=0.18, lw=0)
            ax.plot(10 ** logn, model_p(theta, d, perr, rho, logn), color=color, lw=2, label=f"{lo}-{hi} pc")
            # binned detection rates (points) and trial counts (ticks at the bottom)
            g = sub.groupby(pd.cut(sub.n_surviving, n_bins, right=False), observed=True)
            centres = np.array([np.sqrt(iv.left * iv.right) for iv in g.groups])
            ax.scatter(centres * (1 + 0.04 * (k - 2)), g.y.mean(), s=28, color=color, edgecolor="white", lw=0.8, zorder=4)
            ax.plot(sub.n_surviving, np.full(len(sub), -0.04 - 0.012 * k), "|", color=color, ms=6, alpha=0.8)
            rows.append(pd.DataFrame({"dist_bin": f"{lo}-{hi}", "n_star_bin": [str(i) for i in g.groups],
                                      "trials": g.size().to_numpy(), "detected_rate": g.y.mean().to_numpy()}))
    ax.set_xscale("log")
    ax.set_xlim(4.5, 2200)
    ax.set_ylim(-0.11, 1.04)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("surviving injected members n* (log scale)")
    ax.set_ylabel(f"P(detected), CST > {tau}")
    ax.set_title("Detection probability in block 6: fitted model, 95% bootstrap band, binned rates")
    ax.legend(title="distance", fontsize=8, title_fontsize=8, loc="center right")
    held = fit["held_out"]["full_model"]
    ax.text(0.99, 0.22, f"held-out trials: n = {held['n']}, Brier {held['brier']:.3f}, AUC "
            f"{held['auc']:.2f}, calibration slope {held['slope']:.2f}", transform=ax.transAxes, fontsize=7,
            color=INK2, ha="right", va="bottom")
    fig.tight_layout()
    save(fig, "fig5_detectability",
         f"Probability that the baseline detects an injected group (CST > {tau}) against its surviving member count, "
         "for five distance bins in block 6. Lines: the PREREG section 6 logistic model at each bin's median "
         "parallax error and density, with 95% bootstrap bands; points: detection rate per n* bin; ticks: "
         "individual trials. Limitations: one region; held-out trials, not unseen sky; twin-star survival near 1; "
         "trials with n* < 5 are excluded from the fit and reported separately.",
         pd.concat(rows, ignore_index=True))

    # heatmap: distance x n_* detection rate, trial counts printed, cells with < 3 trials greyed
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    rate = np.full((len(d_bins), len(n_bins) - 1), np.nan)
    count = np.zeros_like(rate, dtype=int)
    for i, (lo, hi) in enumerate(d_bins):
        for j in range(len(n_bins) - 1):
            s = trials[(trials.dist_pc >= lo) & (trials.dist_pc < hi) & (trials.n_surviving >= n_bins[j])
                       & (trials.n_surviving < n_bins[j + 1])]
            count[i, j] = len(s)
            if len(s) >= 3:
                rate[i, j] = s.y.mean()
    ax.imshow(np.where(count < 3, 1, np.nan), cmap=plt.matplotlib.colors.ListedColormap([FIELD]), aspect="auto",
              origin="lower", alpha=0.35)
    im = ax.imshow(rate, cmap=BLUE_RAMP, vmin=0, vmax=1, aspect="auto", origin="lower")
    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            txt = f"{rate[i, j]:.2f}\n(n={count[i, j]})" if count[i, j] >= 3 else f"n={count[i, j]}"
            dark = not np.isnan(rate[i, j]) and rate[i, j] > 0.6
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="white" if dark else INK)
    ax.set_xticks(range(len(n_bins) - 1), [f"[{a}, {b})" for a, b in zip(n_bins[:-1], n_bins[1:])], fontsize=7)
    ax.set_yticks(range(len(d_bins)), [f"{lo}-{hi}" for lo, hi in d_bins], fontsize=7)
    ax.set_xlabel("surviving members n*")
    ax.set_ylabel("distance (pc)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(f"detected fraction (CST > {tau})", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("Detected fraction by distance and n* (grey: fewer than 3 trials)", fontsize=9)
    fig.tight_layout()
    cells = pd.DataFrame([{"dist_bin": f"{d_bins[i][0]}-{d_bins[i][1]}", "n_star_bin": f"[{n_bins[j]}, {n_bins[j+1]})",
                           "trials": int(count[i, j]), "detected_rate": rate[i, j]}
                          for i in range(rate.shape[0]) for j in range(rate.shape[1])])
    save(fig, "fig5_heatmap",
         f"Detected fraction (CST > {tau}) per distance and surviving-member cell in block 6, with the number of "
         "trials in each cell; cells with fewer than 3 trials are greyed and not coloured. Limitation: halved "
         "allocation (about 10 trials per cell below n* = 25 and 3 above), so rich cells carry wide uncertainty.",
         cells)


if __name__ == "__main__":
    main()
