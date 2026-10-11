"""Shared style for the article figures (E5).

Cold palette only (blues, aqua, violet), validated with the dataviz skill's
validate_palette.js (light mode, all pairs): blue/aqua/violet pass every check;
aqua is below 3:1 on the light surface, so it is always paired with a label.
Field stars are neutral grey; magnitudes use a single-hue blue ramp.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "figures"

BLUE, AQUA, VIOLET = "#2a78d6", "#1baf7a", "#4a3aa7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
FIELD = "#b9b8b1"  # field stars: neutral, recessive
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 4,
})


def save(fig, name, caption, data=None):
    """Save results/figures/<name>.png (dpi 200), <name>_caption.txt and,
    if given, the DataFrame behind it as <name>_data.csv."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    (OUT / f"{name}_caption.txt").write_text(caption.strip() + "\n", encoding="utf-8")
    if data is not None:
        data.to_csv(OUT / f"{name}_data.csv", index=False)
    plt.close(fig)
    print(f"saved {OUT / name}.png")
