"""Shared matplotlib style for all figures (IEEE double-column friendly)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#1f4e79", "#c0504d", "#4f9d69", "#e08a1e", "#7f5fa0", "#5a5a5a", "#2aa1b3", "#8c564b"]


def setup():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 120,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.3, "grid.linewidth": 0.5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
    })
    return plt


def new_fig(w=3.5, h=2.6):
    setup()
    return plt.subplots(figsize=(w, h))
