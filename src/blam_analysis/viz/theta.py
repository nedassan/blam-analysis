import matplotlib.pyplot as plt
import pandas as pd

from blam_analysis.viz.common import (
    COLOR_ABOVE,
    COLOR_BASELINE,
    COLOR_BELOW,
    COLOR_ERRORBAR,
    COLOR_GRID,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_PRIMARY,
    SURFACE,
    named_row,
    resolve_display,
)


def compute_theta_rows(theta_df: pd.DataFrame, display_names: dict = None) -> list:
    display_names = display_names or {}
    return [
        named_row(r.model, r.theta_mean, r.theta_std, display_names)
        for r in theta_df.sort_values("theta_mean", ascending=False).itertuples(index=False)
    ]


def _style_axes(ax, n):
    ax.axvline(0, color=COLOR_BASELINE, linewidth=1, zorder=0)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("theta (latent ability, higher = better)", color=COLOR_TEXT_MUTED, fontsize=9)
    ax.tick_params(axis="x", colors=COLOR_TEXT_MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=COLOR_GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(COLOR_BASELINE)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_ABOVE, label="Above average"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_BELOW, label="Below average"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=8, labelcolor=COLOR_TEXT_MUTED)


def plot_theta(df: pd.DataFrame, output: str, display_names: dict = None) -> None:
    display_names = display_names or {}
    df = df.sort_values("theta_mean", ascending=True).reset_index(drop=True)
    colors = [COLOR_ABOVE if v >= 0 else COLOR_BELOW for v in df["theta_mean"]]
    labels = [resolve_display(n, display_names) for n in df["model"]]

    fig_height = max(4, 0.22 * len(df))
    fig, ax = plt.subplots(figsize=(9, fig_height), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    y = range(len(df))
    ax.barh(
        y, df["theta_mean"], height=0.6, color=colors,
        xerr=df["theta_std"], ecolor=COLOR_ERRORBAR, capsize=2, error_kw={"elinewidth": 1},
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7, color=COLOR_TEXT_PRIMARY)
    _style_axes(ax, len(df))

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"Wrote {output}")
