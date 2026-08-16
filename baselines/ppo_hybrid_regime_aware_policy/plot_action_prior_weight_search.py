from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "ppo_hybrid_regime_aware_policy_mpl"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.common.asset_utils import RESULTS_FIGURES_DIR, RESULTS_TABLES_DIR, asset_slug, normalize_asset_symbol


PROPOSED_METHOD_ID = "ppo_hybrid_regime_aware_policy"
ASSET_DISPLAY_NAMES = {
    "SPY": "SPY (S&P 500 ETF)",
    "QQQ": "QQQ (Nasdaq-100 ETF)",
    "DIA": "DIA (Dow Jones ETF)",
    "IWM": "IWM (Russell 2000 ETF)",
}


def get_validation_summary_path(asset_symbol: str) -> Path:
    normalized_asset = normalize_asset_symbol(asset_symbol)
    return RESULTS_TABLES_DIR / asset_slug(normalized_asset) / "raw" / PROPOSED_METHOD_ID / "validation_candidate_summary.json"


def get_output_figure_path(asset_symbol: str) -> Path:
    normalized_asset = normalize_asset_symbol(asset_symbol)
    return (
        RESULTS_FIGURES_DIR
        / asset_slug(normalized_asset)
        / PROPOSED_METHOD_ID
        / "validation_calmar_vs_action_prior_weight.png"
    )


def load_validation_ranking(summary_path: Path) -> tuple[list[dict], float | None]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    ranking = list(payload.get("ranking", []))
    if not ranking:
        raise ValueError(f"No ranking entries found in {summary_path}.")
    ranking.sort(key=lambda row: float(row["action_prior_weight"]))
    selected = payload.get("selected_action_prior_weight")
    selected_weight = None if selected is None else float(selected)
    return ranking, selected_weight


def plot_validation_calmar_vs_rho(
    ranking: list[dict],
    selected_weight: float | None,
    asset_symbol: str,
    output_path: Path,
) -> Path:
    asset_label = ASSET_DISPLAY_NAMES[normalize_asset_symbol(asset_symbol)]
    rhos = [float(row["action_prior_weight"]) for row in ranking]
    calmars = [float(row["validation_calmar"]) for row in ranking]
    filter_flags = [bool(row.get("behavior_filter_passed", False)) for row in ranking]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.6, 5.8), dpi=200)

    ax.plot(
        rhos,
        calmars,
        color="#1f4e79",
        linewidth=2.4,
        marker="o",
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.8,
        label="Validation Calmar",
        zorder=2,
    )

    passed_x = [rho for rho, passed in zip(rhos, filter_flags) if passed]
    passed_y = [calmar for calmar, passed in zip(calmars, filter_flags) if passed]
    failed_x = [rho for rho, passed in zip(rhos, filter_flags) if not passed]
    failed_y = [calmar for calmar, passed in zip(calmars, filter_flags) if not passed]

    if passed_x:
        ax.scatter(
            passed_x,
            passed_y,
            s=64,
            color="#2a9d8f",
            edgecolors="white",
            linewidths=0.8,
            label="Behavior Filter Passed",
            zorder=3,
        )
    if failed_x:
        ax.scatter(
            failed_x,
            failed_y,
            s=64,
            color="#d62828",
            edgecolors="white",
            linewidths=0.8,
            label="Behavior Filter Failed",
            zorder=3,
        )

    best_idx = max(range(len(calmars)), key=lambda idx: calmars[idx])
    best_rho = rhos[best_idx]
    best_calmar = calmars[best_idx]

    if selected_weight is not None:
        selected_idx = min(range(len(rhos)), key=lambda idx: abs(rhos[idx] - selected_weight))
        ax.axvline(selected_weight, color="#6a4c93", linestyle="--", linewidth=1.6, alpha=0.9, zorder=1)
        ax.scatter(
            [rhos[selected_idx]],
            [calmars[selected_idx]],
            s=150,
            color="#6a4c93",
            marker="*",
            edgecolors="white",
            linewidths=1.0,
            label=rf"Selected $\rho$ = {selected_weight:.1f}",
            zorder=4,
        )

    for rho, calmar in zip(rhos, calmars):
        ax.annotate(
            f"{calmar:.3f}",
            (rho, calmar),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=9,
            color="#2f2f2f",
        )

    ax.annotate(
        rf"Best validation Calmar at $\rho={best_rho:.1f}$",
        (best_rho, best_calmar),
        textcoords="offset points",
        xytext=(12, -18),
        ha="left",
        fontsize=9.5,
        color="#1f4e79",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#c7d5e0", "alpha": 0.95},
    )

    ax.set_title(rf"Validation Calmar Ratio vs. Action-Prior Weight $\rho$ ({asset_label})", pad=14, fontsize=14)
    ax.set_xlabel(r"Action-Prior Weight $\rho$")
    ax.set_ylabel("Validation Calmar Ratio")
    ax.set_xticks(rhos)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.grid(True, alpha=0.35, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the validation Calmar ratio versus action-prior weight for PPO-HRAP."
    )
    parser.add_argument(
        "--asset",
        default="SPY",
        help="Asset symbol to plot. Supported: SPY, QQQ, DIA, IWM.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional override path to validation_candidate_summary.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional override output path for the generated figure.",
    )
    args = parser.parse_args()

    asset_symbol = normalize_asset_symbol(args.asset)
    input_path = args.input or get_validation_summary_path(asset_symbol)
    output_path = args.output or get_output_figure_path(asset_symbol)

    if not input_path.exists():
        raise FileNotFoundError(f"Validation summary file not found: {input_path}")

    ranking, selected_weight = load_validation_ranking(input_path)
    figure_path = plot_validation_calmar_vs_rho(ranking, selected_weight, asset_symbol, output_path)
    print(f"Saved figure to: {figure_path}")


if __name__ == "__main__":
    main()
