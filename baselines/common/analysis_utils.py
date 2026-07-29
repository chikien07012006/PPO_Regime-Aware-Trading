from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "ppo_hybrid_regime_aware_policy_mpl"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from baselines.common.metrics import TRADING_DAYS_PER_YEAR, compute_performance_metrics


REGIME_ORDER = ["Low-vol", "Mid-vol", "High-vol", "Crisis"]
REGIME_COLORS = {
    "Low-vol": "#cfe8cf",
    "Mid-vol": "#f4e3b2",
    "High-vol": "#f5b971",
    "Crisis": "#d97373",
}
METHOD_LABEL_MAP = {
    "buy_and_hold_spy": "Buy & Hold",
    "risk_parity_spy_cash": "Risk-Parity Proxy",
    "cppi_spy_cash": "CPPI",
    "ppo_profit_only": "PPO + Profit Only",
    "sac_profit_only": "SAC + Profit Only",
    "ppo_variance_penalized": "PPO + Variance-Penalized Reward",
    "ppo_markovian_mdd_static": "PPO + Static MDD Reward",
    "ppo_hybrid_regime_aware_policy": "PPO-HRAP (Our Proposed Method)",
}


def _apply_paper_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.edgecolor": "#333333",
            "grid.color": "#d9d9d9",
            "grid.alpha": 0.5,
            "grid.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _format_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.4, linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")


def _format_date_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


def _display_method_name(method_name: str) -> str:
    return METHOD_LABEL_MAP.get(method_name, method_name)


def label_vix_regime(vix_value: float) -> str:
    if pd.isna(vix_value):
        return "Unclassified"
    if vix_value > 40:
        return "Crisis"
    if vix_value >= 25:
        return "High-vol"
    if vix_value >= 15:
        return "Mid-vol"
    return "Low-vol"


def add_vix_regimes(frame: pd.DataFrame, vix_column: str = "vix_close") -> pd.DataFrame:
    enriched = frame.copy()
    if vix_column not in enriched.columns:
        raise ValueError(f"DataFrame must contain '{vix_column}' to assign regimes.")
    enriched["regime"] = enriched[vix_column].apply(label_vix_regime)
    return enriched


def compute_regime_metrics(
    diagnostics_by_method: dict[str, pd.DataFrame],
    *,
    vix_column: str = "vix_close",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for method_name, diagnostics in diagnostics_by_method.items():
        frame = add_vix_regimes(diagnostics, vix_column=vix_column)

        for regime_name in REGIME_ORDER:
            regime_frame = frame.loc[frame["regime"] == regime_name].copy()
            if regime_frame.empty:
                continue

            regime_returns = regime_frame["daily_returns"].fillna(0.0).reset_index(drop=True)
            regime_turnover = regime_frame["turnover"].fillna(0.0).reset_index(drop=True)
            regime_equity = (1.0 + regime_returns).cumprod()
            regime_equity.iloc[0] = max(regime_equity.iloc[0], 1.0 + regime_returns.iloc[0])

            equity_series = pd.Series(regime_equity.to_numpy(), name="portfolio_value")
            returns_series = pd.Series(regime_returns.to_numpy(), name="daily_returns")
            turnover_series = pd.Series(regime_turnover.to_numpy(), name="turnover")

            metrics = compute_performance_metrics(
                portfolio_value=equity_series,
                daily_returns=returns_series,
                turnover=turnover_series,
                periods_per_year=TRADING_DAYS_PER_YEAR,
            )
            row = {"method": method_name, "regime": regime_name}
            row.update(metrics)
            rows.append(row)

    regime_metrics = pd.DataFrame(rows)
    if regime_metrics.empty:
        return pd.DataFrame()

    regime_metrics = regime_metrics.set_index(["method", "regime"]).sort_index()
    return regime_metrics


def save_regime_metrics(
    regime_metrics: pd.DataFrame,
    output_csv_path: str | Path,
    output_json_path: str | Path,
) -> tuple[Path, Path]:
    output_csv_path = Path(output_csv_path)
    output_json_path = Path(output_json_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    regime_metrics.to_csv(output_csv_path)
    json_payload = (
        regime_metrics.reset_index()
        .to_dict(orient="records")
    )
    output_json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    return output_csv_path, output_json_path


def plot_regime_equity_with_background(
    diagnostics: pd.DataFrame,
    method_name: str,
    output_path: str | Path,
    *,
    vix_column: str = "vix_close",
    asset_label: str | None = None,
) -> None:
    frame = add_vix_regimes(diagnostics, vix_column=vix_column).copy()
    frame = frame.sort_values("date").reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=300)

    start_idx = 0
    while start_idx < len(frame):
        regime_name = frame.loc[start_idx, "regime"]
        end_idx = start_idx
        while end_idx + 1 < len(frame) and frame.loc[end_idx + 1, "regime"] == regime_name:
            end_idx += 1
        ax.axvspan(
            frame.loc[start_idx, "date"],
            frame.loc[end_idx, "date"],
            color=REGIME_COLORS.get(regime_name, "#cccccc"),
            alpha=0.16,
        )
        start_idx = end_idx + 1

    ax.plot(
        frame["date"],
        frame["portfolio_value"],
        color="#1f4e79",
        linewidth=2.4,
        label=method_name,
    )
    title = f"{method_name}: Equity Curve with VIX Regime Overlay"
    if asset_label:
        title = f"{title}\n{asset_label}"
    fig.suptitle(title, y=0.97)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (USD)")
    _format_axes(ax)
    _format_date_axis(ax)
    regime_handles = [
        Patch(facecolor=REGIME_COLORS[name], edgecolor="none", alpha=0.45, label=name)
        for name in REGIME_ORDER
    ]
    line_handle = Line2D([0], [0], color="#1f4e79", lw=2.4, label=method_name)
    fig.legend(
        handles=[line_handle, *regime_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=3,
        frameon=True,
    )

    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.12)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_regime_bar_comparison(
    regime_metrics: pd.DataFrame,
    proposed_method_name: str,
    static_method_name: str,
    output_path: str | Path,
    *,
    asset_label: str | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    focus_regimes = ["High-vol", "Crisis"]
    focus_metrics = ["Sharpe Ratio", "MDD", "Calmar Ratio"]

    _apply_paper_style()
    fig, axes = plt.subplots(1, len(focus_metrics), figsize=(15.5, 5.8), dpi=300)

    for axis, metric_name in zip(axes, focus_metrics):
        subset = []
        for regime_name in focus_regimes:
            for method_name in [proposed_method_name, static_method_name]:
                if (method_name, regime_name) in regime_metrics.index:
                    subset.append(
                        {
                            "regime": regime_name,
                            "method": method_name,
                            "value": regime_metrics.loc[(method_name, regime_name), metric_name],
                        }
                    )

        subset_frame = pd.DataFrame(subset)
        if subset_frame.empty:
            axis.set_visible(False)
            continue

        x_positions = np.arange(len(focus_regimes))
        width = 0.35
        proposed_values = [
            subset_frame.loc[
                (subset_frame["regime"] == regime_name) & (subset_frame["method"] == proposed_method_name),
                "value",
            ].iloc[0]
            if not subset_frame.loc[
                (subset_frame["regime"] == regime_name) & (subset_frame["method"] == proposed_method_name)
            ].empty
            else np.nan
            for regime_name in focus_regimes
        ]
        static_values = [
            subset_frame.loc[
                (subset_frame["regime"] == regime_name) & (subset_frame["method"] == static_method_name),
                "value",
            ].iloc[0]
            if not subset_frame.loc[
                (subset_frame["regime"] == regime_name) & (subset_frame["method"] == static_method_name)
            ].empty
            else np.nan
            for regime_name in focus_regimes
        ]

        axis.bar(
            x_positions - (width / 2),
            proposed_values,
            width=width,
            label="Proposed",
            color="#1f4e79",
        )
        axis.bar(
            x_positions + (width / 2),
            static_values,
            width=width,
            label="Static MDD",
            color="#b24a3a",
        )
        axis.set_xticks(x_positions)
        axis.set_xticklabels(focus_regimes)
        axis.set_title(metric_name, fontsize=12, weight="bold")
        _format_axes(axis)
        axis.grid(True, axis="y", alpha=0.35)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True)
    suptitle = "Regime-Specific Comparison: Proposed Method vs Static MDD"
    if asset_label:
        suptitle = f"{suptitle}\n{asset_label}"
    fig.suptitle(suptitle)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_lambda_vs_vix(
    diagnostics: pd.DataFrame,
    output_path: str | Path,
    *,
    vix_column: str = "vix_close",
    lambda_column: str = "lambda_rrc",
    asset_label: str | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = diagnostics.sort_values("date").reset_index(drop=True)
    if vix_column not in frame.columns or lambda_column not in frame.columns:
        raise ValueError(f"Diagnostics must contain '{vix_column}' and '{lambda_column}'.")

    _apply_paper_style()
    fig, ax_left = plt.subplots(figsize=(13, 6.5), dpi=300)
    ax_right = ax_left.twinx()

    ax_left.plot(frame["date"], frame[vix_column], color="#c96a28", linewidth=2.2, label="VIX")
    ax_right.plot(frame["date"], frame[lambda_column], color="#1f4e79", linewidth=2.2, label=r"$\lambda_{rrc}(t)$")

    title = r"Dynamic Risk Conditioning: VIX and $\lambda_{rrc}(t)$"
    if asset_label:
        title = f"{title}\n{asset_label}"
    fig.suptitle(title, y=0.97)
    ax_left.set_xlabel("Date")
    ax_left.set_ylabel("VIX", color="#c96a28")
    ax_right.set_ylabel(r"$\lambda_{rrc}(t)$", color="#1f4e79")
    _format_axes(ax_left)
    _format_date_axis(ax_left)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.spines["right"].set_color("#444444")

    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    fig.legend(
        lines_left + lines_right,
        labels_left + labels_right,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=True,
    )

    fig.subplots_adjust(left=0.08, right=0.92, top=0.82, bottom=0.12)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_pareto_frontier(
    metrics_frame: pd.DataFrame,
    output_path: str | Path,
    *,
    method_column: str = "method",
    return_column: str = "Annualized Return",
    mdd_column: str = "MDD",
    asset_label: str | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(10.5, 7), dpi=300)

    frame = metrics_frame.copy()
    frame["MDD_abs"] = frame[mdd_column].abs()
    frame = frame.sort_values(by=[return_column, "MDD_abs"], ascending=[False, True]).reset_index(drop=True)

    colors = ["#1f77b4" if row[method_column] != "ppo_hybrid_regime_aware_policy" else "#b22222" for _, row in frame.iterrows()]
    ax.scatter(frame["MDD_abs"], frame[return_column], s=85, color=colors, alpha=0.9, edgecolors="white", linewidths=0.8)
    offset_cycle = [(8, 8), (8, -10), (-70, 8), (-70, -10), (12, 18), (12, -20)]
    for idx, (_, row) in enumerate(frame.iterrows()):
        dx, dy = offset_cycle[idx % len(offset_cycle)]
        ax.annotate(
            _display_method_name(str(row[method_column])),
            (row["MDD_abs"], row[return_column]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.75},
        )

    title = "Pareto Frontier: Annualized Return vs Maximum Drawdown"
    if asset_label:
        title = f"{title}\n{asset_label}"
    ax.set_title(title)
    ax.set_xlabel("Absolute Maximum Drawdown")
    ax.set_ylabel("Annualized Return")
    _format_axes(ax)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_all_methods_equity(
    method_results: dict[str, pd.DataFrame],
    output_path: str | Path,
    *,
    title: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(14, 7.8), dpi=300)

    deduped: list[tuple[str, pd.DataFrame, list[str]]] = []
    for method_name, diagnostics in method_results.items():
        matched = False
        for idx, (_, existing_frame, grouped_methods) in enumerate(deduped):
            if diagnostics["portfolio_value"].reset_index(drop=True).equals(
                existing_frame["portfolio_value"].reset_index(drop=True)
            ):
                grouped_methods.append(method_name)
                matched = True
                break
        if not matched:
            deduped.append((method_name, diagnostics, [method_name]))

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]

    for color, (_, diagnostics, grouped_methods) in zip(colors, deduped):
        label = " / ".join(_display_method_name(method_name) for method_name in grouped_methods)
        ax.plot(
            pd.to_datetime(diagnostics["date"]),
            diagnostics["portfolio_value"],
            linewidth=2.3,
            color=color,
            label=label,
        )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (USD)")
    _format_axes(ax)
    _format_date_axis(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=2,
        frameon=True,
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
