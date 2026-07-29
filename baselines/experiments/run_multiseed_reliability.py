from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.common.analysis_utils import METHOD_LABEL_MAP
from baselines.common.asset_utils import (
    DEFAULT_ASSET,
    cross_asset_figures_dir,
    cross_asset_models_dir,
    cross_asset_tables_dir,
    normalize_asset_symbol,
)
from baselines.common.metrics import compute_performance_metrics
from baselines.ppo_hybrid_regime_aware_policy.pipeline import (
    PROPOSED_METHOD_ID,
    build_single_config,
    create_train_validation_combined_csv,
    get_asset_context,
)
from baselines.rl.rl_baseline_common import (
    FIGURES_DIR,
    RESULTS_DIR,
    TABLES_DIR,
    RLBaselineConfig,
    evaluate_saved_model,
    save_single_result,
    train_model,
)
from baselines.rl.train_ppo_markovian_mdd_static import build_config as build_ppo_markovian_mdd_static_config
from baselines.rl.train_ppo_profit_only import build_config as build_ppo_profit_only_config
from baselines.rl.train_ppo_variance_penalized import build_config as build_ppo_variance_penalized_config
from baselines.rl.train_sac_profit_only import build_config as build_sac_profit_only_config

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


SEEDS = [7, 11, 19, 23, 42]
MULTISEED_METHODS = [
    "ppo_profit_only",
    "sac_profit_only",
    "ppo_variance_penalized",
    "ppo_markovian_mdd_static",
    PROPOSED_METHOD_ID,
]

COLOR_MAP = {
    "buy_and_hold_spy": "#1f77b4",
    "risk_parity_spy_cash": "#ff7f0e",
    "cppi_spy_cash": "#2ca02c",
    "ppo_profit_only": "#4c78a8",
    "sac_profit_only": "#f58518",
    "ppo_variance_penalized": "#54a24b",
    "ppo_markovian_mdd_static": "#e45756",
    PROPOSED_METHOD_ID: "#7a5195",
}
PLOT_ORDER = [
    "buy_and_hold_spy",
    "risk_parity_spy_cash",
    "cppi_spy_cash",
    "ppo_profit_only",
    "sac_profit_only",
    "ppo_variance_penalized",
    "ppo_markovian_mdd_static",
    PROPOSED_METHOD_ID,
]
METRIC_COLUMNS = [
    "Total Return",
    "Annualized Return",
    "Sharpe Ratio",
    "Sortino Ratio",
    "Calmar Ratio",
    "MDD",
    "Turnover",
]

ASSET_DISPLAY_NAMES = {
    "SPY": "SPY (S&P 500 ETF)",
    "QQQ": "QQQ (Nasdaq-100 ETF)",
    "DIA": "DIA (Dow Jones ETF)",
    "IWM": "IWM (Russell 2000 ETF)",
}


def _ensure_output_dirs() -> None:
    return None


def get_multiseed_paths(asset_symbol: str) -> dict[str, Path]:
    normalized_asset = normalize_asset_symbol(asset_symbol)
    if normalized_asset == DEFAULT_ASSET:
        tables_dir = TABLES_DIR / "multiseed_rl"
        models_dir = RESULTS_DIR / "models" / "multiseed_rl"
        figure_path = FIGURES_DIR / "cumulative_return_all_methods_multiseed.png"
        financial_portfolios_csv = TABLES_DIR / "financial baselines" / "financial_baselines_portfolios.csv"
    else:
        tables_dir = cross_asset_tables_dir(normalized_asset) / "multiseed_rl"
        models_dir = cross_asset_models_dir(normalized_asset) / "multiseed_rl"
        figure_path = cross_asset_figures_dir(normalized_asset) / "cumulative_return_all_methods_multiseed.png"
        financial_portfolios_csv = (
            cross_asset_tables_dir(normalized_asset) / "financial_baselines" / "financial_baselines_portfolios.csv"
        )

    return {
        "tables_dir": tables_dir,
        "models_dir": models_dir,
        "figure_path": figure_path,
        "per_seed_metrics_csv": tables_dir / "per_seed_metrics.csv",
        "per_seed_portfolios_csv": tables_dir / "per_seed_portfolios.csv",
        "summary_mean_std_csv": tables_dir / "summary_mean_std_metrics.csv",
        "summary_mean_std_numeric_csv": tables_dir / "summary_mean_std_metrics_numeric.csv",
        "summary_mean_std_json": tables_dir / "summary_mean_std_metrics.json",
        "mean_cumulative_return_csv": tables_dir / "mean_cumulative_returns.csv",
        "all_methods_metrics_csv": tables_dir / "all_methods_metrics_multiseed.csv",
        "financial_portfolios_csv": financial_portfolios_csv,
    }


def _load_best_proposed_settings(asset_symbol: str) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "lambda_base": 0.05,
        "alpha": 0.50,
        "beta_target": 0.02,
        "timesteps": 30_000,
    }
    best_config_json = Path(get_asset_context(asset_symbol)["best_config_json"])
    if not best_config_json.exists():
        return defaults

    payload = json.loads(best_config_json.read_text(encoding="utf-8"))
    return {
        "lambda_base": float(payload.get("lambda_base", defaults["lambda_base"])),
        "alpha": float(payload.get("alpha", defaults["alpha"])),
        "beta_target": float(payload.get("beta_target", defaults["beta_target"])),
        "timesteps": int(payload.get("timesteps", defaults["timesteps"])),
    }


def build_base_rl_configs(asset_symbol: str) -> list[RLBaselineConfig]:
    return [
        build_ppo_profit_only_config(asset_symbol=asset_symbol),
        build_sac_profit_only_config(asset_symbol=asset_symbol),
        build_ppo_variance_penalized_config(asset_symbol=asset_symbol),
        build_ppo_markovian_mdd_static_config(asset_symbol=asset_symbol),
    ]


def build_proposed_final_config(asset_symbol: str, seed: int) -> RLBaselineConfig:
    settings = _load_best_proposed_settings(asset_symbol)
    combined_train_path = create_train_validation_combined_csv(asset_symbol=asset_symbol)
    config = build_single_config(
        asset_symbol=asset_symbol,
        total_timesteps=int(settings["timesteps"]),
        seed=seed,
        lambda_base=float(settings["lambda_base"]),
        alpha=float(settings["alpha"]),
        beta_target=float(settings["beta_target"]),
    )
    config.train_data_path = combined_train_path
    config.retrain_if_exists = False
    return config


def make_seeded_config(base_config: RLBaselineConfig, method_name: str, seed: int, paths: dict[str, Path]) -> RLBaselineConfig:
    return replace(
        base_config,
        name=f"{method_name}_seed_{seed}",
        seed=seed,
        retrain_if_exists=False,
        model_dir=paths["models_dir"] / method_name,
        tables_dir=paths["tables_dir"] / method_name,
        figures_dir=FIGURES_DIR,
    )


def _diagnostics_to_frame(
    diagnostics: pd.DataFrame,
    *,
    method_name: str,
    seed: int | None,
) -> pd.DataFrame:
    frame = diagnostics.copy().reset_index()
    if "date" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "date"})
    frame["date"] = pd.to_datetime(frame["date"])
    initial_portfolio_value = float(frame["portfolio_value"].iloc[0])
    frame["method"] = method_name
    frame["seed"] = seed
    frame["cumulative_return"] = (frame["portfolio_value"] / initial_portfolio_value) - 1.0
    return frame


def run_multiseed_rl_experiments(asset_symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    portfolio_frames: list[pd.DataFrame] = []
    paths = get_multiseed_paths(asset_symbol)

    base_configs = build_base_rl_configs(asset_symbol)
    proposed_template_seed = SEEDS[0]
    proposed_template = build_proposed_final_config(asset_symbol, proposed_template_seed)

    method_templates: list[tuple[str, RLBaselineConfig]] = [
        (config.name, config) for config in base_configs
    ]
    method_templates.append((PROPOSED_METHOD_ID, proposed_template))

    for method_name, template in method_templates:
        for seed in SEEDS:
            seeded_template = template if method_name != PROPOSED_METHOD_ID else build_proposed_final_config(asset_symbol, seed)
            seeded_config = make_seeded_config(seeded_template, method_name=method_name, seed=seed, paths=paths)
            print(f"\n=== Training and evaluating {method_name} with seed={seed} ===")
            model_path = train_model(seeded_config)
            result = evaluate_saved_model(seeded_config, model_path=model_path)
            save_single_result(seeded_config, result)

            metric_rows.append(
                {
                    "method": method_name,
                    "seed": seed,
                    "run_name": seeded_config.name,
                    "algorithm": seeded_config.algorithm,
                    "reward_mode": seeded_config.reward_mode,
                    "model_path": str(model_path),
                    **{metric_name: float(metric_value) for metric_name, metric_value in result["metrics"].items()},
                }
            )

            portfolio_frames.append(
                _diagnostics_to_frame(
                    result["diagnostics"],
                    method_name=method_name,
                    seed=seed,
                )
            )

    metrics_frame = pd.DataFrame(metric_rows).sort_values(["method", "seed"]).reset_index(drop=True)
    portfolios_frame = pd.concat(portfolio_frames, ignore_index=True)
    return metrics_frame, portfolios_frame


def build_summary_tables(per_seed_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_summary = (
        per_seed_metrics.groupby("method")[METRIC_COLUMNS]
        .agg(["mean", "std"])
        .sort_index()
    )
    numeric_summary.columns = [f"{metric}_{stat}" for metric, stat in numeric_summary.columns]
    numeric_summary = numeric_summary.reset_index()

    display_summary = numeric_summary[["method"]].copy()
    for metric_name in METRIC_COLUMNS:
        mean_column = f"{metric_name}_mean"
        std_column = f"{metric_name}_std"
        display_summary[metric_name] = numeric_summary.apply(
            lambda row: f"{row[mean_column]:.6f} +/- {row[std_column]:.6f}",
            axis=1,
        )

    return numeric_summary, display_summary


def build_mean_cumulative_return_frame(per_seed_portfolios: pd.DataFrame) -> pd.DataFrame:
    rl_mean = (
        per_seed_portfolios.groupby(["method", "date"], as_index=False)["cumulative_return"]
        .mean()
        .sort_values(["method", "date"])
        .reset_index(drop=True)
    )
    return rl_mean


def load_financial_diagnostics(asset_symbol: str) -> pd.DataFrame:
    financial_portfolios_csv = get_multiseed_paths(asset_symbol)["financial_portfolios_csv"]
    if not financial_portfolios_csv.exists():
        raise FileNotFoundError(
            f"Missing financial baseline diagnostics at '{financial_portfolios_csv}'. "
            "Run the financial baselines first."
        )

    frame = pd.read_csv(financial_portfolios_csv, parse_dates=["date"])
    if "baseline" not in frame.columns:
        raise ValueError("Financial baseline diagnostics must contain a 'baseline' column.")

    frame = frame.rename(columns={"baseline": "method"}).copy()
    frames: list[pd.DataFrame] = []
    for method_name, subset in frame.groupby("method"):
        frames.append(
            _diagnostics_to_frame(
                subset.drop(columns=["method"]),
                method_name=method_name,
                seed=None,
            )
        )
    return pd.concat(frames, ignore_index=True)


def compute_metrics_from_diagnostics(diagnostics: pd.DataFrame) -> dict[str, float]:
    return compute_performance_metrics(
        portfolio_value=pd.Series(diagnostics["portfolio_value"].to_numpy()),
        daily_returns=pd.Series(diagnostics["daily_returns"].to_numpy()),
        turnover=pd.Series(diagnostics["turnover"].to_numpy()),
    )


def build_all_methods_metrics(numeric_summary: pd.DataFrame, financial_diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for method_name, subset in financial_diagnostics.groupby("method"):
        metrics = compute_metrics_from_diagnostics(subset)
        rows.append({"method": method_name, **metrics})

    for _, row in numeric_summary.iterrows():
        rows.append(
            {
                "method": row["method"],
                **{metric_name: float(row[f"{metric_name}_mean"]) for metric_name in METRIC_COLUMNS},
            }
        )

    return pd.DataFrame(rows).set_index("method").loc[PLOT_ORDER].reset_index()


def plot_cumulative_returns(mean_cumulative_returns: pd.DataFrame, asset_symbol: str) -> Path:
    normalized_asset = normalize_asset_symbol(asset_symbol)
    figure_path = get_multiseed_paths(asset_symbol)["figure_path"]
    figure_path.parent.mkdir(parents=True, exist_ok=True)

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
        }
    )

    fig, ax = plt.subplots(figsize=(15.2, 8.2), dpi=300)
    for method_name in PLOT_ORDER:
        subset = mean_cumulative_returns.loc[mean_cumulative_returns["method"] == method_name].copy()
        if subset.empty:
            continue
        ax.plot(
            subset["date"],
            subset["cumulative_return"],
            linewidth=2.2,
            label=METHOD_LABEL_MAP.get(method_name, method_name),
            color=COLOR_MAP.get(method_name),
        )

    fig.suptitle(
        f"Cumulative Return Comparison Across All Methods\n{ASSET_DISPLAY_NAMES[normalized_asset]}",
        y=0.97,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(True, alpha=0.4, linewidth=0.8)
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=4,
        frameon=True,
        columnspacing=1.2,
        handlelength=2.2,
    )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")

    fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.13)
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def main(asset_symbol: str = DEFAULT_ASSET) -> None:
    paths = get_multiseed_paths(asset_symbol)
    paths["tables_dir"].mkdir(parents=True, exist_ok=True)
    paths["models_dir"].mkdir(parents=True, exist_ok=True)
    paths["figure_path"].parent.mkdir(parents=True, exist_ok=True)

    per_seed_metrics, per_seed_portfolios = run_multiseed_rl_experiments(asset_symbol)
    numeric_summary, display_summary = build_summary_tables(per_seed_metrics)
    rl_mean_cumulative_returns = build_mean_cumulative_return_frame(per_seed_portfolios)
    financial_diagnostics = load_financial_diagnostics(asset_symbol)
    financial_mean_cumulative_returns = financial_diagnostics[["method", "date", "cumulative_return"]].copy()

    mean_cumulative_returns = (
        pd.concat([financial_mean_cumulative_returns, rl_mean_cumulative_returns], ignore_index=True)
        .sort_values(["method", "date"])
        .reset_index(drop=True)
    )
    all_methods_metrics = build_all_methods_metrics(numeric_summary, financial_diagnostics)

    per_seed_metrics.to_csv(paths["per_seed_metrics_csv"], index=False)
    per_seed_portfolios.to_csv(paths["per_seed_portfolios_csv"], index=False)
    numeric_summary.to_csv(paths["summary_mean_std_numeric_csv"], index=False)
    display_summary.to_csv(paths["summary_mean_std_csv"], index=False)
    mean_cumulative_returns.to_csv(paths["mean_cumulative_return_csv"], index=False)
    all_methods_metrics.to_csv(paths["all_methods_metrics_csv"], index=False)

    paths["summary_mean_std_json"].write_text(
        json.dumps(
            {
                row["method"]: {
                    metric_name: {
                        "mean": float(row[f"{metric_name}_mean"]),
                        "std": float(row[f"{metric_name}_std"]),
                    }
                    for metric_name in METRIC_COLUMNS
                }
                for _, row in numeric_summary.iterrows()
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    figure_path = plot_cumulative_returns(mean_cumulative_returns, asset_symbol)

    print("\nSaved multi-seed RL reliability outputs:")
    print(f"  seeds: {SEEDS}")
    print(f"  asset: {asset_symbol}")
    print(f"  per-seed metrics: {paths['per_seed_metrics_csv']}")
    print(f"  summary mean/std: {paths['summary_mean_std_csv']}")
    print(f"  numeric summary: {paths['summary_mean_std_numeric_csv']}")
    print(f"  per-seed portfolios: {paths['per_seed_portfolios_csv']}")
    print(f"  mean cumulative returns: {paths['mean_cumulative_return_csv']}")
    print(f"  all-method metrics: {paths['all_methods_metrics_csv']}")
    print(f"  figure: {figure_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-seed RL reliability evaluation for a selected asset.")
    parser.add_argument(
        "--asset",
        default=DEFAULT_ASSET,
        help="Asset symbol to evaluate. Supported: SPY, QQQ, DIA, IWM.",
    )
    args = parser.parse_args()
    main(asset_symbol=args.asset)
