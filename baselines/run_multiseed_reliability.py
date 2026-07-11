from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.analysis_utils import METHOD_LABEL_MAP
from baselines.metrics import compute_performance_metrics
from baselines.ppo_hybrid_regime_aware_policy.pipeline import (
    BEST_CONFIG_JSON,
    PROPOSED_METHOD_ID,
    build_single_config,
    create_train_validation_combined_csv,
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
MULTISEED_TABLES_DIR = TABLES_DIR / "multiseed_rl"
MULTISEED_MODELS_DIR = RESULTS_DIR / "models" / "multiseed_rl"
MULTISEED_CUMULATIVE_RETURN_FIGURE = FIGURES_DIR / "cumulative_return_all_methods_multiseed.png"
PER_SEED_METRICS_CSV = MULTISEED_TABLES_DIR / "per_seed_metrics.csv"
PER_SEED_PORTFOLIOS_CSV = MULTISEED_TABLES_DIR / "per_seed_portfolios.csv"
SUMMARY_MEAN_STD_CSV = MULTISEED_TABLES_DIR / "summary_mean_std_metrics.csv"
SUMMARY_MEAN_STD_NUMERIC_CSV = MULTISEED_TABLES_DIR / "summary_mean_std_metrics_numeric.csv"
SUMMARY_MEAN_STD_JSON = MULTISEED_TABLES_DIR / "summary_mean_std_metrics.json"
MEAN_CUMULATIVE_RETURN_CSV = MULTISEED_TABLES_DIR / "mean_cumulative_returns.csv"
ALL_METHODS_METRICS_CSV = MULTISEED_TABLES_DIR / "all_methods_metrics_multiseed.csv"
FINANCIAL_PORTFOLIOS_CSV = TABLES_DIR / "financial baselines" / "financial_baselines_portfolios.csv"

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


def _ensure_output_dirs() -> None:
    MULTISEED_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MULTISEED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _load_best_proposed_settings() -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "lambda_base": 0.05,
        "alpha": 0.50,
        "beta_target": 0.02,
        "timesteps": 30_000,
    }
    if not BEST_CONFIG_JSON.exists():
        return defaults

    payload = json.loads(BEST_CONFIG_JSON.read_text(encoding="utf-8"))
    return {
        "lambda_base": float(payload.get("lambda_base", defaults["lambda_base"])),
        "alpha": float(payload.get("alpha", defaults["alpha"])),
        "beta_target": float(payload.get("beta_target", defaults["beta_target"])),
        "timesteps": int(payload.get("timesteps", defaults["timesteps"])),
    }


def build_base_rl_configs() -> list[RLBaselineConfig]:
    return [
        build_ppo_profit_only_config(),
        build_sac_profit_only_config(),
        build_ppo_variance_penalized_config(),
        build_ppo_markovian_mdd_static_config(),
    ]


def build_proposed_final_config(seed: int) -> RLBaselineConfig:
    settings = _load_best_proposed_settings()
    combined_train_path = create_train_validation_combined_csv()
    config = build_single_config(
        total_timesteps=int(settings["timesteps"]),
        seed=seed,
        lambda_base=float(settings["lambda_base"]),
        alpha=float(settings["alpha"]),
        beta_target=float(settings["beta_target"]),
    )
    config.train_data_path = combined_train_path
    config.retrain_if_exists = False
    return config


def make_seeded_config(base_config: RLBaselineConfig, method_name: str, seed: int) -> RLBaselineConfig:
    return replace(
        base_config,
        name=f"{method_name}_seed_{seed}",
        seed=seed,
        retrain_if_exists=False,
        model_dir=MULTISEED_MODELS_DIR / method_name,
        tables_dir=MULTISEED_TABLES_DIR / method_name,
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


def run_multiseed_rl_experiments() -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    portfolio_frames: list[pd.DataFrame] = []

    base_configs = build_base_rl_configs()
    proposed_template_seed = SEEDS[0]
    proposed_template = build_proposed_final_config(proposed_template_seed)

    method_templates: list[tuple[str, RLBaselineConfig]] = [
        (config.name, config) for config in base_configs
    ]
    method_templates.append((PROPOSED_METHOD_ID, proposed_template))

    for method_name, template in method_templates:
        for seed in SEEDS:
            seeded_template = template if method_name != PROPOSED_METHOD_ID else build_proposed_final_config(seed)
            seeded_config = make_seeded_config(seeded_template, method_name=method_name, seed=seed)
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


def load_financial_diagnostics() -> pd.DataFrame:
    if not FINANCIAL_PORTFOLIOS_CSV.exists():
        raise FileNotFoundError(
            f"Missing financial baseline diagnostics at '{FINANCIAL_PORTFOLIOS_CSV}'. "
            "Run the financial baselines first."
        )

    frame = pd.read_csv(FINANCIAL_PORTFOLIOS_CSV, parse_dates=["date"])
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


def plot_cumulative_returns(mean_cumulative_returns: pd.DataFrame) -> Path:
    MULTISEED_CUMULATIVE_RETURN_FIGURE.parent.mkdir(parents=True, exist_ok=True)

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

    fig, ax = plt.subplots(figsize=(13, 7), dpi=300)
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

    ax.set_title("Cumulative Return Comparison Across All Methods")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(True, alpha=0.4, linewidth=0.8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(MULTISEED_CUMULATIVE_RETURN_FIGURE, bbox_inches="tight")
    plt.close(fig)
    return MULTISEED_CUMULATIVE_RETURN_FIGURE


def main() -> None:
    _ensure_output_dirs()

    per_seed_metrics, per_seed_portfolios = run_multiseed_rl_experiments()
    numeric_summary, display_summary = build_summary_tables(per_seed_metrics)
    rl_mean_cumulative_returns = build_mean_cumulative_return_frame(per_seed_portfolios)
    financial_diagnostics = load_financial_diagnostics()
    financial_mean_cumulative_returns = financial_diagnostics[["method", "date", "cumulative_return"]].copy()

    mean_cumulative_returns = (
        pd.concat([financial_mean_cumulative_returns, rl_mean_cumulative_returns], ignore_index=True)
        .sort_values(["method", "date"])
        .reset_index(drop=True)
    )
    all_methods_metrics = build_all_methods_metrics(numeric_summary, financial_diagnostics)

    per_seed_metrics.to_csv(PER_SEED_METRICS_CSV, index=False)
    per_seed_portfolios.to_csv(PER_SEED_PORTFOLIOS_CSV, index=False)
    numeric_summary.to_csv(SUMMARY_MEAN_STD_NUMERIC_CSV, index=False)
    display_summary.to_csv(SUMMARY_MEAN_STD_CSV, index=False)
    mean_cumulative_returns.to_csv(MEAN_CUMULATIVE_RETURN_CSV, index=False)
    all_methods_metrics.to_csv(ALL_METHODS_METRICS_CSV, index=False)

    SUMMARY_MEAN_STD_JSON.write_text(
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

    figure_path = plot_cumulative_returns(mean_cumulative_returns)

    print("\nSaved multi-seed RL reliability outputs:")
    print(f"  seeds: {SEEDS}")
    print(f"  per-seed metrics: {PER_SEED_METRICS_CSV}")
    print(f"  summary mean/std: {SUMMARY_MEAN_STD_CSV}")
    print(f"  numeric summary: {SUMMARY_MEAN_STD_NUMERIC_CSV}")
    print(f"  per-seed portfolios: {PER_SEED_PORTFOLIOS_CSV}")
    print(f"  mean cumulative returns: {MEAN_CUMULATIVE_RETURN_CSV}")
    print(f"  all-method metrics: {ALL_METHODS_METRICS_CSV}")
    print(f"  figure: {figure_path}")


if __name__ == "__main__":
    main()
