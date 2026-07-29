from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from baselines.common.asset_utils import (
    DEFAULT_ASSET,
    asset_slug,
    cross_asset_figures_dir,
    cross_asset_models_dir,
    cross_asset_tables_dir,
    get_processed_split_paths,
    normalize_asset_symbol,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.common.analysis_utils import (
    compute_regime_metrics,
    plot_all_methods_equity,
    plot_lambda_vs_vix,
    plot_pareto_frontier,
    plot_regime_bar_comparison,
    plot_regime_equity_with_background,
    save_regime_metrics,
)
from baselines.common.metrics import compute_performance_metrics
from baselines.rl.rl_baseline_common import (
    FIGURES_DIR,
    RESULTS_DIR,
    RL_TABLES_DIR,
    RLBaselineConfig,
    TABLES_DIR,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    evaluate_saved_model,
    load_market_data,
    save_single_result,
    train_model,
)


PROPOSED_METHOD_ID = "ppo_hybrid_regime_aware_policy"
PROPOSED_RESULTS_DIR = RESULTS_DIR / PROPOSED_METHOD_ID
PROPOSED_MODELS_DIR = PROPOSED_RESULTS_DIR / "models"
PROPOSED_TABLES_DIR = TABLES_DIR / PROPOSED_METHOD_ID
PROPOSED_FIGURES_DIR = FIGURES_DIR / PROPOSED_METHOD_ID
GRID_SUMMARY_CSV = PROPOSED_TABLES_DIR / "grid_search_validation.csv"
GRID_SUMMARY_JSON = PROPOSED_TABLES_DIR / "grid_search_validation.json"
BEST_CONFIG_JSON = PROPOSED_TABLES_DIR / "best_config.json"
PROPOSED_METRICS_CSV = PROPOSED_TABLES_DIR / "proposed_method_metrics.csv"
PROPOSED_METRICS_JSON = PROPOSED_TABLES_DIR / "proposed_method_metrics.json"
PROPOSED_PORTFOLIOS_CSV = PROPOSED_TABLES_DIR / "proposed_method_portfolios.csv"
REGIME_METRICS_CSV = PROPOSED_TABLES_DIR / "all_methods_regime_metrics.csv"
REGIME_METRICS_JSON = PROPOSED_TABLES_DIR / "all_methods_regime_metrics.json"
VALIDATION_SUMMARY_JSON = PROPOSED_TABLES_DIR / "validation_candidate_summary.json"


def get_proposed_feature_columns() -> list[str]:
    return [
        "log_return",
        "sma_ratio",
        "rsi_14",
        "bollinger_band_width",
        "vix_zscore_252",
        "ret_5d",
        "ret_20d",
        "ma_spread_5_20",
        "vix_change_5d",
    ]


def get_proposed_reward_kwargs(beta_target: float = 0.02) -> dict[str, float]:
    return {
        "beta_target": beta_target,
        "beta_turnover": 0.0015,
        "action_prior_weight": 0.60,
        "stress_threshold": 0.80,
        "crisis_threshold": 1.80,
        "bull_ret20_threshold": 0.0,
        "bull_ma_threshold": 0.0,
        "bear_ret20_threshold": 0.0,
        "bear_ma_threshold": 0.0,
        "weight_bull": 1.0,
        "weight_bull_stress": 0.50,
        "weight_neutral": 0.30,
        "weight_neutral_negative": 0.10,
        "weight_bear": 0.0,
        "weight_bear_stress": -0.10,
        "weight_bear_crisis": -0.80,
    }


def get_proposed_algo_kwargs() -> dict[str, float | int]:
    return {
        "learning_rate": 1e-4,
        "n_steps": 1024,
        "batch_size": 128,
        "n_epochs": 10,
        "ent_coef": 0.0,
    }


def get_asset_context(asset_symbol: str = DEFAULT_ASSET) -> dict[str, Path | str]:
    normalized_asset = normalize_asset_symbol(asset_symbol)
    if normalized_asset == DEFAULT_ASSET:
        return {
            "asset_symbol": normalized_asset,
            "train_data_path": TRAIN_DATA_PATH,
            "validation_data_path": VALIDATION_DATA_PATH,
            "test_data_path": TEST_DATA_PATH,
            "results_dir": PROPOSED_RESULTS_DIR,
            "models_dir": PROPOSED_MODELS_DIR,
            "tables_dir": PROPOSED_TABLES_DIR,
            "figures_dir": PROPOSED_FIGURES_DIR,
            "grid_summary_csv": GRID_SUMMARY_CSV,
            "grid_summary_json": GRID_SUMMARY_JSON,
            "best_config_json": BEST_CONFIG_JSON,
            "proposed_metrics_csv": PROPOSED_METRICS_CSV,
            "proposed_metrics_json": PROPOSED_METRICS_JSON,
            "proposed_portfolios_csv": PROPOSED_PORTFOLIOS_CSV,
            "regime_metrics_csv": REGIME_METRICS_CSV,
            "regime_metrics_json": REGIME_METRICS_JSON,
            "validation_summary_json": VALIDATION_SUMMARY_JSON,
            "pipeline_summary_json": PROPOSED_TABLES_DIR / "pipeline_summary.json",
            "combined_train_path": PROPOSED_TABLES_DIR / "train_validation_combined.csv",
            "all_methods_metrics_csv": PROPOSED_TABLES_DIR / "all_methods_metrics.csv",
        }

    split_paths = get_processed_split_paths(normalized_asset)
    asset_tables_dir = cross_asset_tables_dir(normalized_asset) / PROPOSED_METHOD_ID
    asset_figures_dir = cross_asset_figures_dir(normalized_asset) / PROPOSED_METHOD_ID
    asset_models_dir = cross_asset_models_dir(normalized_asset) / PROPOSED_METHOD_ID
    return {
        "asset_symbol": normalized_asset,
        "train_data_path": split_paths["train"],
        "validation_data_path": split_paths["validation"],
        "test_data_path": split_paths["test"],
        "results_dir": cross_asset_models_dir(normalized_asset).parent / asset_slug(normalized_asset),
        "models_dir": asset_models_dir,
        "tables_dir": asset_tables_dir,
        "figures_dir": asset_figures_dir,
        "grid_summary_csv": asset_tables_dir / "grid_search_validation.csv",
        "grid_summary_json": asset_tables_dir / "grid_search_validation.json",
        "best_config_json": asset_tables_dir / "best_config.json",
        "proposed_metrics_csv": asset_tables_dir / "proposed_method_metrics.csv",
        "proposed_metrics_json": asset_tables_dir / "proposed_method_metrics.json",
        "proposed_portfolios_csv": asset_tables_dir / "proposed_method_portfolios.csv",
        "regime_metrics_csv": asset_tables_dir / "all_methods_regime_metrics.csv",
        "regime_metrics_json": asset_tables_dir / "all_methods_regime_metrics.json",
        "validation_summary_json": asset_tables_dir / "validation_candidate_summary.json",
        "pipeline_summary_json": asset_tables_dir / "pipeline_summary.json",
        "combined_train_path": asset_tables_dir / "train_validation_combined.csv",
        "all_methods_metrics_csv": asset_tables_dir / "all_methods_metrics.csv",
    }


def build_single_config(
    *,
    asset_symbol: str = DEFAULT_ASSET,
    total_timesteps: int = 30_000,
    seed: int = 42,
    lambda_base: float = 0.05,
    alpha: float = 0.50,
    beta_target: float = 0.02,
) -> RLBaselineConfig:
    asset_context = get_asset_context(asset_symbol)
    return RLBaselineConfig(
        name=PROPOSED_METHOD_ID,
        algorithm="ppo",
        reward_mode="ppo_hybrid_regime_aware_policy",
        asset_symbol=asset_context["asset_symbol"],
        total_timesteps=total_timesteps,
        seed=seed,
        device="cpu",
        lambda_base=lambda_base,
        alpha=alpha,
        feature_columns=get_proposed_feature_columns(),
        reward_kwargs=get_proposed_reward_kwargs(beta_target=beta_target),
        train_data_path=asset_context["train_data_path"],
        validation_data_path=asset_context["validation_data_path"],
        test_data_path=asset_context["test_data_path"],
        model_dir=asset_context["models_dir"],
        figures_dir=asset_context["figures_dir"],
        tables_dir=asset_context["tables_dir"],
        algo_kwargs=get_proposed_algo_kwargs(),
        deterministic_eval=True,
        retrain_if_exists=True,
    )


def build_grid_configs(asset_symbol: str = DEFAULT_ASSET, total_timesteps: int = 200_000, seed: int = 42) -> list[RLBaselineConfig]:
    asset_context = get_asset_context(asset_symbol)
    configs: list[RLBaselineConfig] = []
    for lambda_base in (0.15, 0.20, 0.30):
        for alpha in (0.10, 0.20, 0.30):
            configs.append(
                RLBaselineConfig(
                    name=f"ppo_mdd_rrc_lb_{str(lambda_base).replace('.', 'p')}_a_{str(alpha).replace('.', 'p')}",
                    algorithm="ppo",
                    reward_mode="ppo_hybrid_regime_aware_policy",
                    asset_symbol=asset_context["asset_symbol"],
                    total_timesteps=total_timesteps,
                    seed=seed,
                    device="cpu",
                    lambda_base=lambda_base,
                    alpha=alpha,
                    feature_columns=get_proposed_feature_columns(),
                    reward_kwargs=get_proposed_reward_kwargs(),
                    train_data_path=asset_context["train_data_path"],
                    validation_data_path=asset_context["validation_data_path"],
                    test_data_path=asset_context["test_data_path"],
                    model_dir=Path(asset_context["models_dir"]) / "grid",
                    figures_dir=asset_context["figures_dir"],
                    tables_dir=Path(asset_context["tables_dir"]) / "grid",
                    algo_kwargs=get_proposed_algo_kwargs(),
                    deterministic_eval=True,
                    retrain_if_exists=True,
                )
            )
    return configs


def build_best_retrain_config(best_config: RLBaselineConfig, total_timesteps: int = 200_000) -> RLBaselineConfig:
    asset_context = get_asset_context(best_config.asset_symbol)
    combined_train_path = Path(asset_context["combined_train_path"])
    return RLBaselineConfig(
        name=PROPOSED_METHOD_ID,
        algorithm="ppo",
        reward_mode="ppo_hybrid_regime_aware_policy",
        asset_symbol=best_config.asset_symbol,
        total_timesteps=total_timesteps,
        seed=best_config.seed,
        device="cpu",
        lambda_base=best_config.lambda_base,
        alpha=best_config.alpha,
        feature_columns=list(best_config.feature_columns or get_proposed_feature_columns()),
        reward_kwargs=dict(best_config.reward_kwargs),
        train_data_path=combined_train_path,
        validation_data_path=best_config.validation_data_path,
        test_data_path=best_config.test_data_path,
        model_dir=asset_context["models_dir"],
        figures_dir=asset_context["figures_dir"],
        tables_dir=asset_context["tables_dir"],
        algo_kwargs=dict(best_config.algo_kwargs),
        deterministic_eval=True,
        retrain_if_exists=True,
    )


def create_train_validation_combined_csv(asset_symbol: str = DEFAULT_ASSET) -> Path:
    asset_context = get_asset_context(asset_symbol)
    train_frame = load_market_data(asset_context["train_data_path"])
    validation_frame = load_market_data(asset_context["validation_data_path"])
    combined = pd.concat([train_frame, validation_frame], ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)
    tables_dir = Path(asset_context["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(asset_context["combined_train_path"])
    combined.to_csv(output_path, index=False)
    return output_path


def evaluate_on_validation(config: RLBaselineConfig, model_path: str | Path) -> dict[str, Any]:
    validation_config = RLBaselineConfig(
        **{
            **asdict(config),
            "train_data_path": config.train_data_path,
            "validation_data_path": config.validation_data_path,
            "test_data_path": config.validation_data_path,
            "model_dir": config.model_dir,
            "tables_dir": config.tables_dir,
            "figures_dir": config.figures_dir,
        }
    )
    result = evaluate_saved_model(validation_config, model_path=model_path)
    final_return = float(result["portfolio_value"].iloc[-1] / result["portfolio_value"].iloc[0] - 1.0)
    return {
        "config": config,
        "model_path": str(model_path),
        "validation_final_return": final_return,
        "validation_metrics": result["metrics"],
        "validation_result": result,
    }


def compute_behavior_stats(diagnostics: pd.DataFrame) -> dict[str, float]:
    frame = diagnostics.copy()
    weights = frame["weight_spy"].fillna(0.0)
    vix_z = frame["vix_zscore_t"].fillna(0.0)

    high_vix_mask = vix_z > 0.0
    low_vix_mask = ~high_vix_mask

    return {
        "mean_weight": float(weights.mean()),
        "max_weight": float(weights.max()),
        "min_weight": float(weights.min()),
        "pct_full_long": float((weights.round(6) >= 0.999999).mean()),
        "pct_short": float((weights < 0.0).mean()),
        "avg_weight_high_vix": float(weights.loc[high_vix_mask].mean()) if high_vix_mask.any() else float("nan"),
        "avg_weight_low_vix": float(weights.loc[low_vix_mask].mean()) if low_vix_mask.any() else float("nan"),
    }


def passes_behavior_filter(behavior_stats: dict[str, float]) -> bool:
    pct_full_long = behavior_stats["pct_full_long"]
    pct_short = behavior_stats["pct_short"]
    mean_weight = abs(behavior_stats["mean_weight"])
    avg_weight_high_vix = behavior_stats["avg_weight_high_vix"]
    avg_weight_low_vix = behavior_stats["avg_weight_low_vix"]
    high_vs_low_ok = (
        pd.notna(avg_weight_high_vix)
        and pd.notna(avg_weight_low_vix)
        and avg_weight_high_vix < avg_weight_low_vix
    )
    return pct_full_long < 0.65 and pct_short < 0.65 and mean_weight < 0.75 and high_vs_low_ok


def run_grid_search(asset_symbol: str = DEFAULT_ASSET, total_timesteps: int = 200_000, seed: int = 42) -> tuple[pd.DataFrame, RLBaselineConfig]:
    asset_context = get_asset_context(asset_symbol)
    Path(asset_context["tables_dir"]).mkdir(parents=True, exist_ok=True)
    grid_results: list[dict[str, Any]] = []

    for config in build_grid_configs(asset_symbol=asset_symbol, total_timesteps=total_timesteps, seed=seed):
        print(
            f"\n=== Grid train lambda_base={config.lambda_base:.2f}, alpha={config.alpha:.2f} "
            f"({config.total_timesteps} timesteps) ==="
        )
        model_path = train_model(config)
        validation_payload = evaluate_on_validation(config, model_path=model_path)
        validation_metrics = validation_payload["validation_metrics"]
        grid_results.append(
            {
                "name": config.name,
                "lambda_base": config.lambda_base,
                "alpha": config.alpha,
                "model_path": str(model_path),
                "validation_final_return": validation_payload["validation_final_return"],
                "validation_sharpe": validation_metrics["Sharpe Ratio"],
                "validation_mdd": validation_metrics["MDD"],
                "validation_calmar": validation_metrics["Calmar Ratio"],
            }
        )

    grid_frame = pd.DataFrame(grid_results).sort_values(
        ["validation_final_return", "validation_sharpe"],
        ascending=[False, False],
    ).reset_index(drop=True)
    Path(asset_context["grid_summary_csv"]).parent.mkdir(parents=True, exist_ok=True)
    grid_frame.to_csv(asset_context["grid_summary_csv"], index=False)
    Path(asset_context["grid_summary_json"]).write_text(
        grid_frame.to_json(orient="records", indent=2),
        encoding="utf-8",
    )

    best_row = grid_frame.iloc[0]
    best_config = next(
        config
        for config in build_grid_configs(asset_symbol=asset_symbol, total_timesteps=total_timesteps, seed=seed)
        if config.name == best_row["name"]
    )
    Path(asset_context["best_config_json"]).write_text(
        json.dumps(
            {
                "selected_by": "validation_final_return",
                "best_name": best_config.name,
                "lambda_base": best_config.lambda_base,
                "alpha": best_config.alpha,
                "timesteps": best_config.total_timesteps,
                "seed": best_config.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return grid_frame, best_config


def save_proposed_test_outputs(result: dict[str, Any]) -> tuple[Path, Path]:
    asset_context = get_asset_context(result["baseline_asset_symbol"])
    diagnostics = result["diagnostics"].copy()
    diagnostics.insert(0, "method", PROPOSED_METHOD_ID)
    diagnostics.insert(1, "date", diagnostics.index)
    diagnostics.reset_index(drop=True).to_csv(asset_context["proposed_portfolios_csv"], index=False)

    payload = {
        "method": PROPOSED_METHOD_ID,
        "metrics": {key: float(value) for key, value in result["metrics"].items()},
        "model_path": result["model_path"],
    }
    Path(asset_context["proposed_metrics_json"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame([{"method": payload["method"], **payload["metrics"]}]).to_csv(
        asset_context["proposed_metrics_csv"],
        index=False,
    )
    return Path(asset_context["proposed_portfolios_csv"]), Path(asset_context["proposed_metrics_json"])


def load_financial_results(asset_symbol: str = DEFAULT_ASSET) -> dict[str, pd.DataFrame]:
    if asset_symbol == DEFAULT_ASSET:
        path = TABLES_DIR / "financial_baselines_portfolios.csv"
    else:
        path = cross_asset_tables_dir(asset_symbol) / "financial_baselines" / "financial_baselines_portfolios.csv"
    frame = pd.read_csv(path, parse_dates=["date"])
    results: dict[str, pd.DataFrame] = {}
    for baseline_name, subset in frame.groupby("baseline"):
        results[baseline_name] = subset.copy().reset_index(drop=True)
    return results


def load_rl_baseline_results(asset_symbol: str = DEFAULT_ASSET) -> dict[str, pd.DataFrame]:
    if asset_symbol == DEFAULT_ASSET:
        path = RL_TABLES_DIR / "rl_baselines_portfolios.csv"
    else:
        path = cross_asset_tables_dir(asset_symbol) / "rl_baselines" / "rl_baselines_portfolios.csv"
    frame = pd.read_csv(path, parse_dates=["date"])
    results: dict[str, pd.DataFrame] = {}
    for baseline_name, subset in frame.groupby("baseline"):
        results[baseline_name] = subset.copy().reset_index(drop=True)
    return results


def build_all_method_diagnostics(proposed_result: dict[str, Any]) -> dict[str, pd.DataFrame]:
    asset_symbol = proposed_result["baseline_asset_symbol"]
    financial = load_financial_results(asset_symbol)
    rl = load_rl_baseline_results(asset_symbol)
    proposed_diagnostics = proposed_result["diagnostics"].copy()
    proposed_diagnostics = proposed_diagnostics.reset_index()

    test_market = load_market_data(get_asset_context(asset_symbol)["test_data_path"])[["date", "vix_close", "vix_zscore_252"]].copy()

    merged: dict[str, pd.DataFrame] = {}
    for collection in [financial, rl]:
        for method_name, diagnostics in collection.items():
            merged_frame = diagnostics.merge(test_market, on="date", how="left")
            merged[method_name] = merged_frame

    merged[PROPOSED_METHOD_ID] = proposed_diagnostics.merge(test_market, on="date", how="left")
    return merged


def save_all_method_regime_outputs(
    diagnostics_by_method: dict[str, pd.DataFrame],
    asset_symbol: str = DEFAULT_ASSET,
) -> None:
    asset_context = get_asset_context(asset_symbol)
    regime_metrics = compute_regime_metrics(diagnostics_by_method)
    save_regime_metrics(regime_metrics, asset_context["regime_metrics_csv"], asset_context["regime_metrics_json"])

    plot_regime_bar_comparison(
        regime_metrics=regime_metrics,
        proposed_method_name=PROPOSED_METHOD_ID,
        static_method_name="ppo_markovian_mdd_static",
        output_path=Path(asset_context["figures_dir"]) / "regime_bar_proposed_vs_static.png",
    )
    plot_regime_equity_with_background(
        diagnostics=diagnostics_by_method[PROPOSED_METHOD_ID],
        method_name="PPO Hybrid Regime-Aware Policy (Our Proposed Method)",
        output_path=Path(asset_context["figures_dir"]) / "proposed_equity_with_regimes.png",
    )
    plot_lambda_vs_vix(
        diagnostics=diagnostics_by_method[PROPOSED_METHOD_ID],
        output_path=Path(asset_context["figures_dir"]) / "lambda_rrc_vs_vix.png",
    )

    overall_metrics_rows: list[dict[str, Any]] = []
    for method_name, diagnostics in diagnostics_by_method.items():
        metrics = result_metrics_from_diagnostics(diagnostics)
        overall_metrics_rows.append({"method": method_name, **metrics})
    overall_metrics_frame = pd.DataFrame(overall_metrics_rows)
    overall_metrics_frame.to_csv(asset_context["all_methods_metrics_csv"], index=False)
    plot_pareto_frontier(
        metrics_frame=overall_metrics_frame,
        output_path=Path(asset_context["figures_dir"]) / "pareto_return_vs_mdd.png",
    )
    plot_all_methods_equity(
        diagnostics_by_method,
        output_path=(
            FIGURES_DIR / "backtest_all_methods_equity_comparison.png"
            if asset_symbol == DEFAULT_ASSET
            else Path(asset_context["figures_dir"]).parent / "backtest_all_methods_equity_comparison.png"
        ),
        title=f"Backtest of All Methods ({asset_symbol}): Equity Curve Comparison",
    )


def result_metrics_from_diagnostics(diagnostics: pd.DataFrame) -> dict[str, float]:
    return compute_performance_metrics(
        portfolio_value=pd.Series(diagnostics["portfolio_value"].to_numpy()),
        daily_returns=pd.Series(diagnostics["daily_returns"].to_numpy()),
        turnover=pd.Series(diagnostics["turnover"].to_numpy()),
    )


def run_proposed_method_pipeline(
    asset_symbol: str = DEFAULT_ASSET,
    total_timesteps: int = 200_000,
    seed: int = 42,
) -> dict[str, Any]:
    asset_context = get_asset_context(asset_symbol)
    Path(asset_context["results_dir"]).mkdir(parents=True, exist_ok=True)
    Path(asset_context["tables_dir"]).mkdir(parents=True, exist_ok=True)
    Path(asset_context["figures_dir"]).mkdir(parents=True, exist_ok=True)
    Path(asset_context["models_dir"]).mkdir(parents=True, exist_ok=True)

    grid_frame, best_grid_config = run_grid_search(asset_symbol=asset_symbol, total_timesteps=total_timesteps, seed=seed)
    combined_train_path = create_train_validation_combined_csv(asset_symbol=asset_symbol)
    best_retrain_config = build_best_retrain_config(best_grid_config, total_timesteps=total_timesteps)
    best_retrain_config.train_data_path = combined_train_path

    print(
        f"\n=== Retraining best proposed model on train+validation "
        f"(lambda_base={best_retrain_config.lambda_base:.2f}, alpha={best_retrain_config.alpha:.2f}) ==="
    )
    best_model_path = train_model(best_retrain_config)
    proposed_test_result = evaluate_saved_model(best_retrain_config, model_path=best_model_path)
    save_single_result(best_retrain_config, proposed_test_result)
    save_proposed_test_outputs(proposed_test_result)

    diagnostics_by_method = build_all_method_diagnostics(proposed_test_result)
    save_all_method_regime_outputs(diagnostics_by_method, asset_symbol=asset_symbol)

    summary_payload = {
        "best_grid_config": {
            "name": best_grid_config.name,
            "lambda_base": best_grid_config.lambda_base,
            "alpha": best_grid_config.alpha,
        },
        "retrained_model_path": str(best_model_path),
        "test_metrics": proposed_test_result["metrics"],
        "grid_rows": len(grid_frame),
    }
    Path(asset_context["pipeline_summary_json"]).write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    return {
        "grid_frame": grid_frame,
        "best_grid_config": best_grid_config,
        "best_retrain_config": best_retrain_config,
        "proposed_test_result": proposed_test_result,
    }


def run_proposed_method_single_config(
    *,
    asset_symbol: str = DEFAULT_ASSET,
    total_timesteps: int = 30_000,
    seed: int = 42,
    lambda_base: float = 0.05,
    alpha: float = 0.50,
    beta_target: float = 0.02,
) -> dict[str, Any]:
    asset_context = get_asset_context(asset_symbol)
    Path(asset_context["results_dir"]).mkdir(parents=True, exist_ok=True)
    Path(asset_context["tables_dir"]).mkdir(parents=True, exist_ok=True)
    Path(asset_context["figures_dir"]).mkdir(parents=True, exist_ok=True)
    Path(asset_context["models_dir"]).mkdir(parents=True, exist_ok=True)

    config = build_single_config(
        asset_symbol=asset_symbol,
        total_timesteps=total_timesteps,
        seed=seed,
        lambda_base=lambda_base,
        alpha=alpha,
        beta_target=beta_target,
    )

    print(
        f"\n=== Training proposed candidate on train "
        f"(lambda_base={config.lambda_base:.2f}, alpha={config.alpha:.2f}, "
        f"beta_target={config.reward_kwargs.get('beta_target', 0.0):.4f}, "
        f"ent_coef={config.algo_kwargs.get('ent_coef', 0.0):.4f}) ==="
    )
    validation_model_path = train_model(config)
    validation_payload = evaluate_on_validation(config, model_path=validation_model_path)
    validation_behavior = compute_behavior_stats(validation_payload["validation_result"]["diagnostics"])
    validation_passed = passes_behavior_filter(validation_behavior)

    Path(asset_context["validation_summary_json"]).write_text(
        json.dumps(
            {
                "mode": "single_candidate_validation",
                "config": {
                    "name": config.name,
                    "reward_mode": config.reward_mode,
                    "lambda_base": config.lambda_base,
                    "alpha": config.alpha,
                    "beta_target": config.reward_kwargs.get("beta_target", 0.0),
                    "timesteps": config.total_timesteps,
                    "seed": config.seed,
                    "algo_kwargs": config.algo_kwargs,
                },
                "validation_metrics": validation_payload["validation_metrics"],
                "validation_final_return": validation_payload["validation_final_return"],
                "behavior_stats": validation_behavior,
                "behavior_filter_passed": validation_passed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    combined_train_path = create_train_validation_combined_csv(asset_symbol=asset_symbol)
    retrain_config = build_single_config(
        asset_symbol=asset_symbol,
        total_timesteps=total_timesteps,
        seed=seed,
        lambda_base=lambda_base,
        alpha=alpha,
        beta_target=beta_target,
    )
    retrain_config.train_data_path = combined_train_path

    print(
        f"\n=== Retraining proposed model on train+validation "
        f"(validation Calmar={validation_payload['validation_metrics']['Calmar Ratio']:.4f}, "
        f"behavior_filter_passed={validation_passed}) ==="
    )
    best_model_path = train_model(retrain_config)
    proposed_test_result = evaluate_saved_model(retrain_config, model_path=best_model_path)
    save_single_result(retrain_config, proposed_test_result)
    save_proposed_test_outputs(proposed_test_result)

    Path(asset_context["best_config_json"]).write_text(
        json.dumps(
            {
                "selected_by": "manual_single_candidate_calmar_with_anti_full_long_validation",
                "best_name": retrain_config.name,
                "reward_mode": retrain_config.reward_mode,
                "lambda_base": retrain_config.lambda_base,
                "alpha": retrain_config.alpha,
                "beta_target": retrain_config.reward_kwargs.get("beta_target", 0.0),
                "timesteps": retrain_config.total_timesteps,
                "seed": retrain_config.seed,
                "algo_kwargs": retrain_config.algo_kwargs,
                "validation_metrics": validation_payload["validation_metrics"],
                "validation_behavior": validation_behavior,
                "behavior_filter_passed": validation_passed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    diagnostics_by_method = build_all_method_diagnostics(proposed_test_result)
    save_all_method_regime_outputs(diagnostics_by_method, asset_symbol=asset_symbol)

    summary_payload = {
        "mode": "single_config",
        "config": {
            "name": retrain_config.name,
            "reward_mode": retrain_config.reward_mode,
            "lambda_base": retrain_config.lambda_base,
            "alpha": retrain_config.alpha,
            "beta_target": retrain_config.reward_kwargs.get("beta_target", 0.0),
            "timesteps": retrain_config.total_timesteps,
            "seed": retrain_config.seed,
            "algo_kwargs": retrain_config.algo_kwargs,
        },
        "validation": {
            "metrics": validation_payload["validation_metrics"],
            "final_return": validation_payload["validation_final_return"],
            "behavior_stats": validation_behavior,
            "behavior_filter_passed": validation_passed,
        },
        "model_path": str(best_model_path),
        "test_metrics": proposed_test_result["metrics"],
    }
    Path(asset_context["pipeline_summary_json"]).write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    return {
        "config": retrain_config,
        "validation_payload": validation_payload,
        "validation_behavior": validation_behavior,
        "proposed_test_result": proposed_test_result,
        "model_path": best_model_path,
    }
