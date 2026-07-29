from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.common.asset_utils import (
    DEFAULT_ASSET,
    SUPPORTED_ASSETS,
    cross_asset_figures_dir,
    cross_asset_tables_dir,
    normalize_asset_symbol,
)
from baselines.financial.financial_baselines import run_all_financial_baselines
from baselines.ppo_hybrid_regime_aware_policy.pipeline import run_proposed_method_pipeline
from baselines.rl.rl_baseline_common import plot_rl_equity_curves, run_single_rl_baseline, save_aggregate_results
from baselines.rl.train_ppo_markovian_mdd_static import build_config as build_ppo_markovian_mdd_static_config
from baselines.rl.train_ppo_profit_only import build_config as build_ppo_profit_only_config
from baselines.rl.train_ppo_variance_penalized import build_config as build_ppo_variance_penalized_config
from baselines.rl.train_sac_profit_only import build_config as build_sac_profit_only_config
from baselines.experiments.run_multiseed_reliability import main as run_multiseed_reliability


def run_rl_baselines_for_asset(asset_symbol: str) -> dict[str, dict]:
    asset_tables_dir = cross_asset_tables_dir(asset_symbol) / "rl_baselines"
    asset_figures_dir = cross_asset_figures_dir(asset_symbol) / "rl_baselines"
    asset_tables_dir.mkdir(parents=True, exist_ok=True)
    asset_figures_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        build_ppo_profit_only_config(asset_symbol=asset_symbol),
        build_sac_profit_only_config(asset_symbol=asset_symbol),
        build_ppo_variance_penalized_config(asset_symbol=asset_symbol),
        build_ppo_markovian_mdd_static_config(asset_symbol=asset_symbol),
    ]
    results: dict[str, dict] = {}
    for config in configs:
        config.tables_dir = asset_tables_dir
        config.figures_dir = asset_figures_dir
        config.model_dir = Path(config.model_dir).parent / asset_symbol.lower() / config.name
        config.retrain_if_exists = False
        print(f"\n=== Training RL baseline {config.name} on {asset_symbol} ===")
        results[config.name] = run_single_rl_baseline(config)

    plot_rl_equity_curves(
        results,
        output_path=asset_figures_dir / "rl_baseline.png",
        title=f"Backtest of RL Baselines ({asset_symbol}): Equity Curve Comparison",
    )
    save_aggregate_results(results, tables_dir=asset_tables_dir)
    return results


def build_cross_asset_summary(assets: list[str]) -> None:
    summary_rows: list[dict[str, object]] = []
    multiseed_rows: list[pd.DataFrame] = []

    for asset_symbol in assets:
        proposed_metrics_path = cross_asset_tables_dir(asset_symbol) / "ppo_hybrid_regime_aware_policy" / "all_methods_metrics.csv"
        multiseed_summary_path = cross_asset_tables_dir(asset_symbol) / "multiseed_rl" / "summary_mean_std_metrics_numeric.csv"

        if proposed_metrics_path.exists():
            frame = pd.read_csv(proposed_metrics_path)
            frame.insert(0, "asset", asset_symbol)
            summary_rows.extend(frame.to_dict(orient="records"))

        if multiseed_summary_path.exists():
            frame = pd.read_csv(multiseed_summary_path)
            frame.insert(0, "asset", asset_symbol)
            multiseed_rows.append(frame)

    root_tables_dir = cross_asset_tables_dir(DEFAULT_ASSET).parent
    root_tables_dir.mkdir(parents=True, exist_ok=True)
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(root_tables_dir / "all_assets_all_methods_metrics.csv", index=False)
    if multiseed_rows:
        pd.concat(multiseed_rows, ignore_index=True).to_csv(
            root_tables_dir / "all_assets_multiseed_summary_numeric.csv",
            index=False,
        )


def run_asset_suite(asset_symbol: str) -> None:
    print(f"\n================ {asset_symbol} ================")
    run_all_financial_baselines(asset_symbol=asset_symbol)
    run_rl_baselines_for_asset(asset_symbol)
    run_proposed_method_pipeline(asset_symbol=asset_symbol)
    run_multiseed_reliability(asset_symbol=asset_symbol)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cross-asset baselines, proposed method, and multi-seed reliability.")
    parser.add_argument(
        "--assets",
        nargs="+",
        default=[DEFAULT_ASSET, *[asset for asset in SUPPORTED_ASSETS if asset != DEFAULT_ASSET]],
        help=f"Assets to run. Supported: {', '.join(SUPPORTED_ASSETS)}.",
    )
    args = parser.parse_args()

    assets = [normalize_asset_symbol(asset_symbol) for asset_symbol in args.assets]
    for asset_symbol in assets:
        run_asset_suite(asset_symbol)

    build_cross_asset_summary(assets)

    print("\nFinished cross-asset experiment suite.")
    print(json.dumps({"assets": assets}, indent=2))


if __name__ == "__main__":
    main()
