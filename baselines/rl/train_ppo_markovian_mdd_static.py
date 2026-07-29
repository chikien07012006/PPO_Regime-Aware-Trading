from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.rl.rl_baseline_common import RLBaselineConfig, get_data_split_paths, run_single_rl_baseline


def build_config(asset_symbol: str = "SPY") -> RLBaselineConfig:
    data_paths = get_data_split_paths(asset_symbol)
    return RLBaselineConfig(
        name="ppo_markovian_mdd_static",
        algorithm="ppo",
        reward_mode="markovian_mdd",
        asset_symbol=asset_symbol,
        train_data_path=data_paths["train"],
        validation_data_path=data_paths["validation"],
        test_data_path=data_paths["test"],
        lambda_base=0.15,
        reward_kwargs={"lambda_penalty": 0.15},
        retrain_if_exists=True,
    )


def main() -> None:
    result = run_single_rl_baseline(build_config())
    print(f"[{result['baseline']}]")
    for metric_name, metric_value in result["metrics"].items():
        print(f"  {metric_name}: {metric_value:.6f}")


if __name__ == "__main__":
    main()
