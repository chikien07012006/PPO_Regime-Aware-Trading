from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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

from baselines.common.asset_utils import (
    DEFAULT_ASSET,
    PROCESSED_DATA_DIR,
    cross_asset_figures_dir,
    cross_asset_tables_dir,
    normalize_asset_symbol,
    processed_split_path,
)
from baselines.common.metrics import TRADING_DAYS_PER_YEAR, compute_performance_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST_DATA_PATH = processed_split_path(DEFAULT_ASSET, "test", root=PROCESSED_DATA_DIR)
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_FIGURES_DIR = DEFAULT_RESULTS_DIR / "figures"
DEFAULT_TABLES_DIR = DEFAULT_RESULTS_DIR / "tables"


class WeightStrategy(Protocol):
    def reset(self) -> None: ...

    def target_weight(
        self,
        step: int,
        history: pd.DataFrame,
        current_weight: float,
        current_value: float,
    ) -> float: ...


@dataclass(slots=True)
class BuyAndHoldStrategy:
    initial_weight: float = 1.0

    def reset(self) -> None:
        return None

    def target_weight(
        self,
        step: int,
        history: pd.DataFrame,
        current_weight: float,
        current_value: float,
    ) -> float:
        del history, current_value
        if step == 0:
            return float(np.clip(self.initial_weight, 0.0, 1.0))
        return float(current_weight)


@dataclass(slots=True)
class RiskParitySpyCashStrategy:
    rebalance_frequency: int = 21
    lookback_window: int = 60
    min_history: int = 20
    target_annual_volatility: float = 0.20
    min_weight: float = 0.0
    max_weight: float = 1.0
    annualization_factor: int = TRADING_DAYS_PER_YEAR

    def reset(self) -> None:
        return None

    def target_weight(
        self,
        step: int,
        history: pd.DataFrame,
        current_weight: float,
        current_value: float,
    ) -> float:
        del current_value

        if step != 0 and (step % self.rebalance_frequency) != 0:
            return float(current_weight)

        realized_returns = history["asset_return"].dropna()
        if len(realized_returns) < self.min_history:
            return float(self.max_weight)

        realized_returns = realized_returns.iloc[-self.lookback_window :]
        realized_volatility = float(realized_returns.std(ddof=0) * math.sqrt(self.annualization_factor))
        if realized_volatility <= 0:
            return float(self.max_weight)

        target_weight = self.target_annual_volatility / realized_volatility
        return float(np.clip(target_weight, self.min_weight, self.max_weight))


@dataclass(slots=True)
class CPPIStrategy:
    initial_capital: float
    floor_ratio: float = 0.80
    multiplier: float = 3.0

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")
        if not 0 <= self.floor_ratio <= 1:
            raise ValueError("floor_ratio must be in [0, 1].")
        if self.multiplier < 0:
            raise ValueError("multiplier must be non-negative.")

    @property
    def floor_value(self) -> float:
        return self.floor_ratio * self.initial_capital

    def reset(self) -> None:
        return None

    def target_weight(
        self,
        step: int,
        history: pd.DataFrame,
        current_weight: float,
        current_value: float,
    ) -> float:
        del step, history, current_weight

        if current_value <= 0:
            return 0.0

        cushion = max(0.0, current_value - self.floor_value)
        exposure = self.multiplier * cushion
        target_weight = exposure / current_value
        return float(np.clip(target_weight, 0.0, 1.0))


def load_test_data(data_path: str | Path = DEFAULT_TEST_DATA_PATH) -> pd.DataFrame:
    frame = pd.read_csv(data_path, parse_dates=["date"])
    required_columns = {"date", "close"}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    frame = frame.sort_values("date").reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError("At least two rows are required to run a baseline backtest.")

    frame["asset_return"] = frame["close"].pct_change()
    return frame


def run_weight_backtest(
    data: pd.DataFrame,
    strategy: WeightStrategy,
    initial_capital: float = 10000.0,
    transaction_cost_rate: float = 0.001,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive.")
    if transaction_cost_rate < 0:
        raise ValueError("transaction_cost_rate must be non-negative.")

    strategy.reset()

    dates = pd.DatetimeIndex(data["date"])
    portfolio_value = pd.Series(index=dates, dtype=np.float64)
    daily_returns = pd.Series(index=dates, dtype=np.float64)
    turnover = pd.Series(index=dates, dtype=np.float64)
    weights = pd.Series(index=dates, dtype=np.float64)

    current_value = float(initial_capital)
    current_weight = 0.0

    portfolio_value.iloc[0] = current_value
    daily_returns.iloc[0] = 0.0
    turnover.iloc[0] = 0.0
    weights.iloc[0] = current_weight

    for step in range(len(data) - 1):
        history = data.iloc[: step + 1].copy()
        target_weight = float(np.clip(strategy.target_weight(step, history, current_weight, current_value), 0.0, 1.0))

        current_close = float(data.iloc[step]["close"])
        next_close = float(data.iloc[step + 1]["close"])
        asset_return = (next_close / current_close) - 1.0

        weight_turnover = abs(target_weight - current_weight)
        transaction_cost = current_value * transaction_cost_rate * weight_turnover
        value_after_cost = max(current_value - transaction_cost, 0.0)
        next_value = value_after_cost * (1.0 + (target_weight * asset_return))

        if next_value <= 0:
            drifted_weight = 0.0
        else:
            risky_value = value_after_cost * target_weight * (1.0 + asset_return)
            drifted_weight = risky_value / next_value

        next_date = dates[step + 1]
        portfolio_value.loc[next_date] = next_value
        daily_returns.loc[next_date] = (next_value / current_value) - 1.0 if current_value > 0 else 0.0
        turnover.loc[next_date] = weight_turnover
        weights.loc[next_date] = drifted_weight

        current_value = float(next_value)
        current_weight = float(drifted_weight)

    portfolio_value = portfolio_value.ffill()
    daily_returns = daily_returns.fillna(0.0)
    turnover = turnover.fillna(0.0)
    weights = weights.ffill().fillna(0.0)

    diagnostics = pd.DataFrame(
        {
            "portfolio_value": portfolio_value,
            "daily_returns": daily_returns,
            "turnover": turnover,
            "weight_spy": weights,
            "weight_cash": 1.0 - weights,
        }
    )

    return portfolio_value, daily_returns, turnover, diagnostics


def run_buy_and_hold_baseline(
    data: pd.DataFrame,
    initial_capital: float = 10000.0,
    transaction_cost_rate: float = 0.001,
) -> tuple[pd.Series, pd.Series, dict[str, float], pd.DataFrame]:
    strategy = BuyAndHoldStrategy(initial_weight=1.0)
    portfolio_value, daily_returns, turnover, diagnostics = run_weight_backtest(
        data=data,
        strategy=strategy,
        initial_capital=initial_capital,
        transaction_cost_rate=transaction_cost_rate,
    )
    metrics = compute_performance_metrics(portfolio_value, daily_returns, turnover)
    return portfolio_value, daily_returns, metrics, diagnostics


def run_risk_parity_baseline(
    data: pd.DataFrame,
    initial_capital: float = 10000.0,
    transaction_cost_rate: float = 0.001,
    rebalance_frequency: int = 21,
    lookback_window: int = 60,
    min_history: int = 20,
    target_annual_volatility: float = 0.20,
) -> tuple[pd.Series, pd.Series, dict[str, float], pd.DataFrame]:
    strategy = RiskParitySpyCashStrategy(
        rebalance_frequency=rebalance_frequency,
        lookback_window=lookback_window,
        min_history=min_history,
        target_annual_volatility=target_annual_volatility,
    )
    portfolio_value, daily_returns, turnover, diagnostics = run_weight_backtest(
        data=data,
        strategy=strategy,
        initial_capital=initial_capital,
        transaction_cost_rate=transaction_cost_rate,
    )
    metrics = compute_performance_metrics(portfolio_value, daily_returns, turnover)
    return portfolio_value, daily_returns, metrics, diagnostics


def run_cppi_baseline(
    data: pd.DataFrame,
    initial_capital: float = 10000.0,
    transaction_cost_rate: float = 0.001,
    floor_ratio: float = 0.80,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series, dict[str, float], pd.DataFrame]:
    strategy = CPPIStrategy(
        initial_capital=initial_capital,
        floor_ratio=floor_ratio,
        multiplier=multiplier,
    )
    portfolio_value, daily_returns, turnover, diagnostics = run_weight_backtest(
        data=data,
        strategy=strategy,
        initial_capital=initial_capital,
        transaction_cost_rate=transaction_cost_rate,
    )
    metrics = compute_performance_metrics(portfolio_value, daily_returns, turnover)
    return portfolio_value, daily_returns, metrics, diagnostics


def plot_equity_curves(
    baseline_results: dict[str, dict[str, pd.Series | dict[str, float] | pd.DataFrame]],
    output_path: str | Path,
    *,
    title: str = "Backtest of Financial Baselines: Equity Curve Comparison",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    color_map = {
        "buy_and_hold_spy": "#1f77b4",
        "risk_parity_spy_cash": "#ff7f0e",
        "cppi_spy_cash": "#2ca02c",
    }
    label_map = {
        "buy_and_hold_spy": "Buy & Hold",
        "risk_parity_spy_cash": "Risk-Parity Proxy",
        "cppi_spy_cash": "CPPI",
    }

    for baseline_name, result in baseline_results.items():
        portfolio_value = result["portfolio_value"]
        ax.plot(
            portfolio_value.index,
            portfolio_value.values,
            linewidth=2.3,
            color=color_map.get(baseline_name, None),
            label=label_map.get(baseline_name, baseline_name),
        )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (USD)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=True)
    ax.grid(True, alpha=0.4, linewidth=0.8)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_baseline_results(
    baseline_results: dict[str, dict[str, pd.Series | dict[str, float] | pd.DataFrame]],
    tables_dir: str | Path = DEFAULT_TABLES_DIR,
) -> tuple[Path, Path]:
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    time_series_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float | str]] = []

    for baseline_name, result in baseline_results.items():
        diagnostics = result["diagnostics"].copy()
        diagnostics.insert(0, "baseline", baseline_name)
        diagnostics.insert(1, "date", diagnostics.index)
        time_series_frames.append(diagnostics.reset_index(drop=True))

        metrics = {"baseline": baseline_name}
        metrics.update(result["metrics"])
        metric_rows.append(metrics)

    time_series_path = tables_dir / "financial_baselines_portfolios.csv"
    metrics_path = tables_dir / "financial_baselines_metrics.csv"

    pd.concat(time_series_frames, ignore_index=True).to_csv(time_series_path, index=False)
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)

    json_ready = {
        baseline_name: {
            "metrics": {metric_name: float(metric_value) for metric_name, metric_value in result["metrics"].items()}
        }
        for baseline_name, result in baseline_results.items()
    }
    metrics_json_path = tables_dir / "financial_baselines_metrics.json"
    metrics_json_path.write_text(json.dumps(json_ready, indent=2), encoding="utf-8")

    return time_series_path, metrics_path


def run_all_financial_baselines(
    asset_symbol: str = DEFAULT_ASSET,
    data_path: str | Path = DEFAULT_TEST_DATA_PATH,
    initial_capital: float = 10000.0,
    transaction_cost_rate: float = 0.001,
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
    tables_dir: str | Path = DEFAULT_TABLES_DIR,
) -> dict[str, dict[str, pd.Series | dict[str, float] | pd.DataFrame]]:
    normalized_asset = normalize_asset_symbol(asset_symbol)
    if data_path == DEFAULT_TEST_DATA_PATH and normalized_asset != DEFAULT_ASSET:
        data_path = processed_split_path(normalized_asset, "test", root=PROCESSED_DATA_DIR)
    if figures_dir == DEFAULT_FIGURES_DIR and normalized_asset != DEFAULT_ASSET:
        figures_dir = cross_asset_figures_dir(normalized_asset) / "financial_baselines"
    if tables_dir == DEFAULT_TABLES_DIR and normalized_asset != DEFAULT_ASSET:
        tables_dir = cross_asset_tables_dir(normalized_asset) / "financial_baselines"

    data = load_test_data(data_path)

    buy_and_hold_value, buy_and_hold_returns, buy_and_hold_metrics, buy_and_hold_diagnostics = run_buy_and_hold_baseline(
        data=data,
        initial_capital=initial_capital,
        transaction_cost_rate=transaction_cost_rate,
    )
    risk_parity_value, risk_parity_returns, risk_parity_metrics, risk_parity_diagnostics = run_risk_parity_baseline(
        data=data,
        initial_capital=initial_capital,
        transaction_cost_rate=transaction_cost_rate,
    )
    cppi_value, cppi_returns, cppi_metrics, cppi_diagnostics = run_cppi_baseline(
        data=data,
        initial_capital=initial_capital,
        transaction_cost_rate=transaction_cost_rate,
    )

    baseline_results = {
        "buy_and_hold_spy": {
            "portfolio_value": buy_and_hold_value,
            "daily_returns": buy_and_hold_returns,
            "metrics": buy_and_hold_metrics,
            "diagnostics": buy_and_hold_diagnostics,
        },
        "risk_parity_spy_cash": {
            "portfolio_value": risk_parity_value,
            "daily_returns": risk_parity_returns,
            "metrics": risk_parity_metrics,
            "diagnostics": risk_parity_diagnostics,
        },
        "cppi_spy_cash": {
            "portfolio_value": cppi_value,
            "daily_returns": cppi_returns,
            "metrics": cppi_metrics,
            "diagnostics": cppi_diagnostics,
        },
    }

    plot_equity_curves(
        baseline_results=baseline_results,
        output_path=Path(figures_dir) / "financial_baseline.png",
        title=f"Backtest of Financial Baselines ({normalized_asset}): Equity Curve Comparison",
    )
    save_baseline_results(
        baseline_results=baseline_results,
        tables_dir=tables_dir,
    )

    return baseline_results


def main() -> None:
    baseline_results = run_all_financial_baselines()
    for baseline_name, result in baseline_results.items():
        print(f"[{baseline_name}]")
        for metric_name, metric_value in result["metrics"].items():
            print(f"  {metric_name}: {metric_value:.6f}")


if __name__ == "__main__":
    main()
