from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def compute_performance_metrics(
    portfolio_value: pd.Series,
    daily_returns: pd.Series,
    turnover: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    clean_returns = daily_returns.fillna(0.0)
    realized_returns = clean_returns.iloc[1:]
    periods = max(len(realized_returns), 1)

    total_return = float((portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1.0)
    annualized_return = float(
        (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (periods_per_year / periods) - 1.0
    )

    return_std = float(realized_returns.std(ddof=0))
    sharpe_ratio = (
        0.0
        if return_std <= 0
        else float(np.sqrt(periods_per_year) * realized_returns.mean() / return_std)
    )

    downside_returns = realized_returns[realized_returns < 0]
    downside_std = float(downside_returns.std(ddof=0)) if len(downside_returns) > 0 else 0.0
    sortino_ratio = (
        0.0
        if downside_std <= 0
        else float(np.sqrt(periods_per_year) * realized_returns.mean() / downside_std)
    )

    running_peak = portfolio_value.cummax()
    drawdown = (portfolio_value / running_peak) - 1.0
    max_drawdown = float(drawdown.min())
    calmar_ratio = 0.0 if max_drawdown == 0 else float(annualized_return / abs(max_drawdown))

    return {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Calmar Ratio": calmar_ratio,
        "MDD": max_drawdown,
        "Turnover": float(turnover.sum()),
    }
