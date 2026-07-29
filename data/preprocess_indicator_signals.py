from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.common.asset_utils import (
    DEFAULT_ASSET,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SPLIT_ORDER,
    SUPPORTED_ASSETS,
    normalize_asset_symbol,
    processed_split_path,
    raw_split_path,
)
REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume", "vix_close"]


def load_raw_splits(asset_symbol: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split_name in SPLIT_ORDER:
        file_path = raw_split_path(asset_symbol, split_name)
        if not file_path.exists():
            raise FileNotFoundError(f"Missing raw split file: {file_path}")
        frame = pd.read_csv(file_path, parse_dates=["date"])
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"{file_path} is missing columns: {missing_columns}")
        frame["split"] = split_name
        frames.append(frame[REQUIRED_COLUMNS + ["split"]].copy())

    merged = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    return merged


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    rsi = rsi.mask((avg_gain > 0) & (avg_loss == 0), 100.0)
    return rsi


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    close = enriched["close"]
    vix_close = enriched["vix_close"]

    sma_50 = close.rolling(window=50, min_periods=50).mean()
    sma_20 = close.rolling(window=20, min_periods=20).mean()
    std_20 = close.rolling(window=20, min_periods=20).std(ddof=0)
    vix_mean_252 = vix_close.rolling(window=252, min_periods=252).mean()
    vix_std_252 = vix_close.rolling(window=252, min_periods=252).std(ddof=0)

    enriched["log_return"] = np.log(close / close.shift(1))
    enriched["sma_ratio"] = (close / sma_50) - 1
    enriched["rsi_14"] = compute_rsi(close, window=14)
    enriched["bollinger_band_width"] = ((sma_20 + (2 * std_20)) - (sma_20 - (2 * std_20))) / sma_20
    enriched["vix_zscore_252"] = (vix_close - vix_mean_252) / vix_std_252
    return enriched


def save_processed_splits(asset_symbol: str, frame: pd.DataFrame, output_dir: Path, drop_na: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    indicator_columns = [
        "log_return",
        "sma_ratio",
        "rsi_14",
        "bollinger_band_width",
        "vix_zscore_252",
    ]

    for split_name in SPLIT_ORDER:
        split_frame = frame.loc[frame["split"] == split_name].copy().reset_index(drop=True)
        if drop_na:
            split_frame = split_frame.dropna(subset=indicator_columns).reset_index(drop=True)
        output_path = processed_split_path(asset_symbol, split_name, root=output_dir)
        split_frame.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load split raw asset/VIX data, compute daily indicators, and save processed splits."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory where processed CSV files will be written.",
    )
    parser.add_argument(
        "--drop-na",
        action="store_true",
        help="Drop rows that do not yet have all indicator values because of rolling-window warmup.",
    )
    parser.add_argument(
        "--assets",
        nargs="+",
        default=[DEFAULT_ASSET, *[asset for asset in SUPPORTED_ASSETS if asset != DEFAULT_ASSET]],
        help=f"Asset symbols to preprocess. Supported: {', '.join(SUPPORTED_ASSETS)}.",
    )
    args = parser.parse_args()

    assets = [normalize_asset_symbol(asset_symbol) for asset_symbol in args.assets]
    for asset_symbol in assets:
        raw = load_raw_splits(asset_symbol)
        processed = add_indicators(raw)
        save_processed_splits(asset_symbol, processed, args.output_dir, drop_na=args.drop_na)

        for split_name in SPLIT_ORDER:
            split_frame = processed.loc[processed["split"] == split_name]
            if args.drop_na:
                split_frame = split_frame.dropna(
                    subset=[
                        "log_return",
                        "sma_ratio",
                        "rsi_14",
                        "bollinger_band_width",
                        "vix_zscore_252",
                    ]
                )
            print(
                f"{asset_symbol} {split_name}: {len(split_frame)} rows, "
                f"{split_frame['date'].min().date()} -> {split_frame['date'].max().date()}"
            )


if __name__ == "__main__":
    main()
