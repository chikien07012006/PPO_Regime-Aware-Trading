from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.common.asset_utils import (
    DEFAULT_ASSET,
    RAW_DATA_DIR,
    SPLIT_ORDER,
    SUPPORTED_ASSETS,
    VIX_SYMBOL,
    normalize_asset_symbol,
    raw_split_path,
)
START_DATE = "2010-01-01"
END_DATE = "2023-01-01"
SPLITS = {
    "train": ("2010-01-01", "2017-12-31"),
    "validation": ("2018-01-01", "2019-12-31"),
    "test": ("2020-01-01", "2022-12-31"),
}
YF_CACHE_DIR = Path(tempfile.gettempdir()) / "markovian_mdd_rrc_trading_yfinance"
YF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
if hasattr(yf, "set_tz_cache_location"):
    yf.set_tz_cache_location(str(YF_CACHE_DIR))


def _download_daily_history(symbol: str, start: str, end: str) -> pd.DataFrame:
    frame = yf.download(
        tickers=symbol,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )
    if frame.empty:
        raise ValueError(f"No data returned for symbol '{symbol}'.")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    frame = frame.reset_index()
    frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    return frame


def load_and_align_data(asset_symbol: str, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    normalized_asset = normalize_asset_symbol(asset_symbol)
    asset = _download_daily_history(normalized_asset, start, end)[
        ["date", "open", "high", "low", "close", "volume"]
    ].copy()
    vix = _download_daily_history(VIX_SYMBOL, start, end)[["date", "close"]].copy()
    vix = vix.rename(columns={"close": "vix_close"})

    merged = asset.merge(vix, on="date", how="inner").sort_values("date").reset_index(drop=True)
    merged["volume"] = merged["volume"].astype("int64")
    merged = merged[["date", "open", "high", "low", "close", "volume", "vix_close"]]
    return merged


def split_by_date(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    splits: dict[str, pd.DataFrame] = {}
    for split_name, (start_date, end_date) in SPLITS.items():
        mask = frame["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
        split_frame = frame.loc[mask].copy().reset_index(drop=True)
        if split_frame.empty:
            raise ValueError(f"Split '{split_name}' is empty for date range {start_date} -> {end_date}.")
        splits[split_name] = split_frame
    return splits


def save_splits(asset_symbol: str, splits: dict[str, pd.DataFrame], output_dir: Path = RAW_DATA_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, frame in splits.items():
        output_path = raw_split_path(asset_symbol, split_name, root=output_dir)
        frame.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download selected ETF assets and VIX daily history, align the series, and save raw splits."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Directory where split raw CSV files will be written.",
    )
    parser.add_argument(
        "--assets",
        nargs="+",
        default=[DEFAULT_ASSET, *[asset for asset in SUPPORTED_ASSETS if asset != DEFAULT_ASSET]],
        help=f"Asset symbols to download. Supported: {', '.join(SUPPORTED_ASSETS)}.",
    )
    args = parser.parse_args()

    assets = [normalize_asset_symbol(asset_symbol) for asset_symbol in args.assets]
    for asset_symbol in assets:
        merged = load_and_align_data(asset_symbol)
        split_frames = split_by_date(merged)
        save_splits(asset_symbol, split_frames, args.output_dir)

        for split_name in SPLIT_ORDER:
            frame = split_frames[split_name]
            print(
                f"{asset_symbol} {split_name}: {len(frame)} rows, "
                f"{frame['date'].min().date()} -> {frame['date'].max().date()}"
            )


if __name__ == "__main__":
    main()
