from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_TABLES_DIR = RESULTS_DIR / "tables"
RESULTS_FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_MODELS_DIR = RESULTS_DIR / "models"

SUPPORTED_ASSETS = ("SPY", "QQQ", "DIA", "IWM")
DEFAULT_ASSET = "SPY"
VIX_SYMBOL = "^VIX"
SPLIT_ORDER = ("train", "validation", "test")

ASSET_LABELS = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "DIA": "DIA",
    "IWM": "IWM",
}


def normalize_asset_symbol(asset_symbol: str) -> str:
    normalized = asset_symbol.strip().upper()
    if normalized not in SUPPORTED_ASSETS:
        raise ValueError(f"Unsupported asset '{asset_symbol}'. Expected one of {SUPPORTED_ASSETS}.")
    return normalized


def asset_slug(asset_symbol: str) -> str:
    return normalize_asset_symbol(asset_symbol).lower()


def raw_split_path(asset_symbol: str, split_name: str, root: Path = RAW_DATA_DIR) -> Path:
    return root / f"{asset_slug(asset_symbol)}_vix_{split_name}.csv"


def processed_split_path(asset_symbol: str, split_name: str, root: Path = PROCESSED_DATA_DIR) -> Path:
    return root / f"{asset_slug(asset_symbol)}_vix_indicators_{split_name}.csv"


def get_processed_split_paths(asset_symbol: str, root: Path = PROCESSED_DATA_DIR) -> dict[str, Path]:
    return {split_name: processed_split_path(asset_symbol, split_name, root=root) for split_name in SPLIT_ORDER}


def get_raw_split_paths(asset_symbol: str, root: Path = RAW_DATA_DIR) -> dict[str, Path]:
    return {split_name: raw_split_path(asset_symbol, split_name, root=root) for split_name in SPLIT_ORDER}


def cross_asset_tables_dir(asset_symbol: str) -> Path:
    return RESULTS_TABLES_DIR / "cross_asset" / asset_slug(asset_symbol)


def cross_asset_figures_dir(asset_symbol: str) -> Path:
    return RESULTS_FIGURES_DIR / "cross_asset" / asset_slug(asset_symbol)


def cross_asset_models_dir(asset_symbol: str) -> Path:
    return RESULTS_MODELS_DIR / "cross_asset" / asset_slug(asset_symbol)
