"""
FinAiMine-AI — Settings & Configuration
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Server Config
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# Ollama / LLM Config
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Shared Delta Warehouse Path
warehouse_env = os.getenv("DELTA_WAREHOUSE_PATH", "../FinAiMine-Data/delta_warehouse")
if os.path.isabs(warehouse_env):
    DELTA_WAREHOUSE_PATH = Path(warehouse_env)
else:
    DELTA_WAREHOUSE_PATH = (BASE_DIR / warehouse_env).resolve()

# Table Paths
TABLE_PATHS = {
    # Bronze
    "bronze_prices": str(DELTA_WAREHOUSE_PATH / "bronze" / "prices"),
    "bronze_news": str(DELTA_WAREHOUSE_PATH / "bronze" / "news"),
    "bronze_fundamentals": str(DELTA_WAREHOUSE_PATH / "bronze" / "fundamentals"),
    "bronze_mutual_funds": str(DELTA_WAREHOUSE_PATH / "bronze" / "mutual_funds"),
    # Silver
    "silver_prices": str(DELTA_WAREHOUSE_PATH / "silver" / "prices"),
    "silver_news": str(DELTA_WAREHOUSE_PATH / "silver" / "news"),
    "silver_fundamentals": str(DELTA_WAREHOUSE_PATH / "silver" / "fundamentals"),
    "silver_mutual_funds": str(DELTA_WAREHOUSE_PATH / "silver" / "mutual_funds"),
    # Gold
    "gold_stock_features": str(DELTA_WAREHOUSE_PATH / "gold" / "stock_features"),
    "gold_sentiment_agg": str(DELTA_WAREHOUSE_PATH / "gold" / "sentiment_agg"),
    "gold_predictions": str(DELTA_WAREHOUSE_PATH / "gold" / "predictions"),
    "gold_portfolio_summary": str(DELTA_WAREHOUSE_PATH / "gold" / "portfolio_summary"),
}

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "SPY"]


def read_delta_table(table_name_or_path: str) -> pd.DataFrame:
    """
    Safely reads a Delta table from the shared warehouse.
    Supports either table name key (e.g. 'gold_stock_features') or raw path.
    """
    path_str = TABLE_PATHS.get(table_name_or_path, table_name_or_path)
    p = Path(path_str)
    if not p.exists() or not any(p.iterdir()):
        return pd.DataFrame()

    # Try deltalake native reader
    try:
        from deltalake import DeltaTable
        dt = DeltaTable(str(p))
        return dt.to_pandas()
    except Exception:
        pass

    # Try parquet files inside directory
    parquets = list(p.glob("**/*.parquet"))
    if parquets:
        try:
            dfs = [pd.read_parquet(f) for f in parquets if not f.name.startswith(".")]
            if dfs:
                return pd.concat(dfs, ignore_index=True)
        except Exception:
            pass

    # Try csv files inside directory
    csvs = list(p.glob("**/*.csv"))
    if csvs:
        try:
            dfs = [pd.read_csv(f) for f in csvs if not f.name.startswith(".")]
            if dfs:
                return pd.concat(dfs, ignore_index=True)
        except Exception:
            pass

    return pd.DataFrame()
