"""
FinAiMine-AI — Dashboard API Endpoints
- GET /api/dashboard: Aggregated dashboard summary, top movers, sentiment index, AI market brief
"""

import logging
from fastapi import APIRouter
import pandas as pd

from config.settings import DEFAULT_TICKERS, read_delta_table
from api.routes_stocks import get_latest_quote_data
from agent.chains import generate_market_summary

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


@router.get("")
def get_dashboard_summary():
    """
    Returns aggregated dashboard metrics:
    - Watchlist stock cards with quotes
    - Top gainers & top losers
    - Market sentiment index
    - Natural language AI market summary
    """
    quotes = []
    for sym in DEFAULT_TICKERS:
        quotes.append(get_latest_quote_data(sym))

    # Sort gainers and losers
    sorted_stocks = sorted(quotes, key=lambda x: x["change_pct"], reverse=True)
    top_gainers = sorted_stocks[:3]
    top_losers = sorted_stocks[-3:][::-1]

    # Calculate market averages
    avg_return = round(sum(s["change_pct"] for s in quotes) / len(quotes), 2) if quotes else 0.0

    # Fetch sentiment from Delta table if present
    df_sent = read_delta_table("gold_sentiment_agg")
    avg_sent = 0.22
    if not df_sent.empty and "avg_sentiment" in df_sent.columns:
        avg_sent = round(float(df_sent["avg_sentiment"].mean()), 3)

    # Generate AI market summary brief
    summary_text = generate_market_summary(sorted_stocks, avg_sent)

    return {
        "status": "success",
        "market_stats": {
            "tracked_count": len(quotes),
            "market_avg_change_pct": avg_return,
            "market_sentiment_score": avg_sent,
            "sentiment_label": "Bullish" if avg_sent > 0.05 else ("Bearish" if avg_sent < -0.05 else "Neutral"),
        },
        "top_gainers": top_gainers,
        "top_losers": top_losers,
        "watchlist": quotes,
        "ai_summary": summary_text,
    }
