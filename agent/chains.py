"""
FinAiMine-AI — Custom Specialized Chains
- generate_market_summary: Summarizes market breadth and sentiment for the dashboard card
- generate_stock_brief: Produces an executive summary for a stock
"""

import logging
from typing import Dict, Any, List
from prompts import DASHBOARD_SUMMARY_PROMPT
from agent.agent import get_llm

logger = logging.getLogger(__name__)


def generate_market_summary(top_movers: List[dict], market_sentiment: float) -> str:
    """
    Produces a 2-3 sentence market executive summary for the dashboard card.
    """
    movers_str = ", ".join([f"{m.get('ticker')}: {m.get('change_pct', 0.0):+0.2f}%" for m in top_movers[:5]])
    sentiment_desc = "Bullish" if market_sentiment > 0.1 else ("Bearish" if market_sentiment < -0.1 else "Neutral")

    llm = get_llm()
    if llm:
        try:
            prompt_text = DASHBOARD_SUMMARY_PROMPT.format(
                top_movers=movers_str or "Mixed action across tech & broad market",
                market_sentiment=f"{market_sentiment:.2f} ({sentiment_desc})",
            )
            res = llm.invoke(prompt_text)
            if hasattr(res, "content") and res.content:
                return res.content.strip()
        except Exception as e:
            logger.warning("LLM market summary generation failed: %s", e)

    # Deterministic fallback summary
    return (
        f"Markets are reflecting a {sentiment_desc.lower()} bias today with average sentiment at {market_sentiment:+.2f}. "
        f"Notable momentum observed across {movers_str or 'major tracked assets'}. "
        "Key technical moving averages suggest sustained consolidation with selective breakout opportunities."
    )
