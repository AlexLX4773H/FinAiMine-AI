"""
FinAiMine-AI — LangChain Agent Configuration
Initializes ChatOllama with tools, master system prompt, and context injection.
"""

import logging
import re
from typing import Optional, Dict, Any, List
from config.settings import OLLAMA_URL, OLLAMA_MODEL
from prompts import SYSTEM_PROMPT, STOCK_CONTEXT_PROMPT
from agent.tools import ALL_TOOLS, query_stock_prices, get_predictions, run_technical_analysis, compare_stocks, get_fundamentals, search_news

logger = logging.getLogger(__name__)

_agent_executor = None


def get_llm():
    """
    Returns ChatOllama LLM instance.
    """
    try:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_URL,
            temperature=0.3,
        )
    except Exception as e:
        logger.warning("Error initializing ChatOllama: %s", e)
        return None


def get_agent_executor():
    """
    Constructs and caches LangChain agent executor.
    """
    global _agent_executor
    if _agent_executor is not None:
        return _agent_executor

    llm = get_llm()
    if llm is None:
        return None

    try:
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain.agents import create_tool_calling_agent, AgentExecutor

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)
        _agent_executor = AgentExecutor(
            agent=agent,
            tools=ALL_TOOLS,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
        )
        return _agent_executor
    except Exception as e:
        logger.warning("Could not create tool-calling agent executor: %s", e)
        return None


def run_agent_fallback(user_message: str, ticker_context: Optional[str] = None) -> str:
    """
    Rule-based intelligent tool runner fallback if Ollama daemon is offline or warming up.
    Extracts tickers and runs appropriate tools directly to return accurate answers.
    """
    msg = user_message.upper()
    extracted_tickers = re.findall(r"\b[A-Z]{2,5}\b", msg)
    # Filter out common false-positive words
    stop_words = {"WHAT", "HOW", "WHY", "PRICE", "STOCK", "STOCKS", "TODAY", "TELL", "PREDICT", "NEWS", "ABOUT", "SHOW", "FIND", "WITH", "GIVE", "CHAT", "BUY", "SELL", "WILL", "DOES", "THE"}
    target_tickers = [t for t in extracted_tickers if t not in stop_words]

    ticker = ticker_context.upper() if ticker_context else (target_tickers[0] if target_tickers else "AAPL")

    results = []
    if any(w in msg for w in ["COMPARE", "VS", "VERSUS"]):
        compare_syms = ",".join(target_tickers[:4]) if len(target_tickers) > 1 else f"{ticker},MSFT"
        results.append(f"### Stock Comparison\n{compare_stocks.invoke({'tickers': compare_syms})}")
    elif any(w in msg for w in ["PREDICT", "FORECAST", "FUTURE", "TARGET"]):
        results.append(f"### ML Price Prediction for {ticker}\n{get_predictions.invoke({'ticker': ticker})}")
    elif any(w in msg for w in ["RSI", "MACD", "TECHNICAL", "INDICATOR", "CHART", "AVERAGE", "SMA"]):
        results.append(f"### Technical Analysis for {ticker}\n{run_technical_analysis.invoke({'ticker': ticker})}")
    elif any(w in msg for w in ["NEWS", "HEADLINE", "SENTIMENT"]):
        results.append(f"### News & Sentiment for {ticker}\n{search_news.invoke({'query': '', 'ticker': ticker})}")
    elif any(w in msg for w in ["FUNDAMENTAL", "PE", "P/E", "REVENUE", "EARNINGS", "PROFIT"]):
        results.append(f"### Fundamentals for {ticker}\n{get_fundamentals.invoke({'ticker': ticker})}")
    else:
        results.append(f"### Recent Price & Trends for {ticker}\n{query_stock_prices.invoke({'ticker': ticker, 'days': 5})}")
        results.append(f"### ML Prediction\n{get_predictions.invoke({'ticker': ticker})}")

    header = f"**Fin AI Mine Analysis for {ticker}:**\n\n"
    content = "\n\n".join(results)
    note = (
        "\n\n---\n"
        "*Note: Answered via Medallion Delta Lake & market tools. "
        "For multi-turn conversational reasoning, make sure Ollama is active (`ollama serve`).*"
    )
    return header + content + note


async def execute_chat(message: str, ticker_context: Optional[str] = None, stock_snapshot: Optional[dict] = None) -> str:
    """
    Executes a chat query with optional per-stock context injection.
    """
    executor = get_agent_executor()
    context_prefix = ""
    if ticker_context and stock_snapshot:
        context_prefix = STOCK_CONTEXT_PROMPT.format(
            ticker=ticker_context,
            company_name=stock_snapshot.get("company_name", ticker_context),
            latest_close=stock_snapshot.get("latest_close", "N/A"),
            change_pct=stock_snapshot.get("change_pct", "0.0"),
            week_52_low=stock_snapshot.get("week_52_low", "N/A"),
            week_52_high=stock_snapshot.get("week_52_high", "N/A"),
            sentiment_score=stock_snapshot.get("sentiment_score", "0.0"),
            predicted_close=stock_snapshot.get("predicted_close", "N/A"),
            confidence=stock_snapshot.get("confidence", "75"),
        ) + "\n\nUser Question: "

    prompt_input = f"{context_prefix}{message}" if context_prefix else message

    if executor:
        try:
            res = await executor.ainvoke({"input": prompt_input})
            return res.get("output", "No response generated.")
        except Exception as e:
            logger.warning("Agent execution with Ollama failed (%s). Using tool runner fallback.", e)

    return run_agent_fallback(message, ticker_context)
