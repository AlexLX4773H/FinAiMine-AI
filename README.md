# FinAiMine-AI

> **LangChain Agent + FastAPI Backend — AI-powered Stock Market Analysis & Prediction**

Part of the [Fin AI Mine](../plan.md) system. Provides REST and WebSocket endpoints for AI financial chat, stock data analytics, ML price predictions, and dashboard summaries.

---

## ⚡ Features

- **LangChain Tool-Calling Agent:** Integrates local Ollama (`qwen3:8b`) with fallbacks.
- **7 Financial Agent Tools:**
  1. `query_stock_prices`: Delta Lake `gold_stock_features` & yfinance fallback
  2. `query_news_sentiment`: `gold_sentiment_agg` & live headline scoring
  3. `get_fundamentals`: `silver_fundamentals` (P/E, Margins, Market Cap)
  4. `get_predictions`: `gold_predictions` ML forecast with confidence scores
  5. `run_technical_analysis`: SMA, EMA, RSI, MACD, Bollinger Bands
  6. `search_news`: Full-text news search with sentiment tags
  7. `compare_stocks`: Side-by-side multi-ticker matrix
- **FastAPI Endpoints:**
  - `POST /api/chat`: Global AI financial copilot
  - `POST /api/stocks/{ticker}/chat`: Context-aware per-stock chat
  - `WebSocket /ws/chat`: Real-time streaming chat responses
  - `GET /api/stocks`: List tracked stocks with live metrics
  - `GET /api/stocks/{ticker}`: Full technical & fundamental snapshot
  - `GET /api/stocks/{ticker}/predict`: Next-day ML prediction
  - `GET /api/stocks/{ticker}/history`: OHLCV series for candlestick charts
  - `GET /api/dashboard`: Aggregated market stats & AI market summary

---

## 🚀 Getting Started

### 1. Setup Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate # Linux/Mac
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
```

### 3. (Optional) Run Ollama
```bash
ollama pull qwen3:8b
ollama serve
```
*(If Ollama is not active, FinAiMine-AI uses an intelligent tool runner fallback so all endpoints and chat features work seamlessly).*

### 4. Run the API Server
```bash
uvicorn main:app --reload --port 8000
```
Visit API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
