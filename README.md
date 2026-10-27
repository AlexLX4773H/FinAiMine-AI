# FinAiMine-AI

> LangChain Agent + FastAPI Backend — AI-powered stock analysis

Part of the [Fin AI Mine](../plan.md) project.

## Tech Stack
- FastAPI + Uvicorn
- LangChain + LangChain-Ollama
- Ollama (Qwen 3 8B) — local LLM
- VADER / FinBERT for sentiment

## Getting Started
```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
# Start Ollama: ollama serve
uvicorn main:app --reload --port 8000
```
