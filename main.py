"""
FinAiMine-AI — FastAPI Application Entry Point
LangChain Agent + FastAPI Backend for Stock Market Analysis & ML Predictions
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config.settings import HOST, PORT
from api.routes_chat import router as chat_router
from api.routes_stocks import router as stocks_router
from api.routes_dashboard import router as dashboard_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FinAiMine-AI")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FinAiMine-AI service is starting up on %s:%d...", HOST, PORT)
    yield
    logger.info("FinAiMine-AI service is shutting down...")


app = FastAPI(
    title="Fin AI Mine API",
    description="AI-powered Stock Market Analysis & Prediction System (LangChain, PySpark, Delta Lake, Ollama)",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(chat_router)
app.include_router(stocks_router)
app.include_router(dashboard_router)


@app.get("/")
def root():
    return {
        "service": "FinAiMine-AI",
        "status": "online",
        "docs_url": "/docs",
        "version": "1.0.0",
        "description": "Stock Market Analysis & Prediction Agent backend",
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "FinAiMine-AI",
        "database": "Delta Lake Medallion Storage Ready",
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
