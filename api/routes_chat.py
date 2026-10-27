"""
FinAiMine-AI — Chat API Endpoints & Streaming WebSocket
- POST /api/chat
- POST /api/stocks/{ticker}/chat
- WebSocket /ws/chat
"""

import asyncio
import datetime
import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from agent.agent import execute_chat

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Chat"])


class ChatRequest(BaseModel):
    message: str
    ticker_context: Optional[str] = None
    stock_snapshot: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    response: str
    ticker_context: Optional[str] = None
    timestamp: str


@router.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Global chat endpoint communicating with the LangChain agent.
    """
    ans = await execute_chat(req.message, req.ticker_context, req.stock_snapshot)
    return ChatResponse(
        response=ans,
        ticker_context=req.ticker_context,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


@router.post("/api/stocks/{ticker}/chat", response_model=ChatResponse)
async def stock_chat_endpoint(ticker: str, req: ChatRequest):
    """
    Per-stock contextual chat pre-injected with the stock's current snapshot.
    """
    ans = await execute_chat(req.message, ticker.upper(), req.stock_snapshot)
    return ChatResponse(
        response=ans,
        ticker_context=ticker.upper(),
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


@router.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    """
    Real-time WebSocket streaming chat.
    Receives JSON messages: {"message": "...", "ticker_context": "..."}
    Streams back response chunks and a final {"done": true}.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")
            ticker = data.get("ticker_context")
            snapshot = data.get("stock_snapshot")

            if not message:
                await websocket.send_json({"error": "Empty message"})
                continue

            # Generate response
            response = await execute_chat(message, ticker, snapshot)

            # Stream chunks for natural typing effect
            chunk_size = 20
            words = response.split(" ")
            current_chunk = []
            for word in words:
                current_chunk.append(word)
                if len(current_chunk) >= 5:
                    await websocket.send_json({
                        "chunk": " " + " ".join(current_chunk),
                        "done": False,
                    })
                    current_chunk = []
                    await asyncio.sleep(0.04)

            if current_chunk:
                await websocket.send_json({
                    "chunk": " " + " ".join(current_chunk),
                    "done": False,
                })

            await websocket.send_json({
                "done": True,
                "full_response": response,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        try:
            await websocket.send_json({"error": str(e), "done": True})
        except Exception:
            pass
