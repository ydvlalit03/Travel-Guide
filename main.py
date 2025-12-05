# backend/main.py
from typing import Optional, Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .chat_chain import chat_once


app = FastAPI(
    title="Travel Guide Chatbot (Gemini Flash)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    message: str
    city: Optional[str] = None
    mode: Literal["chat", "day_plan", "multi_day"] = "chat"
    use_web: Optional[bool] = True
    use_weather: Optional[bool] = True
    use_events: Optional[bool] = True


class ChatResponse(BaseModel):
    reply: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    reply = await chat_once(
        session_id=payload.session_id,
        city=payload.city,
        user_message=payload.message,
        mode=payload.mode,
        use_web=payload.use_web,
        use_weather=payload.use_weather,
        use_events=payload.use_events,
    )
    return ChatResponse(reply=reply)
