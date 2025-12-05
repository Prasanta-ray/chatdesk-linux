# server.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional

from llm_backend import ChatSession, LLMClient

app = FastAPI()

# Simple in-memory single-session (for personal use)
chat_session = ChatSession()
llm_client: Optional[LLMClient] = None

templates = Jinja2Templates(directory="static")
app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 256
    temperature: float = 0.7
    system_prompt: Optional[str] = None
    model_path: Optional[str] = None  # allow changing model on the fly


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest):
    global llm_client, chat_session

    if payload.system_prompt is not None:
        chat_session.system_prompt = payload.system_prompt

    if payload.model_path:
        try:
            llm_client = LLMClient(model_path=payload.model_path)
        except Exception as e:
            return JSONResponse({"error": f"Error loading model: {e}"}, status_code=500)

    if llm_client is None:
        return JSONResponse({"error": "Model not loaded. Provide model_path once."}, status_code=400)

    chat_session.add_user_message(payload.message)

    try:
        reply = llm_client.generate(
            chat_session,
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
        )
        chat_session.add_assistant_message(reply)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    return {"reply": reply}


@app.post("/api/clear")
async def clear_chat():
    global chat_session
    chat_session = ChatSession()
    return {"status": "cleared"}
