import os
import json
import asyncio
from typing import List, Dict, Optional
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import httpx
import edge_tts

app = FastAPI(title="Voice Character Bot")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
CHARACTERS_DIR = BASE_DIR / "characters"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

DEFAULT_CHARACTER = {
    "name": "Luna",
    "voice": "en-US-AvaNeural",
    "system_prompt": "You are a lively, engaging conversational partner. Keep responses concise, spoken, and natural."
}

def load_character(name: str = "luna") -> Dict:
    char_file = CHARACTERS_DIR / f"{name}.json"
    if char_file.exists():
        with open(char_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_CHARACTER

@app.get("/", response_class=HTMLResponse)
async def index():
    html_file = STATIC_DIR / "index.html"
    if html_file.exists():
        return html_file.read_text(encoding="utf-8")
    return "<h1>Voice Character Bot is running. Static files missing.</h1>"

@app.get("/api/characters")
async def list_characters():
    characters = []
    for file in CHARACTERS_DIR.glob("*.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                characters.append({
                    "id": file.stem,
                    "name": data.get("name", file.stem),
                    "voice": data.get("voice", "en-US-AvaNeural")
                })
        except Exception:
            continue
    return characters

@app.post("/api/characters/upload")
async def upload_character(file: UploadFile = File(...)):
    if not file.filename.endswith(".json"):
        return JSONResponse(status_code=400, content={"error": "Only JSON files are supported."})
    contents = await file.read()
    try:
        data = json.loads(contents.decode("utf-8"))
        save_path = CHARACTERS_DIR / file.filename
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return {"status": "ok", "filename": file.filename, "name": data.get("name")}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

async def generate_llm_response(messages: List[Dict], api_key: str, model: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.8,
        "max_tokens": 150
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(GROQ_API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

async def synthesize_audio_stream(text: str, voice: str):
    communicate = edge_tts.Communicate(text, voice)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    current_character = load_character("luna")
    voice = current_character.get("voice", "en-US-AvaNeural")
    system_prompt = current_character.get("system_prompt", DEFAULT_CHARACTER["system_prompt"])

    history: List[Dict] = [
        {"role": "system", "content": system_prompt}
    ]

    try:
        while True:
            raw_data = await websocket.receive_text()
            payload = json.loads(raw_data)
            msg_type = payload.get("type")

            if msg_type == "set_character":
                char_id = payload.get("id")
                current_character = load_character(char_id)
                voice = current_character.get("voice", "en-US-AvaNeural")
                system_prompt = current_character.get("system_prompt", DEFAULT_CHARACTER["system_prompt"])
                history = [{"role": "system", "content": system_prompt}]
                await websocket.send_json({"type": "info", "message": f"Switched to {current_character.get('name')}"})

            elif msg_type == "chat":
                user_text = payload.get("text", "").strip()
                if not user_text:
                    continue

                history.append({"role": "user", "content": user_text})

                api_key = payload.get("api_key") or GROQ_API_KEY
                if not api_key:
                    await websocket.send_json({"type": "error", "message": "No API key provided."})
                    continue

                try:
                    reply_text = await generate_llm_response(history, api_key, DEFAULT_MODEL)
                    history.append({"role": "assistant", "content": reply_text})

                    # Send text response
                    await websocket.send_json({"type": "text", "text": reply_text})

                    # Stream binary audio chunks
                    async for audio_chunk in synthesize_audio_stream(reply_text, voice):
                        await websocket.send_bytes(audio_chunk)

                    await websocket.send_json({"type": "audio_end"})

                except Exception as e:
                    await websocket.send_json({"type": "error", "message": f"Inference/TTS error: {str(e)}"})

    except WebSocketDisconnect:
        pass
