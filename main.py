"""
main.py
-------
FastAPI app with two endpoints:
  GET  /health  → readiness check
  POST /chat    → stateless conversational agent

Run locally:
  uvicorn main:app --reload --port 8000

Test health:
  curl http://localhost:8000/health

Test chat:
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "I need to hire a Java developer"}]}'
"""

from fastapi import FastAPI, HTTPException
from models import ChatRequest, ChatResponse
from agent.llm import get_agent_reply

app = FastAPI(title="SHL Assessment Recommender")


# ── GET /health ───────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Readiness check — evaluator calls this first, allows 2 min warm-up."""
    return {"status": "ok"}


# ── POST /chat ────────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main endpoint. Receives full conversation history, returns next agent reply.
    Stateless — no session stored on the server.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages list cannot be empty")

    # All the logic lives in agent/llm.py
    response = get_agent_reply(request.messages)
    return response