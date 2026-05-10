"""
models.py
---------
Defines the exact request and response shapes for POST /chat.
The schema is non-negotiable — deviating breaks the automated evaluator.
"""

from pydantic import BaseModel


# ── A single message in the conversation history ─────────────────────────────
class Message(BaseModel):
    role: str       # "user" or "assistant"
    content: str


# ── What the client sends to POST /chat ──────────────────────────────────────
class ChatRequest(BaseModel):
    messages: list[Message]   # full conversation history, oldest first


# ── A single assessment in the recommendations list ──────────────────────────
class Recommendation(BaseModel):
    name:      str   # e.g. "Java 8 (New)"
    url:       str   # must come from scraped catalog — never made up
    test_type: str   # e.g. "K", "P", "A", "S"


# ── What the server sends back from POST /chat ────────────────────────────────
class ChatResponse(BaseModel):
    reply:               str                    # agent's conversational reply
    recommendations:     list[Recommendation]   # [] when clarifying, 1-10 when recommending
    end_of_conversation: bool                   # True only when task is complete