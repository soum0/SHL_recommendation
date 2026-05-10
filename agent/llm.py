"""
agent/llm.py
------------
Main agent logic. Called by main.py on every /chat request.

Flow:
  1. Build search query from conversation
  2. Retrieve top-15 assessments from FAISS
  3. Build system prompt with those assessments
  4. Call OpenRouter
  5. Parse JSON → return ChatResponse
"""

import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

from models import ChatResponse, Recommendation, Message
from retrieval.search import search
from agent.prompt import build_system_prompt, build_catalog_context

load_dotenv()

_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
_model = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free")


def build_search_query(messages: list[Message]) -> str:
    """Use last 4 messages as the search query."""
    recent = messages[-4:]
    return " ".join(m.content for m in recent)


def parse_llm_response(raw: str, retrieved: list[dict]) -> ChatResponse:
    """
    Parse LLM output into ChatResponse.

    Handles three cases:
    1. Clean JSON  → parse directly
    2. JSON inside extra text → extract with regex
    3. Pure plain text (no JSON) → use the text as reply, empty recommendations
    """
    # Step 1: strip markdown fences
    cleaned = re.sub(r"```json|```", "", raw).strip()

    # Step 2: try to find a JSON object anywhere in the response
    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)

    if json_match:
        try:
            data = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            # JSON found but malformed — use plain text as reply
            data = {"reply": cleaned, "recommendations": [], "end_of_conversation": False}
    else:
        # No JSON at all — treat entire response as the reply text
        # This handles the case where the model ignores JSON instructions
        data = {"reply": cleaned, "recommendations": [], "end_of_conversation": False}

    # Only allow URLs that exist in our FAISS-retrieved catalog
    valid_urls = {a["url"] for a in retrieved}

    safe_recommendations = []
    for rec in data.get("recommendations", []):
        url = rec.get("url", "")
        if url in valid_urls:
            safe_recommendations.append(
                Recommendation(
                    name=rec.get("name", ""),
                    url=url,
                    test_type=rec.get("test_type", ""),
                )
            )

    return ChatResponse(
        reply=data.get("reply", ""),
        recommendations=safe_recommendations[:10],
        end_of_conversation=data.get("end_of_conversation", False),
    )


def get_agent_reply(messages: list[Message]) -> ChatResponse:
    """Main function called by POST /chat."""

    # 1. Build search query
    query = build_search_query(messages)

    # 2. Retrieve from FAISS
    retrieved = search(query, top_k=15)

    # 3. Build system prompt
    catalog_context = build_catalog_context(retrieved)
    system_prompt = build_system_prompt(catalog_context)

    # 4. Format messages
    openai_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        role = "user" if msg.role == "user" else "assistant"
        openai_messages.append({"role": role, "content": msg.content})

    # 5. Call OpenRouter
    response = _client.chat.completions.create(
        model=_model,
        messages=openai_messages,
        temperature=0.1,      # even lower = more consistent JSON
        max_tokens=1000,
    )

    raw_output = response.choices[0].message.content

    # 6. Parse and return
    return parse_llm_response(raw_output, retrieved)