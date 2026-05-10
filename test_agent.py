"""
test_agent.py
-------------
Tests the agent end-to-end without starting the FastAPI server.
Run from the root folder:
    python test_agent.py
"""

from models import Message
from agent.llm import get_agent_reply

def test(label: str, messages: list[dict]):
    print(f"\n{'='*55}")
    print(f"TEST: {label}")
    print(f"{'='*55}")
    for m in messages:
        print(f"  [{m['role']}]: {m['content']}")
    print()

    msg_objs = [Message(**m) for m in messages]
    result = get_agent_reply(msg_objs)

    print(f"REPLY: {result.reply}")
    print(f"RECOMMENDATIONS ({len(result.recommendations)}):")
    for r in result.recommendations:
        print(f"  - {r.name} | {r.test_type} | {r.url}")
    print(f"END OF CONVERSATION: {result.end_of_conversation}")


# ── Test 1: Vague query — should ask a clarifying question ───────────────────
test("Vague query (should clarify, not recommend)", [
    {"role": "user", "content": "I need an assessment"}
])

# ── Test 2: Clear query — should recommend ───────────────────────────────────
test("Clear Java developer query (should recommend)", [
    {"role": "user", "content": "I am hiring a mid-level Java developer who also works with stakeholders"}
])

# ── Test 3: Multi-turn — refine recommendations ──────────────────────────────
test("Multi-turn refinement (should update shortlist)", [
    {"role": "user",      "content": "Hiring a mid-level Java developer"},
    {"role": "assistant", "content": '{"reply": "Here are some Java assessments.", "recommendations": [{"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"}], "end_of_conversation": false}'},
    {"role": "user",      "content": "Actually add a personality test too"}
])

# ── Test 4: Off-topic — should refuse ────────────────────────────────────────
test("Off-topic question (should refuse)", [
    {"role": "user", "content": "Can you help me write a job description for a Java developer?"}
])