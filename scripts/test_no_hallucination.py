"""Headless guardrail tests. Hits backend in-process via FastAPI TestClient.
No live LLM calls — exercises rule-based intents + deterministic negotiation
to confirm hallucination-resistant invariants hold.

Run: python -m scripts.test_no_hallucination
"""
import sys
from pathlib import Path

# Windows consoles default to cp1252, which can't encode ✓/✗/≤/₹ — force UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from backend.main import app, CATALOG
from backend.negotiation import negotiate

client = TestClient(app)

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    mark = "✓" if cond else "✗"
    print(f"  {mark} {name}" + (f"  [{detail}]" if detail else ""))
    if cond:
        PASS += 1
    else:
        FAIL += 1


def post_chat(session: str, msg: str) -> dict:
    r = client.post("/chat", json={"session_id": session, "message": msg})
    r.raise_for_status()
    return r.json()


print("\n=== 1. Off-topic refusal (no LLM) ===")
for q in [
    "Who is the prime minister of India?",
    "How are you?",
    "Tell me a joke",
    "What is 2+2?",
    "kaise ho",
    "what's up",
]:
    r = post_chat(f"off_{q[:8]}", q)
    check(
        f"OFF_TOPIC -> '{q[:30]}'",
        r["intent"] == "OFF_TOPIC" and "products" in r["reply"].lower(),
        r["reply"][:60],
    )


print("\n=== 2. Greeting ===")
r = post_chat("g1", "hi")
check("GREETING for 'hi'", r["intent"] == "GREETING")
r = post_chat("g2", "hello")
check("GREETING for 'hello'", r["intent"] == "GREETING")


print("\n=== 3. Farewell + chat end ===")
r = post_chat("f1", "thanks")
check("FAREWELL for 'thanks'", r["intent"] == "FAREWELL")
check("end_chat=true", r["end_chat"] is True)


print("\n=== 4. Request human (no focal product) ===")
r = post_chat("rh1", "I want to talk to the seller")
check("REQUEST_HUMAN intent", r["intent"] == "REQUEST_HUMAN")
check("notify_seller=true", r["notify_seller"] is True)


print("\n=== 5. Negotiation determinism (no LLM, pure engine) ===")
products = CATALOG.all()
for p in products:
    listed = int(p.get("priceInt") or 0)
    floor = int(p.get("minPrice") or 0)
    if listed == 0:
        continue

    # Buyer offers below floor → counter, never accept.
    out = negotiate(listed, floor, floor - 1, 1)
    check(
        f"[{p.get('title','')[:25]}] reject below floor",
        out.decision != "accept" and (out.counter_price or 0) >= floor,
        f"listed={listed} floor={floor} offer={floor-1} -> {out.decision}/{out.counter_price}",
    )

    # Buyer offers exactly floor → accept at floor.
    out = negotiate(listed, floor, floor, 1)
    check(
        f"[{p.get('title','')[:25]}] accept at floor",
        out.decision == "accept" and out.counter_price == floor,
    )

    # Final round below floor → counter at floor exactly.
    out = negotiate(listed, floor, floor - 50, 3)
    check(
        f"[{p.get('title','')[:25]}] final = floor",
        out.decision == "counter" and out.counter_price == floor,
    )


print("\n=== 6. /products and /sellers contracts ===")
r = client.get("/products")
data = r.json()["data"]
check("all 9 products exposed", len(data) == 9, f"got {len(data)}")
check("each product has post_id", all(p.get("post_id") for p in data))
check("each product has price", all(p.get("price") is not None for p in data))

r = client.get("/sellers")
sellers = r.json()["data"]
check("at least 1 seller", len(sellers) >= 1)
for s in sellers:
    check(
        f"seller {s['name'][:18]} has phone or wa",
        bool(s.get("phone") or s.get("whatsapp")),
    )


print("\n=== 7. Catalog integrity (every priceInt > 0, minPrice ≤ priceInt) ===")
for p in products:
    pi = int(p.get("priceInt") or 0)
    mp = int(p.get("minPrice") or 0)
    check(
        f"[{p.get('title','')[:25]}] minPrice ≤ priceInt > 0",
        pi > 0 and 0 < mp <= pi,
        f"price={pi} min={mp}",
    )


print("\n=== 8. All LLMs down → template fallback (never errors) ===")
from backend import agent as agent_mod

# Monkey-patch chain builder to simulate complete LLM outage.
_orig = agent_mod._build_model_chain
agent_mod._build_model_chain = lambda: None
try:
    queries = [
        "what products do you have",
        "show me cotton tshirts",
        "tell me about jai sri ram tshirt",
    ]
    for q in queries:
        r = client.post("/chat", json={"session_id": f"down_{q[:8]}", "message": q})
        check(f"status 200 for '{q[:30]}'", r.status_code == 200)
        body = r.json()
        reply = body.get("reply", "")
        check(f"non-empty reply", len(reply) > 0)
        check(
            f"no generic error string",
            "having trouble" not in reply.lower() and "unexpected issue" not in reply.lower(),
            reply[:60],
        )
        # Reply should reference at least one real catalog item.
        titles = [(p.get("title") or "").lower() for p in CATALOG.all()]
        reply_l = reply.lower()
        check(
            f"reply grounded in catalog",
            any(t.split()[0] in reply_l for t in titles if t),
            f"reply starts: {reply[:50]}",
        )
finally:
    agent_mod._build_model_chain = _orig


print(f"\n=== Results: {PASS} passed, {FAIL} failed ===")
sys.exit(0 if FAIL == 0 else 1)
