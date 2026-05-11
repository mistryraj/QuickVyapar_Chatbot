"""Terminal chat client for the QuickVyapar Smart Seller Assistant.

Hits the backend in-process via FastAPI TestClient — no need to run uvicorn.
Type messages, see replies. `quit` / `exit` / Ctrl-C to stop.

Run:  python -m scripts.chat_cli
"""
import io
import sys
import uuid
from pathlib import Path

# Windows console is cp1252 by default — force UTF-8 so the rupee sign etc print.
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)
session_id = "cli-" + uuid.uuid4().hex[:8]

print("QuickVyapar Smart Seller Assistant — terminal chat")
print(f"session_id = {session_id}")
print("Type your message. 'quit' to exit.\n")

# opening greeting
try:
    g = client.get("/greeting").json()
    print(f"BOT: {g.get('reply','')}\n")
except Exception:
    pass

while True:
    try:
        msg = input("YOU: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nbye")
        break
    if not msg:
        continue
    if msg.lower() in {"quit", "exit", "q"}:
        print("bye")
        break
    try:
        r = client.post("/chat", json={"session_id": session_id, "message": msg})
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[error] {e}\n")
        continue
    print(f"\nBOT: {data.get('reply','')}")
    meta = []
    if data.get("intent"):
        meta.append(f"intent={data['intent']}")
    if data.get("negotiation"):
        n = data["negotiation"]
        meta.append(f"nego={n.get('decision')}@{n.get('counter_price')} round={n.get('round_num')}")
    if data.get("notify_seller"):
        meta.append("notify_seller=True")
    if data.get("end_chat"):
        meta.append("end_chat=True")
    if data.get("products"):
        meta.append(f"products={[p['title'] for p in data['products']]}")
    if meta:
        print(f"     ({'  '.join(meta)})")
    print()
    if data.get("end_chat"):
        print("(chat ended — starting fresh session)")
        session_id = "cli-" + uuid.uuid4().hex[:8]
