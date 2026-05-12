# QuickVyapar — Smart Seller Assistant (MVP)

24/7 chatbot. Replies to buyers about products, negotiates within seller floor, redirects off-topic politely. **Hybrid architecture**: rule-based fast paths for cheap intents, LangChain tool-calling agent for open-ended product queries. FastAPI + Streamlit, Groq (primary) + Gemini (fallback).

## Setup

```powershell
# 1. virtualenv (optional)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. deps
pip install -r requirements.txt

# 3. API keys
copy .env.example .env
# fill GROQ_API_KEY (https://console.groq.com) + GEMINI_API_KEY (https://aistudio.google.com)

# 4. enrich products (adds minPrice / maxDiscountPct per product)
python scripts/enrich_products.py
```

Hand-edit `data/products_enriched.json` to tune each product's `minPrice` (seller's hard floor for negotiation).

## Run

Two terminals:

```powershell
# backend
uvicorn backend.main:app --reload --port 8000

# UI
python -m streamlit run ui/app.py
```

## Test from the terminal (no server needed)

```powershell
python -m scripts.chat_cli          # interactive chat (hits the app in-process)
python -m scripts.test_no_hallucination   # 67 headless guardrail assertions
```

(`test_no_hallucination.py` forces UTF-8 on stdout/stderr so it runs on a Windows cp1252 console without an `UnicodeEncodeError` on the `✓`/`₹` glyphs.)

`scripts/chat_cli.py` prints the reply plus metadata (`intent`, negotiation verdict, focal products, flags) for each turn. Suggested demo script:

```
what products do you have
price of jai sri ram tshirt
tell me about round neck
is it cotton
do you have shoes
can you do 120 for round neck
can you do 100 again
who is PM of india
I want to talk to seller
thanks
```

## Optional: PostgreSQL + vector embeddings

The MVP runs entirely on the in-memory keyword catalog. For semantic search you can ingest the catalog into a local PostgreSQL — **no `pgvector` extension required** (embeddings are stored in a `double precision[]` column; cosine similarity is done in numpy).

```powershell
# 1. create the DB (native PostgreSQL or pgAdmin)
psql -U postgres -c "CREATE DATABASE quickvyapar;"

# 2. .env
#   DATABASE_URL=postgresql://postgres:YOURPASSWORD@localhost:5432/quickvyapar
#   EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

# 3. deps (torch is ~2 GB)
pip install psycopg2-binary sentence-transformers

# 4. ingest
python -m scripts.enrich_products
python -m scripts.ingest_to_postgres
```

`scripts/ingest_to_postgres.py` creates a `products` table (all catalog fields + a 384-dim `embedding` column), embeds each product's `title | category | description | seller | price | location` with `all-MiniLM-L6-v2`, and upserts (re-runnable). `backend/vectorstore.py:search_similar(query, k)` then does numpy-cosine semantic search; it falls back to keyword `Catalog.search()` if `DATABASE_URL` is unset or the deps are missing, so nothing breaks without it.

> If `transformers` pulls in TensorFlow and you hit a protobuf version error, run `pip uninstall -y tensorflow tensorflow-intel` — `sentence-transformers` only needs `torch`.

## Architecture (hybrid)

| Intent | Path | LLM call? | Why |
|---|---|---|---|
| `OFF_TOPIC` | rule-based canned reply | no | zero cost, polite redirect |
| `GREETING` | rule-based canned reply | no | zero cost |
| `FAREWELL` | rule-based + contact block | no | guaranteed seller info |
| `REQUEST_HUMAN` | rule-based + contact block | no | guaranteed seller info |
| `NEGOTIATE` | deterministic engine + `message_hint` | no | math never hallucinates |
| `PRODUCT_QUERY` (simple) | **deterministic catalog responder** | no | list-all (any phrasing), "price of X", "tell me about X", "is it cotton", "do you have X", "cheapest / most expensive", "compare X and Y", "tshirt under ₹N" — answered straight from data, instant, can't fail |
| `PRODUCT_QUERY` (open-ended) | **LangChain tool-calling agent** | yes | only genuinely complex queries; falls back to the deterministic responder if the LLM is down or emits a malformed tool call |

Intent classifier: rule-first (`backend/intent.py`). Cheap, fast, no LLM.

`backend/main.py:_is_simple_product_query()` routes the easy, high-frequency query shapes to a deterministic responder (`backend/agent.py:_template_reply`) — free-tier LLMs are unreliable at tool-calling, so the bot answers these directly from the catalog. Only the rest hit the agent. The deterministic responder handles list-all (order-independent phrasing detector), single-product detail, price-only, fabric/material follow-ups, **cheapest / most expensive** (min/max by `priceInt`), **compare X and Y** (both products' real fields side by side, never cross-attributed), and **price-range filters** (`under ₹N`, `over ₹N`).

Agent (`backend/agent.py`) uses `langchain.agents.create_agent` with tools:
- `list_all_products()` → full catalog
- `search_products(query)` → top-6 from catalog
- `get_product_details(post_id)` → full product
- `negotiate_price(post_id, buyer_offer)` → wraps deterministic engine
- `get_seller_contact(post_id)` → seller phone/whatsapp/location

Model wiring: 6-tier `with_fallbacks` chain (see below). On a malformed Groq tool call (`tool_use_failed`) the agent retries once, then drops to the deterministic responder. Tools close over `Catalog` + `SessionState` so memory survives across turns (focal product, negotiation round, deal flags). Only the prior buyer turn is replayed into the agent context (not bot replies) to stop weak models anchoring to an old topic.

## API

- `GET /` → service info
- `GET /health` → `{ok, products}`
- `GET /greeting` → opening welcome line
- `GET /products` → catalog (lite, for UI sidebar)
- `POST /chat` body `{session_id, message}` → `{reply, intent, products, negotiation, notify_seller, end_chat}`

## Negotiation rules

Each product: `minPrice` (seller floor, default 85% of listed) + `maxDiscountPct`. Per session round:

| Round | offer ≥ listed | offer ≥ minPrice | offer < minPrice |
|---|---|---|---|
| any | accept @ listed | accept @ offer | counter |
| 1 | — | — | min(listed×0.95, midpoint(offer, listed)) |
| 2 | — | — | min(listed×0.92, midpoint(offer, listed)) |
| 3+ | — | — | final = minPrice |

Generic discount ask (no number): round 1 → 5% teaser, round 2+ → final `minPrice`.

On accept → deterministic deal-confirmation reply with seller phone/WhatsApp/location. Never goes through LLM.

## LLM Fallback Chain (never-fail)

The agent runs through a 6-tier fallback chain. Tiers 1–5 are LLM providers wired via LangChain `with_fallbacks`. Tier 6 is a deterministic non-LLM responder grounded in `data/products_enriched.json`.

| Tier | Provider | Model | Free tier? | Get key |
|---|---|---|---|---|
| 1 | Groq | `llama-3.1-8b-instant` | Yes (6k TPM) | https://console.groq.com |
| 2 | Cerebras | `llama-3.3-70b` | Yes | https://cloud.cerebras.ai |
| 3 | Google | `gemini-2.0-flash` | Yes | https://aistudio.google.com |
| 4 | OpenRouter | `mistralai/mistral-7b-instruct:free` | Yes | https://openrouter.ai/keys |
| 5 | Mistral | `mistral-small-latest` | Yes | https://console.mistral.ai |
| 6 | (none) | template responder | always available | — |

Add as many keys as you want to `.env`. Missing/empty keys are skipped silently. With zero keys configured, **the chat still works** — every reply derives from the real catalog via keyword search + numbered list formatting.

Each tier is tried in order; on `RateLimitError`, network failure, timeout, or any exception the next tier kicks in. End user sees a useful reply 100% of the time.

## Anti-hallucination guarantees

- Retrieval-grounded: only matched products in agent context.
- Pricing math 100% deterministic (`backend/negotiation.py`). LLM cannot override.
- `extract_offer()` strips quantity phrases (`500 pieces`, `order 200 units`) before reading a price — a bulk-order count is never mistaken for a buyer's offer (which would auto-confirm a deal at the listed price).
- NEGOTIATE target picker ignores bare-number overlap (`₹100` vs `100% cotton`) — a buyer haggling stays on the product in focus instead of jumping to whatever the offer digits happen to keyword-match.
- Catalog search drops stopword-only queries and `_best_match` rejects incidental hits (e.g. `iphone 15` latching onto `…15% polyster`) → honest "not in catalog" instead of a random product card.
- Off-topic gated by rule classifier before any LLM call.
- Tool returns are real catalog data; agent told to refuse if a detail is absent and never to attribute one product's fabric/price/description to another when comparing.
- Low temperature (0.3).
- Hard guardrail: any LLM reply starting with off-topic refusal stub gets replaced with canned text.

## UI notes (`ui/app.py`)

- Chat history widget keys are derived from message index (not `list.index()`), so repeated identical replies don't collide → no `StreamlitDuplicateElementKey` crash.
- Product cards only carry the "🤖 Ask about this" button in multi-product lists. A single-product reply (detail, follow-up, negotiation context) shows the card without the button — clicking it would just re-ask about the item already in focus. Single-product detail replies whose text already contains the product card skip the duplicate card entirely.
- `GET /chat` returns the full catalog in `products` when the reply is a list-all, `[]` for not-found / price-range replies, otherwise the focal product — so the UI gallery matches the text.

## Files

```
backend/
  agent.py        # langchain agent + 6-tier fallback + deterministic _template_reply
  catalog.py      # product index + title-weighted keyword search
  config.py       # env loading (LLM keys, DATABASE_URL, EMBED_MODEL)
  intent.py       # rule-based classifier + offer extraction
  llm.py          # multi-provider chat wrapper (legacy parity)
  main.py         # FastAPI routes + chat pipeline + simple-query fast-path
  negotiation.py  # deterministic price engine
  prompts.py      # system prompt + few-shots (legacy fallback)
  schemas.py      # Pydantic request/response
  session.py      # in-memory per-session state
  vectorstore.py  # optional PostgreSQL semantic search (numpy cosine)
data/
  products_enriched.json   # generated by scripts/enrich_products.py
scripts/
  enrich_products.py       # add minPrice/maxDiscountPct from products.json
  ingest_to_postgres.py    # load catalog + sentence embeddings into PostgreSQL
  chat_cli.py              # interactive terminal chat tester
  test_no_hallucination.py # 67 headless guardrail assertions
ui/
  app.py          # Streamlit chat UI
products.json     # source data (read-only)
PRD.txt           # requirements
```

## Test scenarios

In the Streamlit UI or `python -m scripts.chat_cli`:
- "What products do you have?" / "what are the products available with you?" / "show me what you've got" → numbered list of all 9 (deterministic, any phrasing).
- "Price of jai sri ram tshirt?" → ₹239 (deterministic).
- "Tell me about Round Neck" → full product card (deterministic).
- "Is it cotton?" → answers about the focal product (Round Neck → Metty), not a random match.
- "Which is your cheapest tshirt?" → ₹125 item. "Most expensive?" → ₹250 item.
- "Compare Round Neck and jai sri ram tshirt" → both products' real fields, no mixing.
- "Any tshirt under ₹140?" → just the ₹125 + ₹130 items.
- "Can you do ₹180 for polyster dotnet tshirt?" (₹130 listed, ~₹110 floor) → accept + contact block.
- "Can you do ₹100?" on a ₹130 item → counter ₹115 (round 1) → ₹110 (round 3).
- "Can I get a bulk discount for 500 pieces of round neck?" → generic-discount teaser, **not** a confirmed ₹150 deal (the `500` is a quantity, not an offer).
- "Tell me about the iPhone 15" → "not in catalog" (no random tshirt card).
- "Who is PM of India?" → polite redirect, zero LLM cost.
- "I want to talk to the seller" → contact block + `notify_seller=true`.
- "do you have shoes?" → polite "not in catalog".
- "thanks" / "I'll buy it" → farewell + chat ends.

## Future

- Persistent session store (Redis swap for `backend/session.py`).
- Real seller notifications (currently logged + flag in response).
- Seller settings UI (bot on/off, FAQs, per-product min price).
- Vector retrieval if catalog grows beyond ~50 items.
