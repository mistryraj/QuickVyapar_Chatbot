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

## Architecture (hybrid)

| Intent | Path | LLM call? | Why |
|---|---|---|---|
| `OFF_TOPIC` | rule-based canned reply | no | zero cost, polite redirect |
| `GREETING` | rule-based canned reply | no | zero cost |
| `FAREWELL` | rule-based + contact block | no | guaranteed seller info |
| `REQUEST_HUMAN` | rule-based + contact block | no | guaranteed seller info |
| `NEGOTIATE` | deterministic engine + `message_hint` | no | math never hallucinates |
| `PRODUCT_QUERY` | **LangChain tool-calling agent** | yes | LLM picks tools, reasons over results |

Intent classifier: rule-first (`backend/intent.py`). Cheap, fast, no LLM.

Agent (`backend/agent.py`) uses `langchain.agents.create_agent` with tools:
- `search_products(query)` → top-3 from catalog
- `get_product_details(post_id)` → full product
- `negotiate_price(post_id, buyer_offer)` → wraps deterministic engine
- `get_seller_contact(post_id)` → seller phone/whatsapp/location

Model wiring: `ChatGroq` primary, `ChatGoogleGenerativeAI` fallback via `with_fallbacks`. Tools close over `Catalog` + `SessionState` so memory survives across turns (focal product, negotiation round, deal flags).

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
- Off-topic gated by rule classifier before any LLM call.
- Tool returns are real catalog data; agent told to refuse if detail absent.
- Low temperature (0.3).
- Hard guardrail: any LLM reply starting with off-topic refusal stub gets replaced with canned text.

## Files

```
backend/
  agent.py        # langchain tool-calling agent
  catalog.py      # product index + keyword search
  config.py       # env loading
  intent.py       # rule-based classifier + offer extraction
  llm.py          # Groq + Gemini wrapper (legacy, used by negotiation deterministic path)
  main.py         # FastAPI routes + chat pipeline
  negotiation.py  # deterministic price engine
  prompts.py      # system prompt + few-shots (legacy fallback)
  schemas.py      # Pydantic request/response
  session.py      # in-memory per-session state
data/
  products_enriched.json   # generated by scripts/enrich_products.py
scripts/
  enrich_products.py       # add minPrice/maxDiscountPct from products.json
ui/
  app.py          # Streamlit chat UI
products.json     # source data (read-only)
PRD.txt           # requirements
```

## Test scenarios

In Streamlit UI:
- "What products do you have?" → agent calls `search_products`, lists items.
- "Price of jai sri ram tshirt?" → ₹239.
- "Can you do ₹180 for polyster dotnet tshirt?" (₹130 listed, ~₹110 floor) → accept + contact block.
- "Can you do ₹100?" on ₹130 item → counter ₹115 (round 1) → ₹110 (round 3).
- "Who is PM of India?" → polite redirect.
- "I want to talk to the seller" → contact block + `notify_seller=true`.
- "do you have shoes?" → polite "not in catalog".
- "thanks" / "I'll buy it" → farewell + chat ends.

## Future

- Persistent session store (Redis swap for `backend/session.py`).
- Real seller notifications (currently logged + flag in response).
- Seller settings UI (bot on/off, FAQs, per-product min price).
- Vector retrieval if catalog grows beyond ~50 items.
