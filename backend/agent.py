"""LangChain tool-calling agent for product Q&A and negotiation.

Hybrid design: rule-based intents (OFF_TOPIC / GREETING / FAREWELL / REQUEST_HUMAN
when contact is needed) are handled in main.py without an LLM. Open-ended
PRODUCT_QUERY and NEGOTIATE go through this agent — it picks tools dynamically.

Critical guarantees preserved:
- Negotiation math is deterministic (the `negotiate_price` tool calls
  `negotiation.negotiate()` — the LLM never invents a number).
- Catalog answers are grounded — tools return real product data, not LLM guesses.
"""
from __future__ import annotations

import logging
from typing import Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from .catalog import Catalog
from .config import (
    GROQ_API_KEY, GEMINI_API_KEY, CEREBRAS_API_KEY,
    OPENROUTER_API_KEY, MISTRAL_API_KEY,
    GROQ_MODEL, GEMINI_MODEL, CEREBRAS_MODEL,
    OPENROUTER_MODEL, MISTRAL_MODEL,
)
from .intent import classify
from .negotiation import negotiate
from .session import SessionState

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are QuickVyapar's friendly Smart Seller Assistant.

TOOLS (always call — never invent data):
- list_all_products(): every product in catalog. Use when buyer asks "what do you have", "show all", "all products".
- search_products(query): up to 6 matches for a specific query (cotton tshirts, jai sri ram, etc).
- get_product_details(post_id): full info for one item. Call this when buyer names a specific product.
- negotiate_price(post_id, buyer_offer): deterministic price decision. Use its decision and counter_price verbatim.
- get_seller_contact(post_id): seller phone / whatsapp / location.

OUTPUT RULES:
1. NEVER paste raw tool output (no `post_id=...`, `title=...` field dumps). Always synthesize a friendly natural-language reply.
2. When listing products, format as a numbered list: `1. **Title** — ₹price (qty in stock)`. Limit to 6.
3. When buyer names a specific product (e.g. "tell me about Round Neck"), pick the single best match from search and answer ONLY about that one. Do not list multiple unrelated items.
4. If a detail isn't in tool output, say honestly: "I don't have that detail handy — want me to check with the seller?"
5. For ANY haggling, you MUST call negotiate_price. Never invent discounts.
6. Reply in buyer's language (English / Hindi / Hinglish / Gujarati). Quote prices with ₹. Keep it under 80 words unless listing.
"""


def _format_product(p: dict, short: bool = False) -> str:
    desc_limit = 100 if short else 250
    title = (p.get('title') or '').strip()
    price = p.get('priceInt') or p.get('price')
    unit = p.get('priceUnitType', '') or ''
    cat = p.get('categoryName', '') or ''
    qty = f"{p.get('quantity','?')} {p.get('quantityUnitType','')}"
    desc = (p.get('description', '') or '').strip()[:desc_limit]
    return (
        f"[id:{p.get('post_id')}] {title}\n"
        f"  price: ₹{price} {unit}\n"
        f"  category: {cat}\n"
        f"  stock: {qty}\n"
        f"  about: {desc}"
    )


def _format_compact(p: dict) -> str:
    """One-line per product for list_all view."""
    title = (p.get('title') or '').strip()
    price = p.get('priceInt') or p.get('price')
    unit = p.get('priceUnitType', '') or ''
    qty = p.get('quantity', '?')
    return f"[id:{p.get('post_id')}] {title} — ₹{price} {unit} ({qty} in stock)"


def build_tools(catalog: Catalog, state: SessionState):
    """Build per-request tool functions that close over the session state.

    Tools mutate `state` (e.g. focal product, negotiation round, deal flags) so
    the rest of the pipeline can read them after the agent finishes.
    """

    @tool
    def list_all_products() -> str:
        """Return EVERY product in the catalog (compact one-line each). Use when the buyer asks 'what do you have', 'show all', 'list products', 'all items'."""
        items = catalog.all()
        if not items:
            return "Catalog is empty."
        return f"Total {len(items)} products:\n" + "\n".join(_format_compact(p) for p in items)

    @tool
    def search_products(query: str) -> str:
        """Search catalog for products matching a specific query (e.g. 'cotton tshirt', 'jai sri ram', 'round neck'). Returns up to 6 matches in compact form. Use when buyer asks about a category, fabric, or names part of a product."""
        results = catalog.search(query, k=6)
        if not results:
            return "No products matched. Suggest a related category."
        if len(results) == 1:
            state.current_product_id = results[0].get("post_id")
        return "\n".join(_format_compact(p) for p in results)

    @tool
    def get_product_details(post_id: str) -> str:
        """Get full details of a single product by post_id. Call this once the user has narrowed down to one item."""
        p = catalog.get(post_id)
        if not p:
            return f"No product found with post_id={post_id}."
        state.current_product_id = post_id
        return _format_product(p)

    @tool
    def negotiate_price(post_id: str, buyer_offer: Optional[int] = None) -> str:
        """Run the deterministic negotiation engine. Call this for ANY price haggling.
        - post_id: which product is being negotiated. If unsure, use the most recently discussed product.
        - buyer_offer: the price the buyer is offering, in INR. Use null if the buyer asked for a generic discount without a number.
        Returns the decision (accept/counter), the price to use, and a phrasing hint. Use the decision and price verbatim in your reply.
        """
        product = catalog.get(post_id) or (
            catalog.get(state.current_product_id) if state.current_product_id else None
        )
        if not product:
            return "No focal product known. Call search_products or get_product_details first."

        state.current_product_id = product.get("post_id")
        state.negotiation_round += 1
        outcome = negotiate(
            listed_price=int(product.get("priceInt") or 0),
            min_price=int(product.get("minPrice") or 0),
            buyer_offer=buyer_offer,
            round_num=state.negotiation_round,
        )

        if outcome.decision == "accept":
            agreed = outcome.counter_price or int(product.get("priceInt") or 0)
            state.deal_accepted = True
            state.deal_price = agreed
            state.deal_product_id = product.get("post_id")

        return (
            f"product={product.get('title','').strip()} (post_id={product.get('post_id')})\n"
            f"listed_price=₹{int(product.get('priceInt') or 0)}\n"
            f"buyer_offer={('₹'+str(buyer_offer)) if buyer_offer is not None else 'not specified'}\n"
            f"decision={outcome.decision}\n"
            f"counter_price={('₹'+str(outcome.counter_price)) if outcome.counter_price else 'n/a'}\n"
            f"round={state.negotiation_round}\n"
            f"phrasing_hint: {outcome.message_hint}"
        )

    @tool
    def get_seller_contact(post_id: str) -> str:
        """Get the seller's phone, whatsapp, and location for a product. Use when the buyer wants to contact the seller, or right after a deal is accepted."""
        p = catalog.get(post_id) or (
            catalog.get(state.current_product_id) if state.current_product_id else None
        )
        if not p:
            return "No product known."
        seller = (p.get("user_name") or p.get("postCompany") or "the seller").strip()
        phone = (p.get("phoneNumber") or "").strip()
        whatsapp = (p.get("whatsappNumber") or "").strip()
        location = ", ".join(
            x for x in [
                (p.get("location") or "").strip(),
                (p.get("districtName") or "").strip(),
                (p.get("stateName") or "").strip(),
            ] if x
        )
        parts = [f"seller={seller}"]
        if phone:
            parts.append(f"phone={phone}")
        if whatsapp:
            parts.append(f"whatsapp={whatsapp}")
        if location:
            parts.append(f"location={location}")
        return "\n".join(parts)

    return [list_all_products, search_products, get_product_details, negotiate_price, get_seller_contact]


def _build_model_chain():
    """Multi-tier fallback chain. Returns chain (or None if zero providers configured).
    Order: Groq → Cerebras → Gemini → OpenRouter → Mistral.
    Each provider wrapped in try/except so missing package or bad key skips silently.
    """
    chain: list = []

    if GROQ_API_KEY:
        try:
            from langchain_groq import ChatGroq
            chain.append(ChatGroq(
                model=GROQ_MODEL, api_key=GROQ_API_KEY,
                temperature=0.3, max_tokens=400, timeout=15,
            ))
        except Exception as e:
            logger.warning("Groq init failed: %s", e)

    if CEREBRAS_API_KEY:
        try:
            from langchain_cerebras import ChatCerebras
            chain.append(ChatCerebras(
                model=CEREBRAS_MODEL, api_key=CEREBRAS_API_KEY,
                temperature=0.3, max_tokens=400, timeout=15,
            ))
        except Exception as e:
            logger.warning("Cerebras init failed: %s", e)

    if GEMINI_API_KEY:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            chain.append(ChatGoogleGenerativeAI(
                model=GEMINI_MODEL, google_api_key=GEMINI_API_KEY,
                temperature=0.3, max_output_tokens=400,
            ))
        except Exception as e:
            logger.warning("Gemini init failed: %s", e)

    if OPENROUTER_API_KEY:
        try:
            from langchain_openai import ChatOpenAI
            chain.append(ChatOpenAI(
                model=OPENROUTER_MODEL, api_key=OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
                temperature=0.3, max_tokens=400, timeout=15,
            ))
        except Exception as e:
            logger.warning("OpenRouter init failed: %s", e)

    if MISTRAL_API_KEY:
        try:
            from langchain_mistralai import ChatMistralAI
            chain.append(ChatMistralAI(
                model=MISTRAL_MODEL, api_key=MISTRAL_API_KEY,
                temperature=0.3, max_tokens=400, timeout=15,
            ))
        except Exception as e:
            logger.warning("Mistral init failed: %s", e)

    if not chain:
        return None
    primary = chain[0]
    if len(chain) == 1:
        return primary
    return primary.with_fallbacks(chain[1:])


# Backwards-compat alias.
_make_model = _build_model_chain


def _template_reply(catalog: Catalog, state: SessionState, user_msg: str) -> str:
    """Deterministic non-LLM responder. Final tier — never raises, always returns text.
    Used when every LLM provider fails (rate limits, network, expired keys).
    """
    intent = classify(user_msg)
    msg_lower = user_msg.lower()

    if intent == "GREETING":
        return ("Hi 👋 Welcome to QuickVyapar! I can share product details, prices, "
                "sizes, stock, and help you negotiate. What are you looking for?")

    if intent == "OFF_TOPIC":
        return ("That's a great question, but I'm here just to help with our products. "
                "Want to see what we have?")

    if intent == "FAREWELL":
        return ("Glad I could help 🙏 Have a great day!")

    # PRODUCT_QUERY / NEGOTIATE / REQUEST_HUMAN / unknown — show catalog grounded reply.
    listed_all = any(k in msg_lower for k in ("all", "everything", "list", "show me"))
    if listed_all or not user_msg.strip():
        items = catalog.all()[:8]
        header = f"Here are {len(items)} products from our catalog:"
    else:
        items = catalog.search(user_msg, k=6)
        if not items:
            items = catalog.all()[:6]
            header = "I couldn't find an exact match, but here's what we have:"
        else:
            header = f"Found {len(items)} matching product(s):"

    if len(items) == 1:
        state.current_product_id = items[0].get("post_id")

    lines = [header, ""]
    for i, p in enumerate(items, 1):
        title = (p.get("title") or "").strip()
        price = p.get("priceInt") or p.get("price")
        unit = p.get("priceUnitType", "") or ""
        qty = p.get("quantity", "?")
        lines.append(f"{i}. **{title}** — ₹{price} {unit} ({qty} in stock)")

    lines.append("")
    lines.append("Let me know which one you'd like to know more about, or ask for a discount.")
    return "\n".join(lines)


def run_agent(catalog: Catalog, state: SessionState, user_message: str) -> str:
    """Invoke agent for one buyer turn. Never raises — falls back to deterministic
    template responder if every LLM provider fails. Always returns useful reply."""
    model = _build_model_chain()
    if model is None:
        logger.warning("No LLM providers configured — using template fallback.")
        return _template_reply(catalog, state, user_message)

    tools = build_tools(catalog, state)
    agent = create_agent(model=model, tools=tools, system_prompt=SYSTEM_PROMPT)

    # Replay last 2 turns only — keeps token usage low while preserving pronoun context.
    history_msgs = []
    for m in state.history[-5:-1]:  # exclude the just-appended user msg
        if m["role"] == "user":
            history_msgs.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            # Truncate prior assistant replies — we only need pronoun context.
            content = m["content"]
            if len(content) > 200:
                content = content[:200] + "…"
            history_msgs.append(AIMessage(content=content))

    messages = history_msgs + [HumanMessage(content=user_message)]

    try:
        result = agent.invoke({"messages": messages})
    except Exception as e:
        logger.warning("All LLM tiers failed (%s) — using template fallback.", e)
        return _template_reply(catalog, state, user_message)

    # Extract final assistant message.
    msgs = result.get("messages", []) if isinstance(result, dict) else []
    for m in reversed(msgs):
        content = getattr(m, "content", None)
        role = getattr(m, "type", "") or getattr(m, "role", "")
        if content and role in ("ai", "assistant"):
            return content if isinstance(content, str) else str(content)
    return _template_reply(catalog, state, user_message)
