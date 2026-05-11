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
5. Call negotiate_price ONLY when THIS message states a price, makes an offer, or explicitly asks for a discount / lower price. For plain questions like "what products do you have", "price of X", "is it cotton", "show me tshirts" — DO NOT call negotiate_price; use list_all_products / search_products / get_product_details and just answer.
6. Ignore the subject of earlier turns when the current message changes topic. If the buyer was negotiating before but now asks "what do you have", answer the new question — do not continue the old negotiation.
7. Never invent a discount or counter-price. Only negotiate_price produces one.
8. Reply in buyer's language (English / Hindi / Hinglish / Gujarati). Quote prices with ₹. Keep it under 80 words unless listing.
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


_GENERIC_WORDS = {
    "tshirt", "t-shirt", "shirt", "shirts", "tshirts", "product", "products",
    "tell", "me", "about", "details", "detail", "info", "information", "the",
    "a", "an", "of", "for", "is", "it", "this", "that", "your", "you", "have",
    "do", "show", "what", "which", "price", "rate", "cost", "stock", "available",
    "give", "want", "need", "please", "fabric", "size", "sizes",
}


def _format_one(p: dict) -> str:
    """Friendly multi-line description of a single product (deterministic)."""
    title = (p.get("title") or "").strip()
    price = p.get("priceInt") or p.get("price")
    unit = (p.get("priceUnitType") or "").strip()
    qty = p.get("quantity", "?")
    qunit = (p.get("quantityUnitType") or "").strip()
    cat = (p.get("categoryName") or "").strip()
    desc = (p.get("description") or "").strip()
    seller = (p.get("user_name") or p.get("postCompany") or "the seller").strip()
    loc = ", ".join(x for x in [
        (p.get("location") or "").strip(),
        (p.get("districtName") or "").strip(),
        (p.get("stateName") or "").strip(),
    ] if x)
    lines = [f"**{title}**", ""]
    lines.append(f"• Price: ₹{price} {unit}".rstrip())
    if cat:
        lines.append(f"• Category: {cat}")
    lines.append(f"• In stock: {qty} {qunit}".rstrip())
    if seller:
        lines.append(f"• Seller: {seller}" + (f" ({loc})" if loc else ""))
    if desc:
        snippet = desc if len(desc) <= 400 else desc[:400].rsplit(" ", 1)[0] + "…"
        lines.append("")
        lines.append(snippet)
    lines.append("")
    lines.append("Want a small discount on this, or details of another item?")
    return "\n".join(lines)


def _best_match(catalog: Catalog, user_msg: str, state: SessionState):
    """Pick the single product the buyer most likely means.

    Returns (product_or_None, is_followup). `is_followup` True means the query
    was a pronoun-y follow-up about the already-focal product ("is it cotton",
    "price?") rather than a fresh product mention.
    """
    import re as _re
    m = (user_msg or "").lower().strip()
    focal = catalog.get(state.current_product_id) if state.current_product_id else None

    # Pronoun-y follow-up patterns -> stay on the focal product.
    followup = bool(_re.match(
        r"^(is|does|do|are)\s+(it|this|that|they|these)\b", m
    )) or bool(_re.match(
        r"^(what|how)\s+about\s+(it|this|that)\b", m
    )) or m in {"price?", "price", "and the price?", "how much?", "how much"} or (
        # very short + focal set + no obvious product noun
        focal is not None and len(m.split()) <= 4
        and not (set(_re.findall(r"[a-z0-9]+", m)) - _GENERIC_WORDS)
    )
    if followup and focal is not None:
        return focal, True

    results = catalog.search(user_msg, k=5)
    if results:
        return results[0], False
    # No keyword match at all. If there's a focal AND the query is plausibly
    # still about it (no new distinctive nouns), use it; otherwise None.
    distinctive = set(_re.findall(r"[a-z0-9]+", m)) - _GENERIC_WORDS
    if focal is not None and not distinctive:
        return focal, True
    return None, False


def _template_reply(catalog: Catalog, state: SessionState, user_msg: str) -> str:
    """Deterministic non-LLM responder. Final tier — never raises, always returns text.
    Used when every LLM provider fails (rate limits, malformed tool calls, outages).
    Tries to actually answer the question, not just dump a list.
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
        return "Glad I could help 🙏 Have a great day!"

    wants_list = (not user_msg.strip()) or any(
        k in msg_lower for k in ("all product", "all the product", "everything",
                                 "list all", "show all", "what products", "what do you have",
                                 "what all", "show me products", "your products")
    )
    if wants_list:
        items = catalog.all()
        lines = [f"Here are all {len(items)} products in our catalog:", ""]
        for i, p in enumerate(items, 1):
            t = (p.get("title") or "").strip()
            pr = p.get("priceInt") or p.get("price")
            u = (p.get("priceUnitType") or "").strip()
            q = p.get("quantity", "?")
            lines.append(f"{i}. **{t}** — ₹{pr} {u} ({q} in stock)".replace("  ", " "))
        lines.append("")
        lines.append("Tell me which one you'd like details on, or ask for a discount.")
        return "\n".join(lines)

    # Specific-product question (price / details / fabric / "tell me about X" / follow-up).
    p, is_followup = _best_match(catalog, user_msg, state)
    if p is not None:
        state.current_product_id = p.get("post_id")
        asks_price_only = any(k in msg_lower for k in ("price", "rate", "cost", "how much", "kitne", "kitna", "daam")) \
            and not any(k in msg_lower for k in ("detail", "about", "tell me", "describe", "more"))
        if asks_price_only:
            price = p.get("priceInt") or p.get("price")
            unit = (p.get("priceUnitType") or "").strip()
            return (f"The price of **{(p.get('title') or '').strip()}** is ₹{price} {unit}".rstrip()
                    + ". Want a discount on it?")
        # Fabric / material follow-up — pull the line from the description if present.
        if is_followup and any(k in msg_lower for k in ("cotton", "polyester", "polyster", "fabric", "material", "gsm")):
            desc = (p.get("description") or "")
            fab_line = next((ln.strip(" -•\t") for ln in desc.splitlines()
                             if any(w in ln.lower() for w in ("fabric", "cotton", "polyester", "polyster", "gsm", "metty", "blend"))), "")
            title = (p.get("title") or "").strip()
            if fab_line:
                return f"For **{title}** — {fab_line}\n\nWant more details or a discount on it?"
            return f"Here's what I have on **{title}**:\n\n{_format_one(p)}"
        return _format_one(p)

    # Nothing matched — and not a follow-up. Be honest: not in catalog.
    items = catalog.all()
    lines = [
        "I don't have that in the catalog right now. Here's everything we do have:",
        "",
    ]
    for i, q in enumerate(items, 1):
        t = (q.get("title") or "").strip()
        pr = q.get("priceInt") or q.get("price")
        lines.append(f"{i}. **{t}** — ₹{pr}")
    lines.append("")
    lines.append("Want details on any of these?")
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

    # Replay ONLY the last prior buyer message (not assistant replies). This
    # gives just enough context to resolve pronouns ("is it cotton") without
    # anchoring the weak model to the *topic* of the previous bot reply (e.g.
    # an in-progress negotiation). The focal product itself is carried in
    # `state.current_product_id`, which the tools read directly.
    history_msgs = []
    prior_user = None
    for m in reversed(state.history[:-1]):  # skip the just-appended user msg
        if m["role"] == "user":
            prior_user = m["content"]
            break
    if prior_user and prior_user.strip().lower() != user_message.strip().lower():
        history_msgs.append(HumanMessage(content=f"(earlier the buyer asked: {prior_user})"))

    messages = history_msgs + [HumanMessage(content=user_message)]

    result = None
    last_err = None
    for attempt in range(2):  # one retry — Groq llama often emits a bad tool call once, then recovers
        try:
            result = agent.invoke({"messages": messages})
            break
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "tool_use_failed" in msg or "failed to call a function" in msg or "tool call validation" in msg:
                logger.warning("Tool-call malformed (attempt %d) — retrying.", attempt + 1)
                continue
            break  # other errors: don't retry, drop to template
    if result is None:
        logger.warning("Agent failed (%s) — using deterministic template responder.", last_err)
        return _template_reply(catalog, state, user_message)

    # Extract final assistant message.
    msgs = result.get("messages", []) if isinstance(result, dict) else []
    for m in reversed(msgs):
        content = getattr(m, "content", None)
        role = getattr(m, "type", "") or getattr(m, "role", "")
        if content and role in ("ai", "assistant"):
            return content if isinstance(content, str) else str(content)
    return _template_reply(catalog, state, user_message)
