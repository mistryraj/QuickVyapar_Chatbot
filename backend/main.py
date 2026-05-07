import logging
import re
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from . import session as sess
from .agent import run_agent
from .catalog import load_catalog
from .intent import classify, extract_offer
from .negotiation import negotiate
from .schemas import (
    ChatRequest,
    ChatResponse,
    NegotiationResult,
    ProductLite,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart-seller")

app = FastAPI(title="QuickVyapar Smart Seller Assistant", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CATALOG = load_catalog()
OFF_TOPIC_REPLY = (
    "That's a great question, but I'm here just to help with our products. "
    "I'd love to tell you about anything from our catalog — fabric, price, sizes, stock, "
    "or even a small discount. What can I show you?"
)
GREETING_REPLY = (
    "Hi 👋 Welcome to QuickVyapar! I'm the seller's assistant — I can share product details, "
    "prices, sizes, stock, and even help with a small discount. What are you looking for today?"
)
FAREWELL_REPLY = (
    "Glad I could help 🙏 If you change your mind or need anything else, just start a new chat. "
    "Have a great day!"
)


def _contact_block(product: dict) -> str:
    seller = (product.get("user_name") or product.get("postCompany") or "the seller").strip()
    phone = (product.get("phoneNumber") or "").strip()
    whatsapp = (product.get("whatsappNumber") or "").strip()
    location = ", ".join(
        x for x in [
            (product.get("location") or "").strip(),
            (product.get("districtName") or "").strip(),
            (product.get("stateName") or "").strip(),
        ] if x
    )
    lines = [f"**Seller:** {seller}"]
    if phone:
        lines.append(f"**Phone:** [{phone}](tel:{phone})")
    if whatsapp and whatsapp != phone:
        lines.append(f"**WhatsApp:** [{whatsapp}](https://wa.me/91{whatsapp})")
    elif whatsapp:
        lines.append(f"**WhatsApp:** [{whatsapp}](https://wa.me/91{whatsapp})")
    if location:
        lines.append(f"**Location:** {location}")
    return "\n".join(lines)


def _deal_confirmation_reply(product: dict, agreed_price: int) -> str:
    title = (product.get("title") or "this product").strip()
    return (
        f"🎉 **Deal confirmed at ₹{agreed_price}** for *{title}*.\n\n"
        f"I've notified the seller — they'll reach out to you shortly. "
        f"You can also contact them directly:\n\n"
        f"{_contact_block(product)}\n\n"
        f"Thank you for shopping with us! 🙏"
    )


def _to_lite(products) -> List[ProductLite]:
    out = []
    for p in products:
        images = p.get("images") or []
        out.append(
            ProductLite(
                post_id=p.get("post_id", ""),
                title=(p.get("title") or "").strip(),
                price=int(p.get("priceInt") or 0),
                priceUnitType=p.get("priceUnitType", ""),
                categoryName=p.get("categoryName", ""),
                image=images[0] if images else None,
                user_name=(p.get("user_name") or "").strip(),
            )
        )
    return out


@app.get("/")
def root():
    return JSONResponse(
        {
            "service": "QuickVyapar Smart Seller Assistant",
            "endpoints": ["/health", "/products", "/chat (POST)", "/docs"],
        }
    )


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/health")
def health():
    return {"ok": True, "products": len(CATALOG.all())}


@app.get("/greeting")
def greeting():
    return {"reply": GREETING_REPLY, "intent": "GREETING"}


@app.get("/products")
def products():
    return {"data": _to_lite(CATALOG.all())}


@app.get("/sellers")
def sellers():
    """Unique seller contacts derived from catalog. Used by UI seller drawer."""
    seen: dict = {}
    for p in CATALOG.all():
        key = (p.get("phoneNumber") or "") + "|" + (p.get("user_name") or "")
        if not key.strip("|"):
            continue
        if key in seen:
            seen[key]["product_count"] += 1
            continue
        seen[key] = {
            "name": (p.get("user_name") or p.get("postCompany") or "Seller").strip(),
            "company": (p.get("postCompany") or "").strip(),
            "phone": (p.get("phoneNumber") or "").strip(),
            "whatsapp": (p.get("whatsappNumber") or "").strip(),
            "location": ", ".join(
                x for x in [
                    (p.get("location") or "").strip(),
                    (p.get("districtName") or "").strip(),
                    (p.get("stateName") or "").strip(),
                ] if x
            ),
            "avatar": p.get("userProfilePicture") or "",
            "product_count": 1,
        }
    return {"data": list(seen.values())}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    state = sess.get(req.session_id)
    sess.append(req.session_id, "user", req.message)

    intent = classify(req.message)
    logger.info("session=%s intent=%s msg=%r", req.session_id, intent, req.message)

    if intent == "OFF_TOPIC":
        sess.append(req.session_id, "assistant", OFF_TOPIC_REPLY)
        return ChatResponse(reply=OFF_TOPIC_REPLY, intent=intent)

    if intent == "FAREWELL":
        # If a deal has been struck, or the user is signaling purchase intent on a focal product,
        # close out by re-sharing the seller's contact details.
        focal = None
        if state.deal_product_id:
            focal = CATALOG.get(state.deal_product_id)
        elif state.current_product_id and re.search(r"\b(buy|order|purchase|deal)\b", req.message.lower()):
            focal = CATALOG.get(state.current_product_id)

        if focal is not None:
            agreed = state.deal_price or int(focal.get("priceInt") or 0)
            reply = (
                f"Awesome! 🎉 The deal is confirmed at ₹{agreed} for *{(focal.get('title') or '').strip()}*.\n\n"
                f"The seller will contact you shortly. You can also reach out to them directly:\n\n"
                f"{_contact_block(focal)}\n\n"
                f"Thanks for choosing QuickVyapar! 🙏"
            )
            sess.append(req.session_id, "assistant", reply)
            return ChatResponse(
                reply=reply,
                intent=intent,
                products=_to_lite([focal]),
                notify_seller=True,
                end_chat=True,
            )
        sess.append(req.session_id, "assistant", FAREWELL_REPLY)
        return ChatResponse(reply=FAREWELL_REPLY, intent=intent, end_chat=True)

    if intent == "GREETING":
        sess.append(req.session_id, "assistant", GREETING_REPLY)
        return ChatResponse(reply=GREETING_REPLY, intent=intent)

    if intent == "REQUEST_HUMAN":
        focal = CATALOG.get(state.current_product_id) if state.current_product_id else None
        if focal is not None:
            reply = (
                "Sure — I've notified the seller and they'll reach out shortly. "
                "You can also contact them directly:\n\n"
                f"{_contact_block(focal)}"
            )
            sess.append(req.session_id, "assistant", reply)
            return ChatResponse(
                reply=reply,
                intent=intent,
                products=_to_lite([focal]),
                notify_seller=True,
            )
        reply = (
            "Got it — I'll notify the seller right away and they'll join the chat as soon as possible. "
            "Meanwhile, is there anything specific I can share about the products?"
        )
        sess.append(req.session_id, "assistant", reply)
        return ChatResponse(reply=reply, intent=intent, notify_seller=True)

    # Retrieve products. Prefer the previously-discussed product on follow-ups.
    retrieved = CATALOG.search(req.message, k=5)
    current = CATALOG.get(state.current_product_id) if state.current_product_id else None

    # If the new query doesn't strongly identify a different product, keep the focal one in front.
    msg_tokens = set(re.findall(r"[a-z0-9]+", req.message.lower()))
    looks_like_followup = (
        current is not None
        and (
            not retrieved
            or retrieved[0].get("post_id") == current.get("post_id")
            # short follow-ups like "what about size?", "and the price?", "is it cotton?"
            or len(msg_tokens) <= 5
            or "it" in msg_tokens or "this" in msg_tokens or "that" in msg_tokens
        )
    )
    if looks_like_followup:
        # Put the focal product first; keep other matches as supporting context.
        others = [p for p in retrieved if p.get("post_id") != current.get("post_id")]
        retrieved = [current] + others[:4]

    negotiation_payload = None
    negotiation_block = ""

    if intent == "NEGOTIATE":
        target = None
        # If the buyer named a *specific* product in this message, prefer that
        # over a stale focal. Distinguishing tokens = overlap between message
        # and the top retrieved product's title, minus generic noise words.
        GENERIC = {
            "tshirt", "t-shirt", "shirt", "shirts", "tshirts", "product", "products",
            "price", "cost", "rate", "rs", "rupees", "rupee", "piece", "pcs",
            "the", "a", "an", "for", "at", "of", "do", "you", "can", "could",
            "would", "will", "give", "sell", "offer", "this", "that", "it",
        }
        if retrieved:
            top = retrieved[0]
            top_tokens = set(re.findall(r"[a-z0-9]+", (top.get("title") or "").lower())) - GENERIC
            named_tokens = msg_tokens & top_tokens
            current_id = state.current_product_id
            if (
                top.get("post_id") != current_id
                and len(named_tokens) >= 1  # any non-generic match -> trust search
            ):
                target = top
        if target is None and state.current_product_id:
            target = CATALOG.get(state.current_product_id)
        if target is None and retrieved:
            target = retrieved[0]

        if target is None:
            reply = "Which product are you negotiating for? I can share options if you'd like."
            sess.append(req.session_id, "assistant", reply)
            return ChatResponse(reply=reply, intent=intent)

        # Switching focal mid-conversation → reset round counter so new haggling
        # starts fresh, not on round 3+ from a different product.
        if target.get("post_id") != state.current_product_id:
            sess.reset_negotiation(req.session_id)
            state.current_product_id = target.get("post_id")

        offer = extract_offer(req.message)
        state.negotiation_round += 1
        state.last_offer = offer if offer is not None else state.last_offer

        outcome = negotiate(
            listed_price=int(target.get("priceInt") or 0),
            min_price=int(target.get("minPrice") or 0),
            buyer_offer=offer,
            round_num=state.negotiation_round,
        )

        negotiation_payload = NegotiationResult(
            decision=outcome.decision,
            listed_price=int(target.get("priceInt") or 0),
            buyer_offer=offer,
            counter_price=outcome.counter_price,
            min_price=int(target.get("minPrice") or 0),
            round_num=state.negotiation_round,
            product_id=target.get("post_id", ""),
        )

        # Deal accepted → deterministic reply with seller contact, no LLM round-trip.
        if outcome.decision == "accept":
            agreed = outcome.counter_price or int(target.get("priceInt") or 0)
            state.deal_accepted = True
            state.deal_price = agreed
            state.deal_product_id = target.get("post_id")
            reply = _deal_confirmation_reply(target, agreed)
            sess.append(req.session_id, "assistant", reply)
            return ChatResponse(
                reply=reply,
                intent=intent,
                products=_to_lite([target]),
                negotiation=negotiation_payload,
                notify_seller=True,
            )
        # Counter offer — deterministic phrasing from the engine, no LLM call needed.
        state.current_product_id = target.get("post_id")
        reply = outcome.message_hint
        sess.append(req.session_id, "assistant", reply)
        return ChatResponse(
            reply=reply,
            intent=intent,
            products=_to_lite([target]),
            negotiation=negotiation_payload,
            notify_seller=False,
        )

    # PRODUCT_QUERY — hand off to the LangChain tool-calling agent.
    # Track the focal product for follow-ups. Only reset negotiation when focus changes.
    if retrieved:
        new_focus = retrieved[0].get("post_id")
        if new_focus != state.current_product_id:
            state.current_product_id = new_focus
            sess.reset_negotiation(req.session_id)

    deal_was_accepted_before = state.deal_accepted
    reply = run_agent(CATALOG, state, req.message)

    # Hard guardrail: never let an LLM-leaked refusal slip through.
    if reply.strip().lower().startswith("i can only share information"):
        reply = OFF_TOPIC_REPLY

    # If the agent's negotiate_price tool just closed a deal, append guaranteed contact info.
    notify_flag = False
    if state.deal_accepted and not deal_was_accepted_before:
        focal = CATALOG.get(state.deal_product_id) if state.deal_product_id else None
        if focal is not None:
            reply = (
                f"{reply}\n\n"
                f"I've notified the seller — they'll reach out shortly. "
                f"You can also contact them directly:\n\n"
                f"{_contact_block(focal)}"
            )
            notify_flag = True

    sess.append(req.session_id, "assistant", reply)

    # Surface the focal product to the UI as a chip.
    focal_for_ui = []
    if state.current_product_id:
        p = CATALOG.get(state.current_product_id)
        if p:
            focal_for_ui = [p]
    if not focal_for_ui:
        focal_for_ui = retrieved[:3]

    return ChatResponse(
        reply=reply,
        intent=intent,
        products=_to_lite(focal_for_ui),
        negotiation=None,
        notify_seller=notify_flag,
    )
