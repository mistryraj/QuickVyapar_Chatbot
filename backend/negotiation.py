from dataclasses import dataclass
from typing import Optional


@dataclass
class NegotiationOutcome:
    decision: str  # accept | counter | reject
    counter_price: Optional[int]
    message_hint: str


def _midpoint(a: int, b: int) -> int:
    return round((a + b) / 2)


def negotiate(
    listed_price: int,
    min_price: int,
    buyer_offer: Optional[int],
    round_num: int,
) -> NegotiationOutcome:
    """Deterministic negotiation. Returns decision + counter price + a phrasing hint.

    Rules:
    - offer >= listed: accept at listed.
    - offer >= min_price: accept at offer.
    - offer < min_price:
        round 1 -> counter at min(listed*0.95, midpoint(offer, listed))
        round 2 -> counter at min(listed*0.92, midpoint(offer, listed))
        round 3+ -> final at min_price.
    - If buyer_offer is None (asking generic discount), give round-based teaser without committing below min_price.
    """
    if listed_price <= 0:
        return NegotiationOutcome("reject", None, "Price not available for this product.")

    # Generic "can you give discount?" — no concrete number from buyer.
    if buyer_offer is None:
        if round_num >= 2:
            return NegotiationOutcome(
                "counter",
                min_price,
                f"Best final price is ₹{min_price}. That is the lowest the seller can offer.",
            )
        teaser = max(min_price, round(listed_price * 0.95))
        return NegotiationOutcome(
            "counter",
            teaser,
            f"I can offer ₹{teaser} as a small discount from ₹{listed_price}. Let me know if that works.",
        )

    if buyer_offer >= listed_price:
        return NegotiationOutcome(
            "accept",
            listed_price,
            f"Deal confirmed at ₹{listed_price}.",
        )

    if buyer_offer >= min_price:
        return NegotiationOutcome(
            "accept",
            buyer_offer,
            f"Deal confirmed at ₹{buyer_offer}.",
        )

    # Below the floor.
    if round_num <= 1:
        target = min(round(listed_price * 0.95), _midpoint(buyer_offer, listed_price))
        target = max(target, min_price)
        return NegotiationOutcome(
            "counter",
            target,
            f"₹{buyer_offer} is below what the seller can accept. How about ₹{target}?",
        )
    if round_num == 2:
        target = min(round(listed_price * 0.92), _midpoint(buyer_offer, listed_price))
        target = max(target, min_price)
        return NegotiationOutcome(
            "counter",
            target,
            f"I can come down to ₹{target}, but not lower right now.",
        )
    return NegotiationOutcome(
        "counter",
        min_price,
        f"Best final price is ₹{min_price}. That is the lowest the seller can offer.",
    )
