import re
from typing import Optional

# Keep this module dependency-free of LLM so it stays cheap. The LLM fallback
# lives in main.py if we ever need it; rule pass handles 95%+ of real traffic.

PRODUCT_KEYWORDS = {
    "product", "products", "tshirt", "t-shirt", "shirt", "kurti", "abaya",
    "jacket", "pants", "trouser", "dupatta", "sweatshirt", "fabric", "cotton",
    "polyester", "polyster", "size", "sizes", "stock", "available", "availability",
    "ship", "shipping", "delivery", "color", "colour", "quantity", "moq",
    "category", "categories", "rate", "price", "cost", "buy", "order",
    "wholesale", "retail", "piece", "pcs", "discount", "offer", "deal",
    "show", "list", "what", "which", "have", "sell",
}

NEGOTIATE_PATTERNS = [
    r"\b(can|could|will|would)\s+(you|u)\s+(give|do|sell|offer)\b",
    r"\bbest\s+price\b",
    r"\bfinal\s+price\b",
    r"\bdiscount\b",
    r"\bnegotiat",
    r"\bkam\s+kar",  # hindi
    r"\bsasta\b",
    r"\b(₹|rs\.?|rupees?)\s*\d+",
    r"\bfor\s+₹?\s*\d+\b",
    r"\bat\s+₹?\s*\d+\b",
]

REQUEST_HUMAN_PATTERNS = [
    r"\b(talk|speak|connect|call)\s+(to\s+)?(the\s+)?(seller|owner|human|person|agent)\b",
    r"\breal\s+(seller|person)\b",
    r"\bcall\s+seller\b",
]

OFF_TOPIC_KEYWORDS = {
    "prime minister", "president", "weather", "joke", "poem", "story",
    "who invented", "who made you", "who created", "who built you",
    "your name", "are you ai", "are you human", "chatgpt", "gpt",
    "code", "python", "javascript", "math", "calculate", "capital of",
    "movie", "song", "cricket", "football", "election", "stock market",
    "bitcoin", "crypto", "news", "modi", "rahul gandhi", "trump", "biden",
    "putin", "xi jinping", "obama", "covid", "vaccine",
}

OFF_TOPIC_PATTERNS = [
    r"\bhow\s+(are|r)\s+(you|u|ya)\b",
    r"\bkaise\s+(ho|hain|hai)\b",
    r"\bkya\s+haal\b",
    r"\bwhat'?s\s+up\b",
    r"\bwassup\b",
    r"\b(tell|say)\s+(me\s+)?(a\s+)?(joke|story|poem)\b",
    # politicians / leaders — accept with or without "the"
    r"\bwho\s+(is|are)\s+(the\s+)?(prime|president|ceo|pm|cm|king|queen|chairman)\b",
    r"\bwho\s+(is|are)\s+(the\s+)?(pm|cm|president|prime\s+minister)\s+of\b",
    r"\bpm\s+of\s+\w+",
    r"\bcm\s+of\s+\w+",
    r"\bpresident\s+of\s+\w+",
    r"\b(india|pakistan|usa|america|china|russia|uk|britain)\b.*(pm|cm|president|leader|government)",
    r"\b(pm|cm|president|leader)\b.*(india|pakistan|usa|america|china|russia|uk|britain)",
    # math
    r"\b\d+\s*[+\-*/x×]\s*\d+\b",
    r"\b(calculate|compute|solve|equation)\b",
    # capital cities
    r"\bcapital\s+(of|city)\b",
]

GREETING_PATTERNS = [
    r"^\s*(hi|hello|hey|namaste|hii+|helo|good\s+(morning|evening|afternoon))\b",
]

FAREWELL_PATTERNS = [
    r"^\s*(thanks?|thank\s*you|thx|ty|tysm|thanku|shukriya|dhanyavad)\b\s*[.!]*\s*$",
    r"^\s*(thanks?|thank\s*you)[\s,!.]*(so\s+much|a\s+lot|very\s+much|bhai|bro|sir|madam)?[\s,!.]*$",
    r"\b(bye|goodbye|good\s*bye|see\s*you|cya|alvida|ok\s*bye|tata)\b",
    r"\b(that'?s?\s+(all|it|enough))\b",
    r"\bno\s+(thanks?|more\s+questions?)\b",
    r"\bi\s+am\s+(satisfied|done|good|ok)\b",
    r"\bi'?m\s+(satisfied|done|good|ok)\b",
    r"\b(i\s+will\s+)?(buy|order|purchase)\s+(it|this|now|later)\b",
    r"\bdeal\s+done\b",
    r"\b(got\s+it|understood),?\s*(thanks?|thank\s*you)\b",
]


_PRICE_RE = re.compile(r"(?:₹|rs\.?|rupees?)?\s*(\d{2,6})\s*(?:rs|rupees?|/-)?", re.IGNORECASE)


def extract_offer(text: str) -> Optional[int]:
    """Pull a buyer offer (a price number) from a negotiation utterance."""
    matches = _PRICE_RE.findall(text or "")
    if not matches:
        return None
    nums = [int(m) for m in matches if 10 <= int(m) <= 100000]
    if not nums:
        return None
    # Heuristic: smallest number is usually the offer (listed prices tend to be larger).
    return min(nums)


def classify(text: str) -> str:
    t = (text or "").lower().strip()
    if not t:
        return "OFF_TOPIC"

    for pat in FAREWELL_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            # If farewell phrase also clearly asks for more product info, skip.
            if any(k in t for k in {"price", "rate", "available", "stock", "size"}):
                break
            return "FAREWELL"

    for pat in GREETING_PATTERNS:
        if re.search(pat, t):
            # If greeting also asks something product-y, fall through.
            if any(k in t for k in PRODUCT_KEYWORDS):
                break
            return "GREETING"

    for pat in REQUEST_HUMAN_PATTERNS:
        if re.search(pat, t):
            return "REQUEST_HUMAN"

    for pat in NEGOTIATE_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            return "NEGOTIATE"

    for pat in OFF_TOPIC_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            return "OFF_TOPIC"

    if any(k in t for k in OFF_TOPIC_KEYWORDS):
        # Product-related signal wins (avoid false positives).
        if not any(k in t for k in PRODUCT_KEYWORDS):
            return "OFF_TOPIC"

    if any(k in t for k in PRODUCT_KEYWORDS):
        return "PRODUCT_QUERY"

    # Ambiguous short utterance — treat as product query so retrieval can try.
    # Generous threshold for "tell me about <product name>" type phrasings.
    if len(t.split()) <= 6:
        return "PRODUCT_QUERY"

    return "OFF_TOPIC"
