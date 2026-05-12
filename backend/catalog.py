import json
import re
from typing import List, Optional

from .config import DATA_FILE


_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Words that carry no product signal. Stripped from a *query* before scoring so
# junk like "tell me about the iphone 15" doesn't match a random product just
# because "about"/"the" happen to appear in some description. Catalog-side tokens
# are left intact.
_QUERY_STOPWORDS = {
    "a", "an", "the", "of", "for", "to", "in", "on", "with", "and", "or",
    "is", "are", "it", "this", "that", "these", "those", "be", "do", "does",
    "did", "have", "has", "had", "you", "your", "u", "i", "me", "my", "we",
    "tell", "show", "give", "get", "want", "need", "please", "looking", "look",
    "about", "more", "info", "information", "detail", "details", "describe",
    "what", "which", "who", "how", "where", "when", "any", "some", "all",
    "can", "could", "would", "will", "should", "hi", "hello", "hey", "there",
    "kya", "hai", "ka", "ki", "ke", "ko", "mujhe", "muje", "bata", "batao",
}


def _tokenize(text: str) -> set:
    return set(_TOKEN_RE.findall((text or "").lower()))


class Catalog:
    def __init__(self, products: List[dict]) -> None:
        self.products = products
        # Per product keep separate token sets so title matches can be weighted
        # higher than description / category matches during search.
        self._index = []
        for p in products:
            title_tok = _tokenize(p.get("title", ""))
            other_tok = _tokenize(" ".join([
                p.get("description", ""),
                p.get("categoryName", ""),
                p.get("postCompany", ""),
                p.get("user_name", ""),
            ]))
            self._index.append((p, title_tok, other_tok))
        self._by_id = {p["post_id"]: p for p in products if p.get("post_id")}

    def all(self) -> List[dict]:
        return self.products

    def get(self, post_id: str) -> Optional[dict]:
        return self._by_id.get(post_id)

    def search(self, query: str, k: int = 5) -> List[dict]:
        raw = _tokenize(query)
        if not raw:
            return self.products[:k]
        q_tokens = raw - _QUERY_STOPWORDS
        if not q_tokens:
            # Query was nothing but stopwords ("tell me about it"). No real
            # signal — return nothing so callers fall back to focal product or
            # an honest "not in catalog" instead of matching a random item.
            return []
        scored = []
        for product, title_tok, other_tok in self._index:
            title_hits = len(q_tokens & title_tok)
            other_hits = len(q_tokens & other_tok)
            if title_hits == 0 and other_hits == 0:
                continue
            # Title hits weigh 3x; tie-break by total hits. So a product whose
            # *title* is "Round Neck" beats one that only has "Round Neck" in
            # its category name.
            score = title_hits * 3 + other_hits
            scored.append((score, title_hits, product))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [p for _, _, p in scored[:k]]

    def summary_for_llm(self, products: List[dict]) -> str:
        lines = []
        for p in products:
            lines.append(
                f"- post_id={p.get('post_id')} | title={p.get('title','').strip()} "
                f"| price=₹{p.get('priceInt', p.get('price'))} {p.get('priceUnitType','')} "
                f"| min_price=₹{p.get('minPrice')} "
                f"| qty={p.get('quantity')} {p.get('quantityUnitType','')} "
                f"| seller={p.get('user_name','').strip()} "
                f"| category={p.get('categoryName','').strip()} "
                f"| location={p.get('location','')}, {p.get('districtName','')}, {p.get('stateName','')} "
                f"| description={(p.get('description','') or '').strip()[:300]}"
            )
        return "\n".join(lines) if lines else "(no matching products)"


def load_catalog() -> Catalog:
    raw = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    return Catalog(raw.get("data", []))
