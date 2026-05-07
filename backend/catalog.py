import json
import re
from typing import List, Optional

from .config import DATA_FILE


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set:
    return set(_TOKEN_RE.findall((text or "").lower()))


class Catalog:
    def __init__(self, products: List[dict]) -> None:
        self.products = products
        self._index = [
            (p, _tokenize(" ".join([
                p.get("title", ""),
                p.get("description", ""),
                p.get("categoryName", ""),
                p.get("postCompany", ""),
                p.get("user_name", ""),
            ])))
            for p in products
        ]
        self._by_id = {p["post_id"]: p for p in products if p.get("post_id")}

    def all(self) -> List[dict]:
        return self.products

    def get(self, post_id: str) -> Optional[dict]:
        return self._by_id.get(post_id)

    def search(self, query: str, k: int = 5) -> List[dict]:
        q_tokens = _tokenize(query)
        if not q_tokens:
            return self.products[:k]
        scored = []
        for product, tokens in self._index:
            overlap = len(q_tokens & tokens)
            if overlap == 0:
                continue
            scored.append((overlap, product))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:k]]

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
