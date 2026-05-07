"""One-time: read products.json, write data/products_enriched.json with minPrice & maxDiscountPct per product."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "products.json"
OUT = ROOT / "data" / "products_enriched.json"

DEFAULT_DISCOUNT_PCT = 15


def to_int_price(raw) -> int:
    try:
        return int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return 0


def enrich(product: dict) -> dict:
    price = to_int_price(product.get("price"))
    min_price = round(price * (1 - DEFAULT_DISCOUNT_PCT / 100)) if price else 0
    product["priceInt"] = price
    product["minPrice"] = min_price
    product["maxDiscountPct"] = DEFAULT_DISCOUNT_PCT
    return product


def main() -> None:
    raw = json.loads(SRC.read_text(encoding="utf-8"))
    items = raw.get("data", [])
    enriched = [enrich(dict(p)) for p in items]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"data": enriched}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {len(enriched)} products -> {OUT}")


if __name__ == "__main__":
    main()
