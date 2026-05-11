"""Ingest data/products_enriched.json into a local PostgreSQL database and
store a sentence-embedding for each product.

Plain-vanilla PostgreSQL — NO pgvector extension required. The embedding is
stored in a `double precision[]` array column. Similarity search is done in
Python (numpy cosine) at query time — fine for small catalogs (< a few thousand
rows). If the catalog grows large, switch to the pgvector extension later.

Prereqs:
  - PostgreSQL running locally (you installed it natively).
  - A database created:  psql -U postgres -c "CREATE DATABASE quickvyapar;"
  - pip install psycopg2-binary sentence-transformers
  - DATABASE_URL in .env, e.g.
      DATABASE_URL=postgresql://postgres:YOURPASSWORD@localhost:5432/quickvyapar

Run:  python -m scripts.ingest_to_postgres
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip()
DATA_FILE = ROOT / "data" / "products_enriched.json"

DDL = """
CREATE TABLE IF NOT EXISTS products (
    post_id          TEXT PRIMARY KEY,
    title            TEXT,
    description      TEXT,
    category_name    TEXT,
    price_int        INTEGER,
    price_unit_type  TEXT,
    quantity         TEXT,
    quantity_unit    TEXT,
    min_price        INTEGER,
    max_discount_pct INTEGER,
    user_name        TEXT,
    post_company     TEXT,
    phone_number     TEXT,
    whatsapp_number  TEXT,
    location         TEXT,
    district_name    TEXT,
    state_name       TEXT,
    images           JSONB,
    raw              JSONB,
    embedding        DOUBLE PRECISION[],   -- sentence embedding, no extension needed
    embedding_dim    INTEGER
);
"""

UPSERT = """
INSERT INTO products (
    post_id, title, description, category_name, price_int, price_unit_type,
    quantity, quantity_unit, min_price, max_discount_pct, user_name,
    post_company, phone_number, whatsapp_number, location, district_name,
    state_name, images, raw, embedding, embedding_dim
) VALUES (
    %s,%s,%s,%s,%s,%s, %s,%s,%s,%s,%s, %s,%s,%s,%s,%s, %s,%s,%s,%s,%s
)
ON CONFLICT (post_id) DO UPDATE SET
    title=EXCLUDED.title, description=EXCLUDED.description,
    category_name=EXCLUDED.category_name, price_int=EXCLUDED.price_int,
    price_unit_type=EXCLUDED.price_unit_type, quantity=EXCLUDED.quantity,
    quantity_unit=EXCLUDED.quantity_unit, min_price=EXCLUDED.min_price,
    max_discount_pct=EXCLUDED.max_discount_pct, user_name=EXCLUDED.user_name,
    post_company=EXCLUDED.post_company, phone_number=EXCLUDED.phone_number,
    whatsapp_number=EXCLUDED.whatsapp_number, location=EXCLUDED.location,
    district_name=EXCLUDED.district_name, state_name=EXCLUDED.state_name,
    images=EXCLUDED.images, raw=EXCLUDED.raw, embedding=EXCLUDED.embedding,
    embedding_dim=EXCLUDED.embedding_dim
"""


def _searchable_text(p: dict) -> str:
    parts = [
        p.get("title", ""),
        p.get("categoryName", ""),
        p.get("description", ""),
        p.get("postCompany", ""),
        p.get("user_name", ""),
        f"price {p.get('priceInt', p.get('price'))}",
        p.get("location", ""),
        p.get("districtName", ""),
        p.get("stateName", ""),
    ]
    return " | ".join(str(x).strip() for x in parts if str(x).strip())


def main() -> None:
    if not DATABASE_URL:
        sys.exit("DATABASE_URL not set. Add it to .env:\n"
                 "  DATABASE_URL=postgresql://postgres:YOURPASSWORD@localhost:5432/quickvyapar")
    if not DATA_FILE.exists():
        sys.exit(f"{DATA_FILE} missing. Run first: python -m scripts.enrich_products")

    import psycopg2
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {EMBED_MODEL} (first run downloads ~80 MB)...")
    model = SentenceTransformer(EMBED_MODEL)
    dim = model.get_sentence_embedding_dimension()
    print(f"Model loaded — embedding dimension = {dim}")

    products = json.loads(DATA_FILE.read_text(encoding="utf-8")).get("data", [])
    print(f"Loaded {len(products)} products from {DATA_FILE.name}")

    texts = [_searchable_text(p) for p in products]
    print("Computing embeddings...")
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    target = DATABASE_URL.split("@")[-1]
    print(f"Connecting to {target} ...")
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            cur.execute(DDL)
            for p, vec in zip(products, vectors):
                cur.execute(UPSERT, (
                    p.get("post_id"),
                    (p.get("title") or "").strip(),
                    (p.get("description") or "").strip(),
                    (p.get("categoryName") or "").strip(),
                    int(p.get("priceInt") or p.get("price") or 0),
                    (p.get("priceUnitType") or "").strip(),
                    str(p.get("quantity") or ""),
                    (p.get("quantityUnitType") or "").strip(),
                    int(p.get("minPrice") or 0),
                    int(p.get("maxDiscountPct") or 0),
                    (p.get("user_name") or "").strip(),
                    (p.get("postCompany") or "").strip(),
                    (p.get("phoneNumber") or "").strip(),
                    (p.get("whatsappNumber") or "").strip(),
                    (p.get("location") or "").strip(),
                    (p.get("districtName") or "").strip(),
                    (p.get("stateName") or "").strip(),
                    json.dumps(p.get("images") or []),
                    json.dumps(p),
                    [float(x) for x in vec.tolist()],
                    int(dim),
                ))
        conn.commit()
        print(f"Upserted {len(products)} products into table `products`.")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    # Sanity check — cosine similarity in Python.
    print("\nSanity check — top-3 nearest to 'cotton round neck tshirt':")
    import numpy as np
    qv = model.encode(["cotton round neck tshirt"], normalize_embeddings=True)[0]
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT title, price_int, embedding FROM products")
            rows = cur.fetchall()
        scored = []
        for title, price, emb in rows:
            v = np.asarray(emb, dtype=float)
            sim = float(np.dot(qv, v))  # both normalized -> dot == cosine
            scored.append((sim, price, title))
        scored.sort(reverse=True)
        for sim, price, title in scored[:3]:
            print(f"  {sim:.3f}  Rs.{price:<6} {title}")
    finally:
        conn.close()

    print("\nDone. Data + embeddings stored in PostgreSQL (no pgvector needed).")


if __name__ == "__main__":
    main()
