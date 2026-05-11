"""Optional PostgreSQL-backed semantic search over the product catalog.

Plain-vanilla PostgreSQL — embeddings are stored as DOUBLE PRECISION[] arrays
(see scripts/ingest_to_postgres.py); cosine similarity is computed in numpy at
query time. No pgvector extension required. Fine for small catalogs.

Opt-in: if DATABASE_URL is unset or the deps / table are unavailable, callers
fall back to the keyword Catalog.search().
"""
from __future__ import annotations

import logging
import json
from functools import lru_cache
from typing import List

from .config import DATABASE_URL, EMBED_MODEL

logger = logging.getLogger(__name__)


def available() -> bool:
    if not DATABASE_URL:
        return False
    try:
        import psycopg2  # noqa: F401
        import numpy  # noqa: F401
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _load_rows():
    """Load (raw_dict, numpy_vector) for every product once and cache it.
    Call _load_rows.cache_clear() after re-ingesting to refresh.
    """
    import numpy as np
    import psycopg2
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT raw, embedding FROM products WHERE embedding IS NOT NULL")
            rows = cur.fetchall()
    finally:
        conn.close()
    out = []
    for raw, emb in rows:
        d = raw if isinstance(raw, dict) else json.loads(raw)
        v = np.asarray(emb, dtype=float)
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
        out.append((d, v))
    return out


def search_similar(query: str, k: int = 5) -> List[dict]:
    """Top-k product dicts by cosine similarity. Returns [] on any failure so
    callers can fall back to keyword search."""
    if not query or not available():
        return []
    try:
        import numpy as np
        rows = _load_rows()
        if not rows:
            return []
        qv = _model().encode([query], normalize_embeddings=True)[0]
        qv = np.asarray(qv, dtype=float)
        scored = [(float(np.dot(qv, v)), d) for d, v in rows]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:k]]
    except Exception as e:
        logger.warning("pg semantic search failed (%s) — falling back to keyword.", e)
        return []
