import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
_env = ROOT / ".env"
if _env.exists():
    load_dotenv(_env)
else:
    # Fallback so the app still works if the user filled .env.example directly.
    load_dotenv(ROOT / ".env.example")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "").strip()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "llama-3.3-70b").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct:free").strip()
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest").strip()

DATA_FILE = ROOT / "data" / "products_enriched.json"
RAW_PRODUCTS_FILE = ROOT / "products.json"

MAX_HISTORY_TURNS = 10
LLM_TIMEOUT_SECONDS = 8
