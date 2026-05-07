"""Legacy LLM wrapper. The main chat flow now uses backend.agent.run_agent().
Kept for parity / external callers — mirrors the same multi-tier fallback chain.
"""
import logging
from typing import List, Dict

from .config import (
    GROQ_API_KEY, GEMINI_API_KEY, CEREBRAS_API_KEY,
    OPENROUTER_API_KEY, MISTRAL_API_KEY,
    GROQ_MODEL, GEMINI_MODEL, CEREBRAS_MODEL,
    OPENROUTER_MODEL, MISTRAL_MODEL,
)

logger = logging.getLogger(__name__)

_chain_cache = None


def _build_chain():
    """Build LangChain model chain mirroring agent.py order:
    Groq → Cerebras → Gemini → OpenRouter → Mistral.
    """
    global _chain_cache
    if _chain_cache is not None:
        return _chain_cache

    chain: list = []

    if GROQ_API_KEY:
        try:
            from langchain_groq import ChatGroq
            chain.append(ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY,
                                  temperature=0.3, max_tokens=400, timeout=15))
        except Exception as e:
            logger.warning("Groq init failed: %s", e)

    if CEREBRAS_API_KEY:
        try:
            from langchain_cerebras import ChatCerebras
            chain.append(ChatCerebras(model=CEREBRAS_MODEL, api_key=CEREBRAS_API_KEY,
                                      temperature=0.3, max_tokens=400, timeout=15))
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
        _chain_cache = None
        return None
    primary = chain[0]
    _chain_cache = primary if len(chain) == 1 else primary.with_fallbacks(chain[1:])
    return _chain_cache


def generate(messages: List[Dict[str, str]], temperature: float = 0.3) -> str:
    """Run a chat completion through the fallback chain. Returns plain text reply."""
    chain = _build_chain()
    if chain is None:
        logger.error("No LLM providers configured.")
        return (
            "I'm having trouble reaching my assistant right now. "
            "Please try again in a moment, or tap 'Talk to seller'."
        )
    try:
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        lc_msgs = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                lc_msgs.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_msgs.append(AIMessage(content=content))
            else:
                lc_msgs.append(HumanMessage(content=content))
        resp = chain.invoke(lc_msgs)
        return getattr(resp, "content", str(resp)).strip()
    except Exception as e:
        logger.warning("All LLM tiers failed: %s", e)
        return (
            "I'm having trouble reaching my assistant right now. "
            "Please try again in a moment, or tap 'Talk to seller'."
        )
