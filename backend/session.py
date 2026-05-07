from collections import defaultdict
from typing import Dict, List, Optional

from .config import MAX_HISTORY_TURNS


class SessionState:
    def __init__(self) -> None:
        self.history: List[dict] = []
        self.current_product_id: Optional[str] = None
        self.negotiation_round: int = 0
        self.last_offer: Optional[int] = None
        self.deal_accepted: bool = False
        self.deal_price: Optional[int] = None
        self.deal_product_id: Optional[str] = None


_sessions: Dict[str, SessionState] = defaultdict(SessionState)


def get(session_id: str) -> SessionState:
    return _sessions[session_id]


def append(session_id: str, role: str, content: str) -> None:
    state = _sessions[session_id]
    state.history.append({"role": role, "content": content})
    if len(state.history) > MAX_HISTORY_TURNS * 2:
        state.history = state.history[-MAX_HISTORY_TURNS * 2:]


def reset_negotiation(session_id: str) -> None:
    state = _sessions[session_id]
    state.negotiation_round = 0
    state.last_offer = None
