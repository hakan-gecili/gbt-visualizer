from __future__ import annotations

from typing import Dict

from app.domain.session_types import SessionState


class SessionStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, SessionState] = {}

    def save(self, session: SessionState) -> None:
        self._sessions[session.session_id] = session

    def get(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            raise KeyError(f"Session '{session_id}' was not found.")
        return self._sessions[session_id]


session_store = SessionStore()

