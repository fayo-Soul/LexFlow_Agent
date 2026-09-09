"""Session store — lightweight in-memory store for run context.

Used by unified_entry to persist AgentRequest across clarification rounds.
"""

from __future__ import annotations

from typing import Any

# Simple in-memory dict — replace with Redis for production
_store: dict[str, dict[str, Any]] = {}


def save(run_id: str, data: dict[str, Any]) -> None:
    _store[run_id] = data


def get(run_id: str) -> dict[str, Any] | None:
    return _store.get(run_id)


def delete(run_id: str) -> None:
    _store.pop(run_id, None)


# Alias for clarity
session_store = type("SessionStore", (), {
    "get": staticmethod(get),
    "save": staticmethod(save),
    "delete": staticmethod(delete),
})()
