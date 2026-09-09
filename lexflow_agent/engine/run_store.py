"""Durable API run registry stored separately from LangGraph checkpoints."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from lexflow_agent.config.settings import settings
from lexflow_agent.engine.state import CaseAgentState


def _database_path() -> Path:
    checkpoint = Path(settings.CHECKPOINT_DB_URL.removeprefix("sqlite:///"))
    return checkpoint.with_name(f"{checkpoint.stem}.runs.db")


class RunStore:
    def __init__(self):
        path = _database_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS runs "
                "(run_id TEXT PRIMARY KEY, state_json TEXT NOT NULL, idempotency_key TEXT)"
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_idempotency "
                "ON runs(idempotency_key) WHERE idempotency_key IS NOT NULL"
            )

    def _connect(self):
        return sqlite3.connect(self.path)

    def save(self, state: CaseAgentState) -> None:
        payload = state.model_dump_json()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO runs(run_id, state_json, idempotency_key) VALUES(?, ?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET "
                "state_json=excluded.state_json, idempotency_key=excluded.idempotency_key",
                (state.run_id, payload, state.idempotency_key),
            )

    def load_all(self) -> dict[str, CaseAgentState]:
        with self._connect() as connection:
            rows = connection.execute("SELECT run_id, state_json FROM runs").fetchall()
        return {run_id: CaseAgentState.model_validate_json(payload) for run_id, payload in rows}

    def delete(self, run_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))


run_store = RunStore()
