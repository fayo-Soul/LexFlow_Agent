#!/usr/bin/env python3
"""Final non-mutating runtime security and deletion-residue checks."""

from __future__ import annotations

import sqlite3
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lexflow_agent.config.settings import settings
from lexflow_agent.engine.run_store import run_store


def status(headers=None):
    request = urllib.request.Request(
        "http://127.0.0.1:8000/api/v1/case-runs/not-found",
        headers=headers or {},
    )
    try:
        urllib.request.urlopen(request, timeout=10)
    except urllib.error.HTTPError as error:
        return error.code
    return 200


def main():
    checkpoint_path = Path(settings.CHECKPOINT_DB_URL.removeprefix("sqlite:///"))
    with sqlite3.connect(checkpoint_path) as connection:
        checkpoint_rows = connection.execute(
            "SELECT count(*) FROM checkpoints WHERE thread_id LIKE ?",
            ("run_global_restart_check%",),
        ).fetchone()[0]
    with sqlite3.connect(run_store.path) as connection:
        registry_rows = connection.execute(
            "SELECT count(*) FROM runs WHERE run_id LIKE ?",
            ("run_global_restart_check%",),
        ).fetchone()[0]
    result = {
        "unauthorized_http": status(),
        "authorized_missing_http": status({
            "Authorization": f"Bearer {settings.API_TOKEN}",
        }),
        "checkpoint_rows": checkpoint_rows,
        "registry_rows": registry_rows,
    }
    assert result == {
        "unauthorized_http": 401,
        "authorized_missing_http": 404,
        "checkpoint_rows": 0,
        "registry_rows": 0,
    }
    print(result)


if __name__ == "__main__":
    main()
