"""不记录业务正文的结构化审计日志。"""

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


class AuditLogger:
    def __init__(self, path: str | Path = "logs/audit.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def record(
        self,
        action: str,
        actor: str,
        target: str = "",
        trace_id: str = "",
        result: str = "success",
    ) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "actor": actor,
            "target": target,
            "trace_id": trace_id,
            "result": result,
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
