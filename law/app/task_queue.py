"""Durable Redis queue for document indexing jobs."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid


logger = logging.getLogger("index_task_queue")


class RedisTaskQueue:
    READY = "rag:index:ready"
    PROCESSING = "rag:index:processing"

    def __init__(self, redis_client, handler, max_retries: int = 3):
        self.redis = redis_client
        self.handler = handler
        self.max_retries = max_retries
        self._stop = threading.Event()
        self._thread = None

    def enqueue(self, payload: dict) -> str:
        job = {
            **payload,
            "job_id": uuid.uuid4().hex,
            "attempt": 0,
            "created_at": time.time(),
        }
        encoded = json.dumps(job, ensure_ascii=False)
        self.redis.hset(
            f"rag:index:job:{job['job_id']}",
            mapping={"status": "queued", "payload": encoded},
        )
        self.redis.expire(f"rag:index:job:{job['job_id']}", 7 * 86400)
        self.redis.lpush(self.READY, encoded)
        return job["job_id"]

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._recover()
        self._thread = threading.Thread(
            target=self._run,
            name="rag-index-worker",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def status(self, job_id: str) -> dict:
        return self.redis.hgetall(f"rag:index:job:{job_id}")

    def _recover(self) -> None:
        while True:
            encoded = self.redis.rpop(self.PROCESSING)
            if encoded is None:
                break
            self.redis.lpush(self.READY, encoded)

    def _run(self) -> None:
        while not self._stop.is_set():
            encoded = self.redis.brpoplpush(
                self.READY,
                self.PROCESSING,
                timeout=2,
            )
            if not encoded:
                continue
            job = json.loads(encoded)
            job_key = f"rag:index:job:{job['job_id']}"
            try:
                self.redis.hset(job_key, "status", "processing")
                self.handler(job)
                self.redis.hset(job_key, "status", "completed")
                self.redis.lrem(self.PROCESSING, 1, encoded)
            except Exception as exc:
                self.redis.lrem(self.PROCESSING, 1, encoded)
                job["attempt"] += 1
                if job["attempt"] < self.max_retries:
                    retry = json.dumps(job, ensure_ascii=False)
                    self.redis.hset(
                        job_key,
                        mapping={"status": "retrying", "payload": retry, "error": str(exc)},
                    )
                    self.redis.lpush(self.READY, retry)
                else:
                    self.redis.hset(
                        job_key,
                        mapping={"status": "failed", "error": str(exc)},
                    )
                logger.exception("索引任务失败: %s", job["job_id"])
