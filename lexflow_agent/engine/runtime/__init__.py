"""超时链管理"""

from __future__ import annotations

import asyncio
import signal
from typing import Any, Callable, Dict

from lexflow_agent.config.settings import settings

TIMEOUT_CHAIN: dict[str, float] = {
    "workflow_total": settings.WORKFLOW_TIMEOUT_SECONDS,
    "agent_node": settings.NODE_TIMEOUT_SECONDS,
    "llm_call": settings.LLM_CALL_TIMEOUT_SECONDS,
}


async def execute_with_timeout(coro: Any, timeout: float) -> Any:
    """带超时的异步执行"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(f"执行超时 ({timeout}s)")


class GracefulShutdown:
    """优雅关闭管理器"""

    def __init__(self, wait_seconds: int = 30):
        self.wait = wait_seconds or settings.GRACEFUL_SHUTDOWN_WAIT
        self._shutdown_requested = False
        self._active_tasks: Dict[str, asyncio.Task] = {}

    async def shutdown(self, sig: signal.Signals):
        self._shutdown_requested = True

        pending = list(self._active_tasks.values())
        if pending:
            _, pending = await asyncio.wait(
                pending, timeout=self.wait,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()

    def is_shutting_down(self) -> bool:
        return self._shutdown_requested


graceful_shutdown = GracefulShutdown()


class ConcurrencyController:
    """系统级并发控制"""

    def __init__(self, max_concurrent: int = 5):
        self.max = max_concurrent or settings.MAX_CONCURRENT_TASKS
        self._semaphore = asyncio.Semaphore(self.max)
        self._active_lawyer: Dict[str, str] = {}

    async def acquire(self, lawyer_id: str) -> bool:
        if lawyer_id in self._active_lawyer:
            return False
        await self._semaphore.acquire()
        self._active_lawyer[lawyer_id] = "active"
        return True

    def release(self, lawyer_id: str):
        self._active_lawyer.pop(lawyer_id, None)
        self._semaphore.release()


concurrency_controller = ConcurrencyController()
