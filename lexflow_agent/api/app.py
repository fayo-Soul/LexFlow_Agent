"""FastAPI 应用主入口 — v2 版本（对标 EduAgent 统一入口模式）

路由表：
/api/v1/chat/stream          — unified_entry SSE 统一入口
/api/v1/tasks                 — 兼容旧 /tasks 入口
/api/v1/agents/{name}/runs    — 单 Agent 快捷入口
/api/v1/case-runs             — 原 case-runs CRUD (保留)
/api/v1/case-runs/{id}/resume — 恢复 + 澄清回答 (扩展)
/api/v1/health                — 健康检查
"""

from __future__ import annotations

import asyncio
import os
import signal
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from lexflow_agent.config.settings import settings
from lexflow_agent.engine.gateway.factory import LLMFactory
from lexflow_agent.engine.runtime import graceful_shutdown
from lexflow_agent.api.routes.case_runs import router as case_router
from lexflow_agent.api.routes.case_runs import recover_incomplete_runs
from lexflow_agent.api.routes.health import router as health_router
from lexflow_agent.api.routes.tasks import router as task_router
from lexflow_agent.api.v1.unified_entry import UnifiedChatRequest, unified_chat_events
from lexflow_agent.api.v1.unified_entry import ClarificationSubmitRequest, handle_clarification


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动/关闭生命周期"""
    # 注入 LangSmith 环境变量
    if settings.LANGSMITH_TRACING:
        if settings.LANGSMITH_API_KEY:
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ.setdefault("LANGSMITH_ENDPOINT", settings.LANGSMITH_ENDPOINT)
            os.environ.setdefault("LANGSMITH_PROJECT", settings.LANGSMITH_PROJECT)
            os.environ.setdefault("LANGSMITH_API_KEY", settings.LANGSMITH_API_KEY)

    LLMFactory.initialize()
    await recover_incomplete_runs()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig,
            lambda s=sig: asyncio.create_task(graceful_shutdown.shutdown(s)),
        )
    yield
    await graceful_shutdown.shutdown(signal.SIGTERM)


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 路由注册 ─────────────────────────────────────────────
app.include_router(health_router, prefix="/api/v1", tags=["health"])
app.include_router(case_router, prefix="/api/v1", tags=["case-runs"])
app.include_router(task_router, prefix="/api/v1", tags=["tasks"])

# ★新增：SSE 统一入口
@app.post("/api/v1/chat/stream")
async def unified_chat(request: UnifiedChatRequest):
    """SSE 统一入口 — 对标 EduAgent unified_chat.py"""
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        unified_chat_events(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

# ★新增：澄清回答提交
@app.post("/api/v1/chat/clarify")
async def submit_clarification(request: ClarificationSubmitRequest):
    """提交澄清回答，继续执行原 Agent"""
    from lexflow_agent.engine.models.response import ApiResponse
    result = await asyncio.to_thread(handle_clarification, request)
    return ApiResponse(status="success", data={"result": result.model_dump() if hasattr(result, 'model_dump') else str(result)})
