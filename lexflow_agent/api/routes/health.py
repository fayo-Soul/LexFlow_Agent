"""健康检查路由"""

from __future__ import annotations

from fastapi import APIRouter
from lexflow_agent.engine.tools.law_client import law_client

router = APIRouter()


@router.get("/health")
async def health():
    """健康检查"""
    import os
    ls_tracing = os.environ.get("LANGSMITH_TRACING", "false")
    ls_project = os.environ.get("LANGSMITH_PROJECT", "")
    return {
        "status": "healthy",
        "service": "lexflow-agent",
        "langsmith": {
            "tracing": ls_tracing == "true",
            "project": ls_project,
        },
    }


@router.get("/ready")
async def readiness():
    """就绪检查 - 检查外部依赖"""
    law_ok = law_client.health()
    if not law_ok:
        return {"status": "degraded", "law_rag": "unavailable"}
    return {"status": "ready", "law_rag": "ok"}
