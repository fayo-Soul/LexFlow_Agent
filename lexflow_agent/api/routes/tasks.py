from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from lexflow_agent.api.routes.case_runs import _check_auth
from lexflow_agent.engine.models.response import ApiResponse
from lexflow_agent.engine.orchestrator.models import (
    AgentType,
    AgentRequest,
    ExecutionMode,
    TaskRequest,
)
from lexflow_agent.engine.orchestrator.service import get_orchestrator


router = APIRouter(dependencies=[Depends(_check_auth)])


async def _execute(request: TaskRequest) -> ApiResponse:
    """兼容旧 /tasks 入口 — 翻译为 AgentRequest → Orchestrator.handle()"""
    agent_req = AgentRequest(
        case_id=request.case_id,
        agent_type=request.requested_agent or AgentType.LEGAL_RESEARCH,
        execution_mode=request.mode,
        message=request.message,
        case_type=request.case_type,
        client_role=request.client_role,
        documents=request.documents,
        context=request.context,
        idempotency_key=request.idempotency_key,
    )
    orchestrator = get_orchestrator()
    result = await asyncio.to_thread(orchestrator.handle, agent_req)
    return ApiResponse(
        status="success",
        data=result.model_dump(mode="json") if hasattr(result, "model_dump") else {"result": str(result)},
    )


@router.post("/tasks")
async def create_task(request: TaskRequest):
    """Unified task entry — 兼容旧接口"""
    return await _execute(request)


def _single_agent_request(request: TaskRequest, agent: AgentType) -> TaskRequest:
    return request.model_copy(update={
        "mode": ExecutionMode.SINGLE,
        "requested_agent": agent,
    })


@router.post("/agents/case-understanding/runs")
async def run_case_understanding(request: TaskRequest):
    return await _execute(_single_agent_request(request, AgentType.CASE_UNDERSTANDING))


@router.post("/agents/legal-research/runs")
async def run_legal_research(request: TaskRequest):
    return await _execute(_single_agent_request(request, AgentType.LEGAL_RESEARCH))


@router.post("/agents/case-strategy/runs")
async def run_case_strategy(request: TaskRequest):
    return await _execute(_single_agent_request(request, AgentType.CASE_STRATEGY))


@router.post("/agents/document-generation/runs")
async def run_document_generation(request: TaskRequest):
    return await _execute(_single_agent_request(request, AgentType.DOCUMENT_GENERATION))


@router.post("/agents/legal-consult/runs")
async def run_legal_consult(request: TaskRequest):
    """法律咨询快捷入口 — 已收编：映射到法律研究 Agent 的咨询模式（RAG 检索 + 多轮历史）"""
    return await _execute(_single_agent_request(request, AgentType.LEGAL_RESEARCH))
