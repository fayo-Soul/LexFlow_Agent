from __future__ import annotations

import re

from pydantic import BaseModel, Field

from lexflow_agent.engine.orchestrator.models import AgentName, TaskRequest


class PrecheckResult(BaseModel):
    passed: bool
    rejection_code: str | None = None
    message: str = ""
    missing_inputs: list[str] = Field(default_factory=list)


_DEPENDENCIES: dict[AgentName, tuple[str, ...]] = {
    AgentName.CASE_UNDERSTANDING: ("documents",),
    AgentName.LEGAL_RESEARCH: ("message",),
    AgentName.CASE_STRATEGY: ("facts", "legal_research"),
    AgentName.DOCUMENT_GENERATION: ("facts", "strategy"),
}


def check_request(request: TaskRequest) -> PrecheckResult:
    if re.search(r"<script|union\s+select|drop\s+table", request.message, re.IGNORECASE):
        return PrecheckResult(
            passed=False,
            rejection_code="MALICIOUS_INPUT",
            message="请求包含不允许的参数内容",
        )
    if len(request.documents) > 20:
        return PrecheckResult(
            passed=False,
            rejection_code="TOO_MANY_DOCUMENTS",
            message="单次请求最多支持 20 份材料",
        )
    return PrecheckResult(passed=True)


def check_agent_inputs(request: TaskRequest, agent: AgentName) -> PrecheckResult:
    missing: list[str] = []
    for field in _DEPENDENCIES[agent]:
        if field == "documents" and not request.documents:
            missing.append(field)
        elif field == "message" and not request.message.strip():
            missing.append(field)
        elif field not in {"documents", "message"} and not request.context.get(field):
            missing.append(field)
    return PrecheckResult(passed=not missing, missing_inputs=missing)
