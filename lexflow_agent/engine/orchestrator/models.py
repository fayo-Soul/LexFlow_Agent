"""Orchestrator models — AgentType/ExecutionMode/AgentRequest/AgentResponse

对标 EduAgent 8.2 的 Schema 设计：
- AgentType: 4 个 Agent（法律咨询已收编进统一入口 + legal_research，见 research_law_node 咨询模式）
- ExecutionMode: single / pipeline / clarify
- AgentRequest / AgentResponse: 统一入参/出参契约
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from lexflow_agent.engine.models.case import CaseDocument
from lexflow_agent.engine.models.enums import CaseType, ClientRole


class AgentType(str, Enum):
    """Agent 类型 — 对标 EduAgent AgentType 枚举"""
    CASE_UNDERSTANDING = "case_understanding"
    LEGAL_RESEARCH = "legal_research"
    CASE_STRATEGY = "case_strategy"
    DOCUMENT_GENERATION = "document_generation"


class ExecutionMode(str, Enum):
    """执行模式 — 对标 EduAgent ExecutionMode"""
    SINGLE = "single"          # 单 Agent 直达
    PIPELINE = "pipeline"      # 多 Agent 串联
    CLARIFY = "clarify"        # 意图不明 → 追问澄清


# ── 统一入参 ────────────────────────────────────────────
class AgentRequest(BaseModel):
    """统一 Agent 入参 — 所有 Agent 共用的请求格式

    对标 EduAgent AgentRequest schema.
    """
    run_id: str = ""
    case_id: str = Field(..., min_length=1)
    agent_type: AgentType
    execution_mode: ExecutionMode = ExecutionMode.SINGLE
    message: str = Field(default="", description="用户自然语言 input")
    case_type: CaseType = CaseType.LABOR_DISPUTE
    client_role: ClientRole = ClientRole.EMPLOYEE
    documents: list[CaseDocument] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    thread_id: str = ""                            # 多轮对话线程 ID
    allow_orchestration: bool = False               # 是否允许编排器自动扩展依赖
    idempotency_key: str | None = None

    @model_validator(mode="after")
    def validate_context_keys(self):
        """context 必须可 JSON 序列化"""
        for k, v in self.context.items():
            if not isinstance(k, str):
                raise ValueError(f"context key must be str: {k}")
        return self


# ── 统一出参 ────────────────────────────────────────────
class AgentResponse(BaseModel):
    """统一 Agent 出参 — 所有 Agent 共用的返回格式

    对标 EduAgent AgentResponse schema.
    structured 字段承载 Agent 特有产出，供 pipeline 串联时自动注入.
    """
    agent_type: AgentType
    status: str                      # completed / degraded / waiting_review / waiting_clarification
    structured: dict[str, Any] = Field(default_factory=dict)
    degradations: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    token_usage: int = 0
    warnings: list[str] = Field(default_factory=list)


class PipelineResult(BaseModel):
    """Pipeline 串联结果 — 多 Agent 顺序执行后的聚合结果

    对标 EduAgent PipelineResult schema.
    """
    run_id: str
    steps: list[AgentResponse] = Field(default_factory=list)
    status: str = "completed"        # completed / degraded / partial / failed
    total_token_usage: int = 0


# ── 澄清追问 ────────────────────────────────────────────
class ClarificationItem(BaseModel):
    """单条澄清追问"""
    question_id: str
    question: str                    # 追问内容
    field: str                       # 对应 state 字段名（如 "facts", "strategy"）
    expected_type: str = "text"     # text / select / number / date
    options: list[str] = Field(default_factory=list)


class ClarificationRequest(BaseModel):
    """澄清追问集 — 当单 Agent 缺少数据时返回"""
    agent_type: AgentType
    missing_fields: list[str] = Field(default_factory=list)
    questions: list[ClarificationItem] = Field(default_factory=list)
    guidance: str = ""               # 给前端的引导文字


class ClarificationResponse(BaseModel):
    """用户提交的澄清回答"""
    run_id: str
    answers: dict[str, Any] = Field(default_factory=dict)  # question_id → answer
    action: str = "submit_clarification"


# ── 兼容旧名（别名） ──────────────────────────────────
# 保留 AgentName 别名，以便旧引用不报错
AgentName = AgentType

# ── 旧 Schema 保留（兼容） ─────────────────────────────
TaskMode = ExecutionMode

class TaskIntent(str, Enum):
    LEGAL_CONSULTATION = "legal_consultation"    # 法律咨询
    CASE_UNDERSTANDING = "case_understanding"    # 案件理解
    LEGAL_RESEARCH = "legal_research"            # 法律研究
    CASE_STRATEGY = "case_strategy"              # 案件策略
    DOCUMENT_GENERATION = "document_generation"  # 文书生成
    FULL_CASE_ANALYSIS = "full_case_analysis"    # 完整案件分析
    UNKNOWN = "unknown"                          # 未知


class TaskRequest(BaseModel):
    """兼容旧版 TaskRequest"""
    case_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    mode: ExecutionMode = ExecutionMode.SINGLE
    requested_agent: AgentType | None = None
    allow_orchestration: bool = False
    requested_output: str = "auto"
    case_type: CaseType = CaseType.LABOR_DISPUTE
    client_role: ClientRole = ClientRole.EMPLOYEE
    documents: list[CaseDocument] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = None

    @model_validator(mode="after")
    def validate_mode(self):
        if self.mode == ExecutionMode.SINGLE and self.requested_agent is None and not self.allow_orchestration:
            pass  # 宽松处理
        return self


class RoutingDecision(BaseModel):
    intent: TaskIntent
    candidate_agents: list[AgentType]
    confidence: float = Field(ge=0, le=1)
    source: str = "llm"
    reason: str = ""
    requires_clarification: bool = False


class OrchestrationResult(BaseModel):
    status: str
    route_source: str
    executed_agents: list[AgentType] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
    output: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    # ★新增：clarification 字段
    clarification: ClarificationRequest | None = None
