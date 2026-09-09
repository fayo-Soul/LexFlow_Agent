"""全局工作流 State - 定义 LangGraph 的共享状态结构

本模块定义了 CaseAgentState 类，它是整个法律案件分析工作流的核心状态容器。
所有节点（nodes）通过读取和更新这个状态来实现数据的传递和流程的推进。
状态包含：输入数据、中间产出、复核结果、降级记录、追踪信息等。
"""

from __future__ import annotations  # 启用未来 Python 类型注解特性

from datetime import datetime  # 时间戳处理
from typing import Any, Optional  # 类型提示工具

from pydantic import BaseModel, ConfigDict, Field  # 数据模型验证和字段配置

# 导入工作流所需的各类数据模型
from lexflow_agent.engine.models.enums import WorkflowStatus  # 工作流状态枚举
from lexflow_agent.engine.models.case import CaseInput, CaseDocument  # 案件输入模型
from lexflow_agent.engine.models.facts import Fact  # 事实提取模型
from lexflow_agent.engine.models.issues import Issue  # 争议焦点模型
from lexflow_agent.engine.models.evidence import EvidenceItem  # 证据项模型
from lexflow_agent.engine.models.research import ResearchResult  # 法律研究结果模型
from lexflow_agent.engine.models.strategy import StrategyResult  # 诉讼策略模型
from lexflow_agent.engine.models.document import DraftResult  # 文书草稿模型
from lexflow_agent.engine.models.review import ReviewResult, DegradationRecord, HallucinationRecord  # 复核和降级模型
from lexflow_agent.engine.models.response import NodeTrace  # 节点追踪模型


class MaterialAnalysis(BaseModel):
    """材料分析结果模型 - 存储单个文档的分析信息"""
    document_id: str  # 文档唯一标识
    document_type: str = ""  # 文档类型（如：合同、证据、起诉状等）
    summary: str = ""  # 文档内容摘要
    parties: list[str] = Field(default_factory=list)  # 文档中涉及的当事人列表
    dates: list[dict] = Field(default_factory=list)  # 文档中提取的日期信息
    amounts: list[dict] = Field(default_factory=list)  # 文档中提取的金额信息
    relevance: float = 0.0  # 与案件的相关度评分（0-1）
    warnings: list[str] = Field(default_factory=list)  # 分析过程中的警告信息


class CaseAgentState(BaseModel):
    """全局工作流状态 - LangGraph State 的 Schema
    
    这是整个法律案件分析系统的核心数据结构，所有 Agent 节点都通过读写这个状态
    来实现数据传递和流程控制。状态包含了从案件输入到最终文书生成的完整生命周期数据。
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)  # 允许任意类型字段

    # ── 运行标识 ──
    run_id: str = ""  # 本次运行的唯一标识符
    case_id: str = ""  # 案件的业务标识符
    idempotency_key: Optional[str] = None  # 幂等键，用于防止重复提交

    # ── 流程控制 ──
    status: WorkflowStatus = WorkflowStatus.PENDING  # 当前工作流状态（待处理/运行中/完成等）
    current_node: str = ""  # 当前正在执行的节点名称
    current_agent: str = ""  # 当前正在执行的 Agent 名称
    errors: list[str] = Field(default_factory=list)  # 运行过程中累积的错误信息
    warnings: list[str] = Field(default_factory=list)  # 运行过程中累积的警告信息
    retry_count: int = 0  # 当前重试次数
    revision_count: int = 0  # 当前修订次数（用于控制循环迭代次数）
    max_revision_count: int = 3  # 最大允许修订次数
    phase: str = ""  # 当前阶段：validation → material → facts → issues → evidence → research → strategy → drafting → review

    # ── 输入 ──
    case_input: Optional[CaseInput] = None  # 案件原始输入数据

    # ── 中间产出 ──
    material_analyses: list[MaterialAnalysis] = Field(default_factory=list)  # 材料分析结果列表
    facts: list[Fact] = Field(default_factory=list)  # 提取的案件事实列表
    issues: list[Issue] = Field(default_factory=list)  # 识别的争议焦点列表
    evidence_matrix: list[EvidenceItem] = Field(default_factory=list)  # 证据矩阵（证据项列表）
    legal_research: list[ResearchResult] = Field(default_factory=list)  # 法律研究结果列表
    strategy: Optional[StrategyResult] = None  # 生成的诉讼策略
    drafts: list[DraftResult] = Field(default_factory=list)  # 生成的文书草稿列表

    # ── 复核 ──
    deterministic_review: Optional[ReviewResult] = None  # 确定性复核结果（基于规则的审查）
    semantic_review: Optional[ReviewResult] = None  # 语义复核结果（基于 LLM 的审查）

    # ── 幻觉 ──
    blocked_hallucinations: list[HallucinationRecord] = Field(default_factory=list)  # 被拦截的幻觉记录

    # ── 降级 ──
    degradations: list[DegradationRecord] = Field(default_factory=list)  # 降级执行记录
    fallback_triggered: bool = False  # 是否触发了降级兜底

    # ── 澄清对话 ──
    clarification_questions: list[dict[str, Any]] = Field(default_factory=list)  # 待回答的澄清问题
    clarification_answers: dict[str, Any] = Field(default_factory=dict)  # 用户已提交的澄清回答

    # ── 多轮法律咨询（收编自 legal_consult Agent）──
    # 统一入口层维护的多轮对话历史：[{"role": "user"|"assistant", "content": str}]
    # 由 research_law 节点在无案件材料（纯咨询）模式下注入 Prompt，随 State 持久化获得跨轮记忆
    consult_history: list[dict[str, Any]] = Field(default_factory=list)

    # ── 轨迹 ──
    trace: list[NodeTrace] = Field(default_factory=list)  # 节点执行轨迹（用于监控和调试）
    workflow_events: list[dict[str, Any]] = Field(default_factory=list)  # 工作流事件日志
    prompt_versions: dict[str, str] = Field(default_factory=dict)  # 使用的 Prompt 版本记录
    model_used: str = ""  # 实际使用的 LLM 模型名称

    # ── 时间 ──
    created_at: datetime = Field(default_factory=datetime.now)  # 状态创建时间
    updated_at: datetime = Field(default_factory=datetime.now)  # 状态最后更新时间
    completed_at: Optional[datetime] = None  # 工作流完成时间（None 表示未完成）

    def to_output_package(self) -> dict[str, Any]:
        """生成最终输出包 - 将状态序列化为可返回给客户端的字典格式
        
        该方法将内部状态对象转换为标准化的输出格式，包含：
        - 基本信息（run_id, case_id, status 等）
        - 所有中间产出数据（facts, issues, evidence 等）
        - 复核结果和降级记录
        - 性能指标（token 使用量、总耗时）
        
        Returns:
            dict[str, Any]: 包含完整案件分析结果的字典
        """
        return {
            "run_id": self.run_id,  # 运行标识
            "case_id": self.case_id,  # 案件标识
            "status": self.status.value,  # 工作流状态值
            "current_agent": self.current_agent,  # 当前 Agent
            "revision_count": self.revision_count,  # 修订次数
            "material_analyses": [m.model_dump() for m in self.material_analyses],  # 材料分析（序列化）
            "facts": [f.model_dump() for f in self.facts],  # 事实列表（序列化）
            "issues": [i.model_dump() for i in self.issues],  # 争议焦点（序列化）
            "evidence_matrix": [e.model_dump() for e in self.evidence_matrix],  # 证据矩阵（序列化）
            "legal_research": [r.model_dump() for r in self.legal_research],  # 研究结果（序列化）
            "strategy": self.strategy.model_dump() if self.strategy else None,  # 策略（序列化或 None）
            "drafts": [d.model_dump() for d in self.drafts],  # 文书草稿（序列化）
            "review": {
                "deterministic": self.deterministic_review.model_dump() if self.deterministic_review else None,
                "semantic": self.semantic_review.model_dump() if self.semantic_review else None,
            },  # 复核结果（包含确定性和语义两种）
            "blocked_hallucinations": [h.model_dump() for h in self.blocked_hallucinations],  # 幻觉记录
            "degradations": [d.model_dump() for d in self.degradations],  # 降级记录
            "fallback_triggered": self.fallback_triggered,  # 是否触发降级
            "trace": [t.model_dump() for t in self.trace],  # 执行轨迹
            "workflow_events": self.workflow_events,  # 事件日志
            "prompt_versions": self.prompt_versions,  # Prompt 版本
            "model_used": self.model_used,  # 使用的模型
            "token_usage": sum(t.token_usage for t in self.trace),  # 总 token 消耗
            "total_duration_ms": sum(t.duration_ms for t in self.trace),  # 总执行时间（毫秒）
        }