"""编排器(调度台) — 对标 EduAgent orchestrator.py

核心职责：
- 接收 AgentRequest 请求，根据执行模式分发到不同的执行路径
- handle() → _run_single_agent() / _run_pipeline()
- 懒加载 Agent 子图（按需加载，减少启动时间和内存占用）
- @with_retry 包装（自动重试和降级处理）
- AgentRequest/AgentResponse 统一 Schema（标准化输入输出）
- 前序 structured 自动注入后序 context（Pipeline 模式下的数据传递）

工作流程：
1. 单 Agent 模式：直接调用指定 Agent 子图
2. Pipeline 模式：按预定义顺序串联多个 Agent，前序输出自动作为后序输入
3. 澄清模式：当缺少必要依赖时，返回澄清追问卡
"""

# ── 导入依赖 ────────────────────────────────────────────
# 标准库
from __future__ import annotations

from datetime import datetime
from typing import Any

# LangGraph 框架
from langgraph.graph import StateGraph

# 数据模型（案件、证据、事实、争议焦点、法律研究、策略、响应追踪）
from lexflow_agent.engine.models.case import CaseInput
from lexflow_agent.engine.models.enums import WorkflowStatus
from lexflow_agent.engine.models.evidence import EvidenceItem
from lexflow_agent.engine.models.facts import Fact
from lexflow_agent.engine.models.issues import Issue
from lexflow_agent.engine.models.research import ResearchResult
from lexflow_agent.engine.models.strategy import StrategyResult
from lexflow_agent.engine.models.response import NodeTrace

# 编排器内部模型（Agent 类型、请求/响应、执行模式、Pipeline 结果等）
from lexflow_agent.engine.orchestrator.models import (
    AgentType,
    AgentRequest,
    AgentResponse,
    ClarificationItem,
    ClarificationRequest,
    ExecutionMode,
    PipelineResult,
)

# 弹性执行模块（降级和容错处理）
from lexflow_agent.engine.resilience.fallback import resilience_execute

# Agent 状态定义
from lexflow_agent.engine.state import CaseAgentState


# ── Agent 图构建函数注册表 ─────────────────────────────
# 对标 EduAgent _get_agent_graph() 懒加载模式

_AGENT_BUILDERS: dict[AgentType, Any] = {}  # 延迟导入


def _get_builder(agent_type: AgentType):
    """
    懒加载 Agent 图构建函数
    
    根据 Agent 类型动态导入并缓存对应的构建函数，避免在启动时加载所有 Agent。
    采用懒加载策略可以：
    1. 减少启动时间
    2. 降低内存占用（只加载实际使用的 Agent）
    3. 支持热插拔（运行时添加新的 Agent 类型）
    
    Args:
        agent_type: Agent 类型枚举值
        
    Returns:
        对应的 Agent 构建函数
    """
    # 如果该 Agent 类型尚未加载，则动态导入并缓存
    if agent_type not in _AGENT_BUILDERS:
        if agent_type == AgentType.CASE_UNDERSTANDING:
            # 案件理解 Agent：负责解析案件材料、提取关键事实
            from lexflow_agent.engine.agents.case_understanding import build_case_understanding_agent
            _AGENT_BUILDERS[agent_type] = build_case_understanding_agent
        elif agent_type == AgentType.LEGAL_RESEARCH:
            # 法律研究 Agent：负责检索相关法律法规、司法解释
            from lexflow_agent.engine.agents.legal_research import build_legal_research_agent
            _AGENT_BUILDERS[agent_type] = build_legal_research_agent
        elif agent_type == AgentType.CASE_STRATEGY:
            # 案件策略 Agent：负责制定诉讼策略、分析案件走向
            from lexflow_agent.engine.agents.case_strategy import build_case_strategy_agent
            _AGENT_BUILDERS[agent_type] = build_case_strategy_agent
        elif agent_type == AgentType.DOCUMENT_GENERATION:
            # 文书生成 Agent：负责生成起诉状、答辩状等法律文书
            from lexflow_agent.engine.agents.document_generation import build_document_generation_agent
            _AGENT_BUILDERS[agent_type] = build_document_generation_agent
    return _AGENT_BUILDERS[agent_type]


# ── 依赖定义 ────────────────────────────────────────────
# 定义每个 Agent 类型所需的前置依赖字段
# 用于在编排时检查上下文是否满足执行条件
_DEPENDENCIES: dict[AgentType, tuple[str, ...]] = {
    AgentType.CASE_UNDERSTANDING: ("documents",),  # 需要案件材料
    AgentType.LEGAL_RESEARCH: ("message",),  # 需要用户问题
    AgentType.CASE_STRATEGY: ("facts", "legal_research"),  # 需要案件事实和法律研究结果
    AgentType.DOCUMENT_GENERATION: ("facts", "strategy"),  # 需要案件事实和诉讼策略
}


def _check_context(request: AgentRequest, agent_type: AgentType) -> list[str]:
    """
    检查 context 中是否缺少 Agent 依赖的字段
    
    在执行 Agent 之前，验证请求中是否包含该 Agent 所需的所有前置依赖。
    如果缺少依赖，返回缺失字段列表，用于触发澄清追问。
    
    Args:
        request: Agent 请求对象，包含 message、documents、context 等信息
        agent_type: 目标 Agent 类型
        
    Returns:
        缺失的依赖字段列表，空列表表示所有依赖都已满足
    """
    missing: list[str] = []
    for field in _DEPENDENCIES.get(agent_type, ()):
        # 检查 documents 依赖：需要案件材料
        if field == "documents" and not request.documents:
            missing.append(field)
        # 检查 message 依赖：需要用户问题（去除空白后不能为空）
        elif field == "message" and not request.message.strip():
            missing.append(field)
        # 检查其他 context 字段依赖（如 facts、legal_research、strategy 等）
        elif field not in {"documents", "message"} and not request.context.get(field):
            missing.append(field)
    return missing


class Orchestrator:
    """
    编排器 — 对标 EduAgent Orchestrator
    
    核心职责：
    - 接收 AgentRequest 请求
    - 懒加载 Agent 子图（按需加载，减少资源消耗）
    - 根据执行模式分发：单 Agent 直达 或 Pipeline 串联
    - 返回统一的 AgentResponse 或 PipelineResult
    
    设计模式：
    - 懒加载：Agent 图在首次使用时才加载，避免启动时加载所有 Agent
    - 缓存：加载后的 Agent 图会被缓存，后续调用直接使用
    - 降级：执行异常时自动降级，记录降级信息并返回保守结果
    """

    # ── Pipeline 预置 ──────────────────────────────────
    # 优先从 config/pipelines.yaml 加载，fallback 到硬编码默认值
    # 新增 Pipeline 只需编辑配置文件，无需修改源码
    PIPELINES: dict[str, list[AgentType]] = {}

    @classmethod
    def _load_pipelines(cls) -> dict[str, list[AgentType]]:
        """
        从配置文件加载 Pipeline 定义，降级到硬编码默认值。

        加载优先级：
        1. config/pipelines.yaml 配置文件（推荐，支持热更新）
        2. 硬编码默认值（兜底，保证即使配置文件损坏也能运行）

        Returns:
            Pipeline 定义字典 {key: [AgentType, ...]}
        """
        try:
            from lexflow_agent.config.config_loader import load_pipelines_config
            config = load_pipelines_config()
            if config and config.get("pipelines"):
                pipelines: dict[str, list[AgentType]] = {}
                for key, pipeline_def in config["pipelines"].items():
                    steps = [
                        AgentType(step)
                        for step in pipeline_def.get("steps", [])
                    ]
                    if steps:
                        pipelines[key] = steps
                if pipelines:
                    return pipelines
        except Exception:
            pass  # 配置文件损坏或不存在，降级到默认值

        # 硬编码默认值 — 最小可用保证
        return {
            "labor_dispute_full": [
                AgentType.CASE_UNDERSTANDING,
                AgentType.LEGAL_RESEARCH,
                AgentType.CASE_STRATEGY,
                AgentType.DOCUMENT_GENERATION,
            ],
            "research_fast": [
                AgentType.CASE_UNDERSTANDING,
                AgentType.LEGAL_RESEARCH,
            ],
            "strategy_fast": [
                AgentType.CASE_UNDERSTANDING,
                AgentType.LEGAL_RESEARCH,
                AgentType.CASE_STRATEGY,
            ],
        }

    def __init__(self):
        # 首次实例化时从配置文件加载 Pipeline 定义（后续实例复用缓存）
        if not Orchestrator.PIPELINES:
            Orchestrator.PIPELINES = Orchestrator._load_pipelines()
        # Agent 图缓存：避免重复构建相同的 Agent 图
        self._graph_cache: dict[AgentType, StateGraph] = {}

    # ── 图懒加载 ──────────────────────────────────────
    def _get_agent_graph(self, agent_type: AgentType) -> StateGraph:
        """
        获取 Agent 图（懒加载 + 缓存）
        
        对标 EduAgent _get_agent_graph 的实现模式：
        1. 检查缓存中是否已有该 Agent 类型的图
        2. 如果没有，通过 _get_builder 获取构建函数并执行构建
        3. 将构建好的图存入缓存，供后续调用使用
        
        Args:
            agent_type: Agent 类型枚举值
            
        Returns:
            构建好的 LangGraph StateGraph 对象
        """
        if agent_type not in self._graph_cache:
            builder = _get_builder(agent_type)
            self._graph_cache[agent_type] = builder()
        return self._graph_cache[agent_type]

    # ─ State 构建 ────────────────────────────────────
    def _build_initial_state(self, request: AgentRequest) -> CaseAgentState:
        """
        构建初始 Agent 状态
        
        将统一的 AgentRequest 转换为 CaseAgentState，这是 LangGraph 执行所需的内部状态格式。
        从请求中提取案件信息、上下文数据，并初始化为状态对象。
        
        Args:
            request: Agent 请求对象，包含案件 ID、消息、文档、上下文等信息
            
        Returns:
            初始化好的 CaseAgentState 对象，包含所有必要的状态字段
        """
        ctx = request.context
        # 构建案件输入数据
        case_data = {
            "case_id": request.case_id,
            "case_type": request.case_type,
            "client_role": request.client_role,
            "client_goal": request.message,
            "documents": request.documents,
            "idempotency_key": request.idempotency_key,
        }
        # 根据是否有文档选择不同的验证方式
        case_input = (
            CaseInput.model_validate(case_data)
            if request.documents
            else CaseInput.model_construct(**case_data)
        )
        # 构建完整的状态对象，从上下文中提取各阶段数据
        return CaseAgentState(
            run_id=request.run_id,
            case_id=request.case_id,
            case_input=case_input,
            facts=[Fact.model_validate(item) for item in ctx.get("facts", [])],
            issues=[Issue.model_validate(item) for item in ctx.get("issues", [])],
            legal_research=[
                ResearchResult.model_validate(item)
                for item in ctx.get("legal_research", [])
            ],
            evidence_matrix=[
                EvidenceItem.model_validate(item)
                for item in ctx.get("evidence_matrix", [])
            ],
            strategy=(
                StrategyResult.model_validate(ctx["strategy"])
                if ctx.get("strategy")
                else None
            ),
            consult_history=ctx.get("consult_history", []),
        )

    # ── 提取响应 ──────────────────────────────────────
    def _extract_response(
        self, agent_type: AgentType, state: CaseAgentState, degraded: bool = False
    ) -> AgentResponse:
        """
        从 Agent 状态提取统一响应
        
        将 CaseAgentState（LangGraph 内部状态）转换为统一的 AgentResponse 格式。
        这是 Pipeline 串联的关键：每个 Agent 的 structured 输出会被注入到下一个 Agent 的 context 中。
        
        Args:
            agent_type: Agent 类型，用于确定提取哪些字段
            state: Agent 执行后的最终状态
            degraded: 是否降级执行（异常后降级）
            
        Returns:
            统一的 AgentResponse 对象，包含状态、结构化数据、追踪信息等
        """
        # 根据状态确定执行结果状态
        status = "completed"
        if state.status in (WorkflowStatus.WAITING_CLARIFICATION,):
            status = "waiting_clarification"
        elif state.status in (WorkflowStatus.WAITING_FACT_REVIEW,
                              WorkflowStatus.WAITING_STRATEGY_REVIEW,
                              WorkflowStatus.WAITING_HUMAN_INTERVENTION):
            status = "waiting_review"
        elif degraded or state.fallback_triggered:
            status = "degraded"

        # 构建 structured — 这是 pipeline 串联的关键
        # 根据 Agent 类型提取对应的结构化输出，供下游 Agent 使用
        structured: dict[str, Any] = {}
        if agent_type == AgentType.CASE_UNDERSTANDING:
            # 案件理解 Agent 输出：案件事实 + 争议焦点
            structured["facts"] = [f.model_dump() for f in state.facts]
            structured["issues"] = [i.model_dump() for i in state.issues]
        elif agent_type == AgentType.LEGAL_RESEARCH:
            # 法律研究 Agent 输出：法律研究结果
            structured["legal_research"] = [r.model_dump() for r in state.legal_research]
            # 多轮咨询：把对话历史回传，供统一入口维护跨轮记忆
            if state.consult_history:
                structured["consult_history"] = state.consult_history
        elif agent_type == AgentType.CASE_STRATEGY:
            # 案件策略 Agent 输出：证据矩阵 + 诉讼策略
            from lexflow_agent.engine.models.evidence import EvidenceItem
            if state.evidence_matrix:
                structured["evidence_matrix"] = [e.model_dump() for e in state.evidence_matrix]
            if state.strategy:
                structured["strategy"] = state.strategy.model_dump()
        elif agent_type == AgentType.DOCUMENT_GENERATION:
            # 文书生成 Agent 输出：生成的文书草稿
            structured["drafts"] = [d.model_dump() for d in state.drafts]

        return AgentResponse(
            agent_type=agent_type,
            status=status,
            structured=structured,
            degradations=[d.model_dump() for d in state.degradations],
            trace=[t.model_dump() for t in state.trace],
            token_usage=sum(t.token_usage if hasattr(t, 'token_usage') else 0 for t in state.trace),
            warnings=state.warnings,
        )

    # ── 单 Agent 直达 ─────────────────────────────────
    def _run_single_agent(
        self, request: AgentRequest, agent_type: AgentType | None = None
    ) -> AgentResponse:
        """
        执行单个 Agent
        
        对标 EduAgent _run_single_agent 的实现模式：
        1. 懒加载获取 Agent 图
        2. 构建初始状态
        3. 执行图（带异常处理和降级机制）
        4. 提取并返回统一的 AgentResponse
        
        Args:
            request: Agent 请求对象
            agent_type: 目标 Agent 类型（如果为 None，则使用 request.agent_type）
            
        Returns:
            AgentResponse 对象，包含执行结果、结构化数据、追踪信息等
        """
        target = agent_type or request.agent_type
        graph = self._get_agent_graph(target)
        initial_state = self._build_initial_state(request)

        # 执行 Agent 图（带异常处理和降级机制）
        degraded = False
        try:
            result = graph.invoke(initial_state)
            final_state = CaseAgentState.model_validate(result)
        except Exception as exc:
            # 降级处理：记录异常信息，返回保守结果
            from lexflow_agent.engine.models.review import DegradationRecord
            initial_state.degradations.append(DegradationRecord(
                node=f"agent:{target.value}",
                reason=f"Agent 执行异常: {exc}",
                level=2,
                action="node_fallback",
                timestamp=datetime.now(),
            ))
            initial_state.fallback_triggered = True
            degraded = True
            final_state = initial_state

        return self._extract_response(target, final_state, degraded=degraded)

    # ── Pipeline 串联 ─────────────────────────────────
    def _run_pipeline(
        self, request: AgentRequest, pipeline_key: str
    ) -> PipelineResult:
        """
        执行 Pipeline（多 Agent 串联）
        
        对标 EduAgent _run_pipeline 的实现模式：
        按预定义顺序依次执行多个 Agent，前序 Agent 的 structured 输出自动注入后序 Agent 的 context。
        
        执行流程：
        1. 获取 Pipeline 定义的 Agent 序列
        2. 依次执行每个 Agent，将前序输出注入后序 context
        3. 检查依赖：如果上游未产出必要依赖，标记跳过
        4. 中断检查：如果遇到等待审核的状态，提前终止 Pipeline
        5. 汇总所有步骤的结果，返回 PipelineResult
        
        Args:
            request: Agent 请求对象
            pipeline_key: Pipeline 模板键名（如 "labor_dispute_full"）
            
        Returns:
            PipelineResult 对象，包含所有步骤的执行结果、总 token 消耗等
        """
        steps: list[AgentResponse] = []
        current_context: dict[str, Any] = dict(request.context)
        total_tokens = 0

        # 获取 Pipeline 定义的 Agent 序列
        pipeline_agents = self.PIPELINES.get(pipeline_key, [])
        for i, agent_type in enumerate(pipeline_agents):
            # 构造当前 Agent 的请求，注入前序产出
            step_request = AgentRequest(
                run_id=request.run_id,
                case_id=request.case_id,
                agent_type=agent_type,
                execution_mode=ExecutionMode.PIPELINE,
                message=request.message,
                case_type=request.case_type,
                client_role=request.client_role,
                documents=request.documents,
                context=current_context,
                idempotency_key=request.idempotency_key,
            )

            # 检查依赖：如果是 Pipeline 中间步骤但缺少依赖，跳过该 Agent
            missing = _check_context(step_request, agent_type)
            if missing and i > 0:
                # Pipeline 中如果上游未产出依赖，标记跳过
                steps.append(AgentResponse(
                    agent_type=agent_type,
                    status="skipped",
                    warnings=[f"缺少依赖: {missing}"],
                ))
                continue

            # 执行当前 Agent
            response = self._run_single_agent(step_request, agent_type)
            steps.append(response)
            total_tokens += response.token_usage

            # 把 structured 注入下一环 context（关键：实现数据传递）
            current_context.update(response.structured)

            # Pipeline 中断检查 — 保留人工审核点
            # 如果当前 Agent 返回等待审核状态，提前终止 Pipeline
            if response.status == "waiting_review":
                break

        return PipelineResult(
            run_id=request.run_id,
            steps=steps,
            status=steps[-1].status if steps else "completed",
            total_token_usage=total_tokens,
        )

    # ── 统一入口 ──────────────────────────────────────
    def handle(self, request: AgentRequest) -> AgentResponse | PipelineResult | ClarificationRequest:
        """
        统一请求处理入口
        
        对标 EduAgent handle() 的实现模式：
        根据 execution_mode 分发到不同的执行路径。
        
        执行模式：
        - CLARIFY: 返回澄清追问卡（当缺少必要依赖时）
        - PIPELINE: 执行多 Agent 串联流程
        - SINGLE: 执行单个 Agent（先检查依赖，缺少则返回澄清追问）
        
        Args:
            request: Agent 请求对象，包含执行模式、Agent 类型、上下文等信息
            
        Returns:
            - AgentResponse: 单 Agent 执行结果
            - PipelineResult: Pipeline 执行结果
            - ClarificationRequest: 澄清追问卡（当缺少依赖时）
        """
        # 澄清模式：直接返回澄清追问卡
        if request.execution_mode == ExecutionMode.CLARIFY:
            return self._build_clarification(request)

        # Pipeline 模式：执行多 Agent 串联
        if request.execution_mode == ExecutionMode.PIPELINE:
            # 优先用 context 里的 pipeline_key，fallback 到 labor_dispute_full
            pipeline_key = request.context.get("pipeline_key", "labor_dispute_full")
            return self._run_pipeline(request, pipeline_key)

        # SINGLE 模式：先检查依赖
        missing = _check_context(request, request.agent_type)
        if missing and not request.allow_orchestration:
            # 缺少依赖 → 返回澄清追问
            return self._build_clarification(request, missing_fields=missing)

        # 依赖满足 → 执行单个 Agent
        return self._run_single_agent(request)

    # ── 澄清追问构建 ─────────────────────────────────
    def _build_clarification(
        self, request: AgentRequest, missing_fields: list[str] | None = None
    ) -> ClarificationRequest:
        """
        构建澄清追问卡
        
        当 Agent 缺少必要依赖时，生成结构化的澄清追问请求。
        包含具体的问题列表和指导说明，帮助用户补充缺失信息。
        
        Args:
            request: Agent 请求对象
            missing_fields: 缺失的字段列表（如果为 None，则自动检查）
            
        Returns:
            ClarificationRequest 对象，包含追问问题列表和指导说明
        """
        missing = missing_fields or _check_context(request, request.agent_type)
        questions: list[ClarificationItem] = []
        guidance_parts: list[str] = []

        for field in missing:
            if field == "documents":
                questions.append(ClarificationItem(
                    question_id="missing_docs",
                    question="请上传案件相关材料（合同、工资单、聊天记录等）",
                    field="documents",
                    expected_type="documents",
                ))
                guidance_parts.append("上传案件材料")
            elif field == "facts":
                questions.append(ClarificationItem(
                    question_id="missing_facts",
                    question="请描述案件的基本事实：什么时间、发生了什么事、涉及哪些人？",
                    field="facts",
                    expected_type="text",
                ))
                guidance_parts.append("描述案件事实")
            elif field == "legal_research":
                questions.append(ClarificationItem(
                    question_id="missing_research",
                    question="是否有法律研究结果？可以输入法条引用或法律分析",
                    field="legal_research",
                    expected_type="text",
                ))
                guidance_parts.append("补充法律研究依据")
            elif field == "strategy":
                questions.append(ClarificationItem(
                    question_id="missing_strategy",
                    question="是否有诉讼策略或案件分析结论？可以输入已有策略",
                    field="strategy",
                    expected_type="text",
                ))
                guidance_parts.append("补充策略信息")
            elif field == "message":
                questions.append(ClarificationItem(
                    question_id="missing_message",
                    question="请描述你希望我帮你做什么？",
                    field="message",
                    expected_type="text",
                ))
                guidance_parts.append("描述需求")

        # 构建指导说明
        guidance = "请补充以下信息：\n" + "\n".join(f"• {p}" for p in guidance_parts)

        return ClarificationRequest(
            agent_type=request.agent_type,
            missing_fields=missing,
            questions=questions,
            guidance=guidance,
        )


# ── 全局单例 ────────────────────────────────────────────
# 全局 Orchestrator 单例实例
_orchestrator: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    """
    获取全局 Orchestrator 单例
    
    使用单例模式确保整个应用中只有一个 Orchestrator 实例，
    这样可以共享 Agent 图缓存，避免重复构建。
    
    Returns:
        全局唯一的 Orchestrator 实例
    """
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator