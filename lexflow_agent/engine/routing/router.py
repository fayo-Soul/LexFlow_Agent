from __future__ import annotations

from lexflow_agent.engine.gateway.factory import LLMFactory
from lexflow_agent.engine.orchestrator.models import (
    AgentName,
    AgentType,
    RoutingDecision,
    TaskIntent,
    TaskMode,
    TaskRequest,
)


def _fallback_route(request: TaskRequest) -> RoutingDecision:
    # 兼容旧 SINGLE_AGENT / FULL_CASE（来自旧 Enum）和
    # 新 SINGLE / PIPELINE（来自 ExecutionMode 别名）
    is_single = getattr(request.mode, "value", "") in ("single_agent", "single")
    is_full = getattr(request.mode, "value", "") in ("full_case",) or bool(request.documents)

    if is_single and request.requested_agent:
        return RoutingDecision(
            intent=TaskIntent(request.requested_agent.value),
            candidate_agents=[request.requested_agent],
            confidence=1.0,
            source="explicit",
            reason="用户显式指定单 Agent",
        )

    # ★快速路径规则兜底：带材料 + 明确意图 → 走少 Agent 快速路径（A1，跳过人工审核）
    # 注意：必须放在 "有材料即全流程" 判断之前，否则会被提前截获
    if request.documents:
        msg = (request.message or "").strip()
        # 带材料 + 只要法律研究
        if any(kw in msg for kw in ["研究", "法条", "法律依据", "怎么算", "赔偿标准", "法律规定"]):
            return RoutingDecision(
                intent=TaskIntent.LEGAL_RESEARCH,
                candidate_agents=[AgentType.CASE_UNDERSTANDING, AgentType.LEGAL_RESEARCH],
                confidence=0.85,
                source="rule_fallback",
                reason="带材料且明确要求法律研究，走案件理解+法律研究快速路径",
            )
        # 带材料 + 只要策略方案
        if any(kw in msg for kw in ["策略", "方案", "诉讼策略", "应对", "主张", "请求"]):
            return RoutingDecision(
                intent=TaskIntent.CASE_STRATEGY,
                candidate_agents=[AgentType.CASE_UNDERSTANDING, AgentType.LEGAL_RESEARCH, AgentType.CASE_STRATEGY],
                confidence=0.85,
                source="rule_fallback",
                reason="带材料且明确要求策略方案，走案件理解+法律研究+案件策略快速路径",
            )

    if is_full or request.documents:
        return RoutingDecision(
            intent=TaskIntent.FULL_CASE_ANALYSIS,
            candidate_agents=list(AgentName),
            confidence=0.85,
            source="rule_fallback",
            reason="请求包含案件材料或明确要求完整案件分析",
        )
    return RoutingDecision(
        intent=TaskIntent.LEGAL_CONSULTATION,
        candidate_agents=[AgentName.LEGAL_RESEARCH],
        confidence=0.8,
        source="rule_fallback",
        reason="无案件材料的法律问题按法律研究处理",
    )


def route_request(request: TaskRequest, *, use_llm: bool = True) -> RoutingDecision:
    is_single = getattr(request.mode, "value", "") in ("single_agent", "single")
    is_full = getattr(request.mode, "value", "") in ("full_case",)

    if is_single and request.requested_agent:
        return _fallback_route(request)    # 用户显式指定→直接规则路由
    if is_full:
        return _fallback_route(request)    # 用户显式要求全流程→直接规则路由
    if not use_llm:
        return _fallback_route(request)    # LLM路由被禁用→直接规则路由
    try:
        structured = LLMFactory.with_structured_output(RoutingDecision, task_type="task_routing")
        result = structured([
            {
                "role": "system",
                "content": (
                    "识别法律任务意图，选择必要的业务 Agent：\n"
                    "- 简单法律咨询也走 legal_research（其内置 RAG 检索 + 多轮对话历史支持）\n"
                    "- 带案件材料且只要法律研究结果 → [case_understanding, legal_research]\n"
                    "- 带案件材料且只要策略方案 → [case_understanding, legal_research, case_strategy]\n"
                    "- 完整案件分析（含文书生成）→ 四个 Agent\n"
                    "不要决定认证、权限或执行流程。"
                ),
            },
            {
                "role": "user",
                "content": f"用户请求：{request.message}\n材料数量：{len(request.documents)}",
            },
        ], temperature=0)
        if isinstance(result, RoutingDecision):
            return result.model_copy(update={"source": "llm"})
    except Exception:
        pass
    return _fallback_route(request)
