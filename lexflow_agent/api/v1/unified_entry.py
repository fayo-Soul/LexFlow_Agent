"""统一入口(unified_entry) — SSE 流式 + LLM 路由 + 前置拦截

对标 EduAgent unified_chat.py + 8.4/8.5 统一入口设计：
1. 规则前置拦截 _pre_filter（零 Token）
2. LLM 路由 _llm_route 判 6 类意图
3. SSE 事件推送：routing_decision / token / progress / review_required /
   clarification / guidance / pipeline_plan / done / error
4. 按路由分支分发到编排器 _dispatch
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import AsyncGenerator

from pydantic import BaseModel, Field

from lexflow_agent.config.settings import settings
from lexflow_agent.engine.gateway.factory import LLMFactory
from lexflow_agent.engine.models.case import CaseDocument, CaseType, ClientRole
from lexflow_agent.engine.orchestrator.models import (
    AgentType,
    AgentRequest,
    ClarificationRequest,
    ExecutionMode,
    OrchestrationResult,
)
from lexflow_agent.engine.orchestrator.service import get_orchestrator
from lexflow_agent.engine.precheck.rule_engine import rule_engine
from lexflow_agent.engine.routing.router import route_request, RoutingDecision
from lexflow_agent.engine.orchestrator.models import TaskRequest
from lexflow_agent.engine.tools.file_parser import file_parser


# ── 统一入口请求模型 ────────────────────────────────────
class UnifiedChatRequest(BaseModel):
    """统一入口请求 — 对标 EduAgent UnifiedChatRequest"""
    case_id: str = Field(default="chat", min_length=1)
    message: str = Field(..., min_length=1, description="用户输入")
    case_type: CaseType = CaseType.LABOR_DISPUTE
    client_role: ClientRole = ClientRole.EMPLOYEE
    documents: list[CaseDocument] = Field(default_factory=list)
    file_paths: list[str] = Field(
        default_factory=list,
        description="用户上传的原始文件路径列表（PDF/DOCX/TXT/MD/图片），由 file_parser 自动解析为分页文本",
    )
    thread_id: str = ""
    idempotency_key: str | None = None


# ── SSE 事件类型 ────────────────────────────────────────
SSE_EVENT_TYPES = (
    "routing_decision",    # LLM 路由判决
    "progress",            # 节点进度
    "token",               # 流式 token
    "meta",                # 元信息（法条来源等）
    "review_required",     # 人工审核卡片
    "clarification",       # 澄清追问卡片
    "consult_state",       # 法律咨询对话状态
    "guidance",            # 引导卡片
    "pipeline_plan",       # 多步骤计划
    "done",                # 流结束
    "error",               # 异常
)


def _sse_event(event_type: str, data: dict | str) -> str:
    """格式化为 SSE 事件"""
    payload = json.dumps(data, ensure_ascii=False, default=str) if isinstance(data, dict) else data
    return f"event: {event_type}\ndata: {payload}\n\n"


# ── 规则前置拦截 ────────────────────────────────────────
# 对标 EduAgent 8.4 前置拦截：零 Token，4 类场景直接回复
_PRE_FILTER_TEMPLATES: dict[str, str] = {
    "greeting": (
        "你好！我是 LexFlow 法律案件分析助手。我可以帮你：\n"
        "• 分析劳动争议案件\n"
        "• 检索劳动法律法规\n"
        "• 生成案件策略\n"
        "• 起草法律文书\n"
        "• 提供法律咨询\n\n"
        "请直接描述你的需求即可。"
    ),
    "thanks": "不客气！如有其他法律问题，随时问我。",
    "who_are_you": "我是 LexFlow Agent，专注于劳动争议案件分析的 AI 助手。",
    "capabilities": (
        "我能做这些事：\n"
        "1. 📋 案件理解 — 分析案件材料、提取事实\n"
        "2. 📚 法律研究 — 检索法规、判例\n"
        "3. 🎯 案件策略 — 证据矩阵、诉讼策略\n"
        "4. 📝 文书生成 — 起诉状、答辩状、证据清单\n"
        "5. 💬 法律咨询 — 多轮对话解答劳动法问题\n\n"
        "你可以上传案件文件，也可以直接问法律问题。"
    ),
}


def _pre_filter(message: str) -> dict | None:
    """规则前置拦截，命中返回模板 dict，未命中返回 None"""
    msg = message.strip().lower()

    greetings = {"你好", "您好", "hi", "hello", "嗨", "在吗", "在不在"}
    if msg in greetings or any(msg == g for g in greetings):
        return {"type": "greeting", "template": _PRE_FILTER_TEMPLATES["greeting"]}

    thanks = {"谢谢", "感谢", "thanks", "thank you", "thx", "多谢"}
    if msg in thanks or any(msg == t for t in thanks):
        return {"type": "thanks", "template": _PRE_FILTER_TEMPLATES["thanks"]}

    who = {"你是谁", "你是什么", "who are you", "你叫什么"}
    if any(w in msg for w in who):
        return {"type": "who_are_you", "template": _PRE_FILTER_TEMPLATES["who_are_you"]}

    cap = {"能做什么", "功能", "capabilities", "有什么功能", "能干嘛"}
    if any(c in msg for c in cap) and len(msg) < 20:
        return {"type": "capabilities", "template": _PRE_FILTER_TEMPLATES["capabilities"]}

    return None


# ── LLM 路由 ────────────────────────────────────────────
# 对标 EduAgent 8.4 的 6 类意图路由
async def _llm_route(message: str, documents_count: int) -> RoutingDecision:
    """使用 LLM 做意图分类，失败时回退到规则路由"""
    # 优先用旧路由（规则 + LLM fallback）
    temp_request = TaskRequest(
        case_id="route",
        message=message,
        documents=[],
    )
    return route_request(temp_request, use_llm=True)


# ── 分发到编排器 ────────────────────────────────────────
async def _dispatch(
    agent_type: AgentType,
    message: str,
    documents: list[CaseDocument],
    context: dict,
    case_id: str,
    thread_id: str,
    execution_mode: ExecutionMode = ExecutionMode.SINGLE,
    pipeline_key: str = "",
) -> AgentRequest:
    """构建 AgentRequest 并调用编排器"""
    from datetime import datetime
    request = AgentRequest(
        run_id=f"run_{case_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        case_id=case_id,
        agent_type=agent_type,
        execution_mode=execution_mode,
        message=message,
        documents=documents,
        context=context,
        thread_id=thread_id,
    )
    if pipeline_key:
        request.context["pipeline_key"] = pipeline_key
    return request


# ── 统一 SSE 流 ─────────────────────────────────────────
async def unified_chat_events(request: UnifiedChatRequest) -> AsyncGenerator[str, None]:
    """统一入口 SSE 事件生成器 — 对标 EduAgent unified_chat.py 主循环

    生命周期：
    1. file_parse → pre_filter → routing → dispatch → events → done
    """
    message = request.message
    documents = request.documents

    # ① 文件解析（在路由之前，将原始文件转为 CaseDocument）
    if request.file_paths and settings.FILE_PARSER_ENABLED:
        parsed_docs: list[CaseDocument] = []
        for fp in request.file_paths:
            try:
                doc = file_parser.parse(fp)
                parsed_docs.append(doc)
            except Exception as exc:
                yield _sse_event("error", {
                    "message": f"文件解析失败 [{Path(fp).name}]: {exc}",
                    "file": fp,
                })
        request.documents.extend(parsed_docs)
        if parsed_docs:
            yield _sse_event("progress", {
                "agent": "file_parser",
                "status": "completed",
                "message": f"已解析 {len(parsed_docs)} 个文件，共 {sum(len(d.pages) for d in parsed_docs)} 页",
            })

    # ② 规则前置拦截
    pre = _pre_filter(message)
    if pre is not None:
        yield _sse_event("guidance", {
            "message": pre["template"],
            "type": pre["type"],
        })
        yield _sse_event("done", {"reason": "pre_filter"})
        return

    # ③ LLM 路由
    try:
        routing = await asyncio.to_thread(
            lambda: route_request(
                TaskRequest(
                    case_id=request.case_id,
                    message=message,
                    documents=documents,
                ),
                use_llm=True,
            )
        )
    except Exception:
        from lexflow_agent.engine.routing.router import _fallback_route
        temp = TaskRequest(
            case_id=request.case_id,
            message=message,
            documents=documents,
        )
        routing = _fallback_route(temp)

    yield _sse_event("routing_decision", {
        "intent": routing.intent.value,
        "confidence": routing.confidence,
        "candidate_agents": [a.value for a in routing.candidate_agents],
        "reason": routing.reason,
        "source": routing.source,
    })

    # ③ 按路由分支分发
    if routing.requires_clarification:
        yield _sse_event("guidance", {
            "message": "请补充更多信息，你希望我帮你做什么？",
            "suggestions": ["分析案件", "查法律", "写文书", "法律咨询", "策略建议"],
        })
        yield _sse_event("done", {"reason": "clarify"})
        return

    # 判断执行模式
    candidates = routing.candidate_agents

    # ★快速路径组合识别（A1：跳过人工审核，结果仅供参考）
    # 带材料 + 只要法律研究 → [case_understanding, legal_research]
    # 带材料 + 只要策略方案 → [case_understanding, legal_research, case_strategy]
    _RESEARCH_PAIR = {AgentType.CASE_UNDERSTANDING, AgentType.LEGAL_RESEARCH}
    _STRATEGY_TRIPLE = {AgentType.CASE_UNDERSTANDING, AgentType.LEGAL_RESEARCH, AgentType.CASE_STRATEGY}

    if set(candidates) == _RESEARCH_PAIR:
        # 研究快速路径：两 Agent
        yield _sse_event("pipeline_plan", {
            "steps": [a.value for a in candidates],
            "pipeline_key": "research_fast",
        })
        execution_mode = ExecutionMode.PIPELINE
        pipeline_key = "research_fast"
    elif set(candidates) == _STRATEGY_TRIPLE:
        # 策略快速路径：三 Agent
        yield _sse_event("pipeline_plan", {
            "steps": [a.value for a in candidates],
            "pipeline_key": "strategy_fast",
        })
        execution_mode = ExecutionMode.PIPELINE
        pipeline_key = "strategy_fast"
    elif len(candidates) >= 3:
        # 完整流水线（原逻辑）
        yield _sse_event("pipeline_plan", {
            "steps": [a.value for a in candidates],
            "pipeline_key": "labor_dispute_full",
        })
        execution_mode = ExecutionMode.PIPELINE
        pipeline_key = "labor_dispute_full"
    else:
        # 单 Agent 直达（原逻辑）
        execution_mode = ExecutionMode.SINGLE
        pipeline_key = ""

    primary_agent = candidates[0] if candidates else AgentType.LEGAL_RESEARCH

    # ★多轮法律咨询：纯咨询请求（单 Agent + legal_research + 无案件材料）走会话记忆
    consult_context: dict = {}
    consult_store_key = f"consult_{request.thread_id or request.case_id}"
    is_consult = (
        execution_mode == ExecutionMode.SINGLE
        and primary_agent == AgentType.LEGAL_RESEARCH
        and not documents
    )
    if is_consult:
        from lexflow_agent.engine.memory import session_store
        saved = session_store.get(consult_store_key) or {}
        history = list(saved.get("consult_history", []))
        # 追加本轮用户问题
        history.append({"role": "user", "content": message})
        consult_context["consult_history"] = history
        consult_context["thread_id"] = request.thread_id

    # 构建并发给编排器
    agent_req = await _dispatch(
        agent_type=primary_agent,
        message=message,
        documents=documents,
        context=consult_context,
        case_id=request.case_id,
        thread_id=request.thread_id,
        execution_mode=execution_mode,
        pipeline_key=pipeline_key,
    )

    orchestrator = get_orchestrator()
    result = orchestrator.handle(agent_req)

    # ④ 处理编排器结果 → SSE
    if isinstance(result, ClarificationRequest):
        yield _sse_event("clarification", {
            "agent_type": result.agent_type.value,
            "missing_fields": result.missing_fields,
            "questions": [q.model_dump() for q in result.questions],
            "guidance": result.guidance,
        })
        yield _sse_event("done", {"status": "waiting_clarification"})
        return

    if hasattr(result, "steps"):
        # PipelineResult
        for step in result.steps:
            status = step.status
            if status == "waiting_review":
                yield _sse_event("review_required", step.model_dump())
            elif status == "waiting_clarification":
                yield _sse_event("clarification", step.model_dump())
            else:
                yield _sse_event("progress", {
                    "agent": step.agent_type.value,
                    "status": step.status,
                    "token_usage": step.token_usage,
                })
        yield _sse_event("done", {
            "status": result.status,
            "total_token_usage": result.total_token_usage,
        })
    else:
        # 单 AgentResponse
        resp = result
        if resp.status == "waiting_review":
            yield _sse_event("review_required", resp.model_dump())
        elif resp.status == "waiting_clarification":
            yield _sse_event("clarification", resp.model_dump())
        else:
            # 推送 structured 结果
            if resp.agent_type == AgentType.LEGAL_RESEARCH:
                # 流式问答结果用 guidance 展示
                structured = resp.structured
                research_list = structured.get("legal_research", [])
                # ★多轮咨询：把回答追加进会话历史，写回 session_store 供下一轮使用
                if is_consult:
                    answer = ""
                    if research_list:
                        answer = research_list[0].get("conclusion", "")
                    history = list(consult_context.get("consult_history", []))
                    history.append({"role": "assistant", "content": answer or "（未能形成带有效引用的确定性结论）"})
                    from lexflow_agent.engine.memory import session_store
                    session_store.save(consult_store_key, {"consult_history": history})
                yield _sse_event("guidance", {
                    "message": json.dumps(structured, ensure_ascii=False),
                    "agent": resp.agent_type.value,
                })
                yield _sse_event("meta", {
                    "sources": research_list,
                })
            else:
                yield _sse_event("guidance", {
                    "message": json.dumps(resp.structured, ensure_ascii=False),
                    "agent": resp.agent_type.value,
                })
        yield _sse_event("done", {
            "status": resp.status,
            "token_usage": resp.token_usage,
        })


# ── 澄清回答处理 ────────────────────────────────────────
class ClarificationSubmitRequest(BaseModel):
    """用户提交澄清回答"""
    run_id: str
    answers: dict[str, str] = Field(default_factory=dict)


async def handle_clarification(submit: ClarificationSubmitRequest):
    """处理用户提交的澄清回答，继续执行原 Agent"""
    from lexflow_agent.engine.memory import session_store

    # 从 session_store 恢复 AgentRequest
    saved = session_store.get(submit.run_id)
    if not saved:
        raise ValueError(f"未找到运行记录: {submit.run_id}")

    agent_req = AgentRequest.model_validate(saved)
    # 注入澄清数据到 context
    for question_id, answer in submit.answers.items():
        agent_req.context[question_id] = answer

    orchestrator = get_orchestrator()
    return orchestrator.handle(agent_req)
