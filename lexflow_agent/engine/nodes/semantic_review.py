# 模块文档字符串：语义复核 Agent + 输出级幻觉检测
"""语义复核 Agent + 输出级幻觉检测"""

# 启用 Python 3.10+ 的新特性支持
from __future__ import annotations

# 导入默认字典，用于统计节点幻觉次数
from collections import defaultdict
# 导入日期时间模块
from datetime import datetime

# 导入案件智能体状态
from lexflow_agent.engine.state import CaseAgentState
# 导入工作流状态枚举
from lexflow_agent.engine.models.enums import WorkflowStatus
# 导入复核结果、发现项和幻觉记录模型
from lexflow_agent.engine.models.review import ReviewResult, ReviewFinding, HallucinationRecord
# 导入 LLM 对话调用辅助函数
from lexflow_agent.engine.nodes.llm_helper import llm_chat_call
# 导入 Prompt 管理器
from lexflow_agent.engine.prompts.manager import prompt_manager
# 导入 LangSmith traceable 装饰器
from lexflow_agent.engine.tracing import trace_node


# 提取所有原文文本的辅助函数
def _extract_all_text(state: CaseAgentState) -> str:
    texts = []
    if state.case_input:
        for doc in state.case_input.documents:
            for page in doc.pages:
                texts.append(page.text)
    return " ".join(texts)


# 幻觉检测辅助函数
def _detect_hallucinations(state: CaseAgentState) -> list[HallucinationRecord]:
    # 获取所有原文文本
    all_text = _extract_all_text(state)
    # 初始化被阻止的幻觉列表
    blocked = []
    # 遍历所有已确认的事实
    for fact in state.facts:
        if fact.status.value == "confirmed":
            # 检查事实来源的引用是否在原文中
            found = any(src.quote and src.quote[:30] in all_text for src in fact.sources)
            # 如果找不到匹配，记录为幻觉
            if not found:
                blocked.append(HallucinationRecord(
                    node="extract_facts", assertion=fact.statement,
                    matched=False, detected_at=datetime.now(),
                ))
    return blocked


# 语义复核节点函数
@trace_node("semantic_review")
def semantic_review_node(state: CaseAgentState) -> dict:
    # 初始化降级记录和追踪列表
    all_degradations = list(state.degradations)
    all_traces = list(state.trace)
    # 初始化发现项列表
    findings = []

    # Step 1: LLM 语义复核
    # 构建事实、争议焦点和文书初稿的文本
    facts = "\n".join(f"- {f.statement}" for f in state.facts) or "(无)"
    issues = "\n".join(f"- {i.claim}" for i in state.issues) or "(无)"
    drafts = "\n".join(f"[{d.document_type}] {d.full_text[:500]}" for d in (state.drafts or [])) or "(无)"
    # 加载语义复核 Prompt
    system_prompt = prompt_manager.load("semantic_review_agent")
    # 构建消息列表
    messages = [
        {"role": "system", "content": system_prompt},  # 系统提示
        {"role": "user", "content": f"确认事实:\n{facts}\n\n争议焦点:\n{issues}\n\n文书初稿:\n{drafts}"},  # 用户输入
    ]
    # 调用 LLM 进行语义复核
    review_text, degraded, degs, trace = llm_chat_call(
        "semantic_review", messages, state, fallback_text="语义复核已跳过（降级）",
    )
    # 收集降级记录和追踪信息
    all_degradations.extend(degs)
    all_traces.append(trace)
    # 如果复核文本存在，添加为发现项
    if review_text:
        findings.append(ReviewFinding(severity="info", node="semantic_review", description=review_text[:200]))

    # Step 2: 确定性幻觉检测
    blocked = _detect_hallucinations(state)
    # 统计每个节点的幻觉次数
    node_counts = defaultdict(int)
    for h in blocked:
        node_counts[h.node] += 1
    # 如果某节点幻觉次数 >= 2，需要人工干预
    status_update = {}
    for node, count in node_counts.items():
        if count >= 2:
            status_update["status"] = WorkflowStatus.WAITING_HUMAN_INTERVENTION

    # 返回语义复核结果
    return {"semantic_review": ReviewResult(findings=findings, passed=len(blocked)==0),
        "blocked_hallucinations": state.blocked_hallucinations + blocked,
        "current_node": "semantic_review", "degradations": all_degradations,
        "trace": all_traces, "prompt_versions": {**state.prompt_versions, **prompt_manager.versions},
        **status_update}