# 模块文档字符串：Legal Research Agent backed exclusively by verified Law RAG citations.
"""Legal Research Agent backed exclusively by verified Law RAG citations."""

# 启用 Python 3.10+ 的新特性支持
from __future__ import annotations

# 导入线程池执行器，用于并行检索
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入 Pydantic 基础模型和字段装饰器
from pydantic import BaseModel, Field

# 导入法律研究结果和引用模型
from lexflow_agent.engine.models.research import Citation, ResearchResult
# 导入 LLM 结构化调用辅助函数
from lexflow_agent.engine.nodes.llm_helper import llm_structured_call
# 导入 Prompt 管理器
from lexflow_agent.engine.prompts.manager import prompt_manager
# 导入案件智能体状态
from lexflow_agent.engine.state import CaseAgentState
# 导入法律 RAG 客户端
from lexflow_agent.engine.tools.law_client import law_client
# 导入 LangSmith traceable 装饰器
from lexflow_agent.engine.tracing import trace_node


# 法律研究输出模型
class ResearchOutput(BaseModel):
    conclusion: str                            # 研究结论
    application_conditions: list[str] = Field(default_factory=list)  # 适用条件
    citation_ids: list[str] = Field(default_factory=list)  # 引用 ID 列表
    uncertainties: list[str] = Field(default_factory=list)  # 不确定性列表


# 构建引用对象的辅助函数
def _citation(item: dict) -> Citation:
    return Citation(
        citation_id=item["citation_id"],       # 引用 ID
        document_id=item["document_id"],       # 文档 ID
        chunk_id=item.get("chunk_id", ""),     # 文本块 ID
        title=item["title"],                   # 标题
        location=item.get("location", ""),     # 位置
        quote=item.get("quote", ""),           # 引用原文
    )


# 法律研究节点函数
@trace_node("research_law")
def research_law_node(state: CaseAgentState) -> dict:
    # 初始化降级记录和追踪列表
    all_degradations = list(state.degradations)
    all_traces = list(state.trace)
    # 初始化研究结果列表
    research_results = []

    # ★收编法律咨询：无争议焦点（纯法律咨询模式）时，用用户问题 + 多轮对话历史合成一个虚拟争点
    # 这样多轮法律咨询可以复用法律研究 Agent 的 RAG 检索 + 引用验证，而不再需要独立的咨询 Agent
    if not state.issues:
        return _research_consult_mode(state, all_degradations, all_traces, research_results)

    # 遍历每个争议焦点进行法律研究
    for issue in state.issues:
        # 获取研究问题列表
        questions = [q for q in issue.research_questions if q] or [issue.claim]
        # 初始化检索结果
        retrieved = []
        # 使用线程池并行检索
        with ThreadPoolExecutor(max_workers=min(5, len(questions))) as executor:
            # 提交所有检索任务
            futures = {
                executor.submit(
                    law_client.search,         # 检索方法
                    question,                  # 检索问题
                    ["regulation", "judicial_interpretation", "case"],  # 检索类型
                ): question
                for question in questions
            }
            # 收集检索结果
            for future in as_completed(futures):
                try:
                    response = future.result()
                    retrieved.extend(response.get("results", []))
                except Exception:
                    continue

        # 去重检索结果（基于 citation_id）
        unique = {
            item.get("citation_id"): item
            for item in retrieved
            if item.get("citation_id")
            and item.get("document_id")
            and item.get("title")
            and item.get("quote")
        }
        # 如果没有有效引用，记录不确定性
        if not unique:
            research_results.append(ResearchResult(
                issue_id=issue.issue_id,
                conclusion="未检索到可验证的法律引用，不能形成确定性法律结论。",
                uncertainties=["Law RAG 未返回有效引用"],
                citations_valid=False,
            ))
            continue

        # 构建检索上下文
        context = "\n".join(
            f"[{cid}] {item['title']} {item.get('location', '')}: {item['quote']}"
            for cid, item in unique.items()
        )
        # 构建消息列表
        messages = [
            {"role": "system", "content": prompt_manager.load("research_agent")},  # 系统提示
            {"role": "user", "content": (
                f"争议焦点: {issue.claim}\n研究问题: {'; '.join(questions)}\n\n"
                f"仅可使用以下检索结果：\n{context}"
            )},  # 用户输入
        ]
        # 调用 LLM 进行法律研究分析
        output, _, degradations, trace = llm_structured_call(
            "research_law",
            messages,
            ResearchOutput,
            state,
            fallback_data=None,
        )
        # 收集降级记录和追踪信息
        all_degradations.extend(degradations)
        all_traces.append(trace)

        # 过滤有效的引用 ID
        selected_ids = [
            citation_id for citation_id in (output.citation_ids if output else [])
            if citation_id in unique
        ]
        # 构建引用对象列表
        citations = [_citation(unique[citation_id]) for citation_id in selected_ids]
        # 判断引用是否有效
        citations_valid = bool(citations)
        # 生成结论
        conclusion = (
            output.conclusion
            if output and citations_valid
            else "存在检索结果，但未能形成带有效引用的确定性法律结论。"
        )
        # 添加研究结果
        research_results.append(ResearchResult(
            issue_id=issue.issue_id,
            conclusion=conclusion,
            application_conditions=output.application_conditions if output else [],
            favorable_sources=citations,
            uncertainties=output.uncertainties if output else ["模型输出降级"],
            citations_valid=citations_valid,
        ))

    # 返回法律研究结果
    return {
        "legal_research": research_results,      # 法律研究结果列表
        "current_node": "research_law",          # 当前节点名称
        "degradations": all_degradations,        # 降级记录列表
        "trace": all_traces,                     # 追踪信息列表
        "prompt_versions": {**state.prompt_versions, **prompt_manager.versions},  # Prompt 版本
    }


def _research_consult_mode(
    state: CaseAgentState,
    all_degradations: list,
    all_traces: list,
    research_results: list,
) -> dict:
    """纯法律咨询模式（无争议焦点、无案件材料）— 收编自原 legal_consult Agent

    将用户本轮问题视为虚拟"争议焦点"，携带多轮对话历史走 RAG 检索 + 引用验证。
    复用 research_law 的降级与引用门禁，保证咨询回答同样「只基于验证引用」。
    """
    # 用户本轮问题：优先取 case_input.client_goal，缺失时回退为默认引导
    question = (
        state.case_input.client_goal.strip()
        if state.case_input and state.case_input.client_goal.strip()
        else "请介绍一下劳动争议相关的法律规定"
    )

    # 构建多轮对话历史文本（供 LLM 参考上下文，理解追问）
    history_lines = []
    for turn in state.consult_history[-10:]:  # 最多携带最近 10 轮
        role = "用户" if turn.get("role") == "user" else "助手"
        content = str(turn.get("content", ""))
        if content:
            history_lines.append(f"{role}: {content}")
    history_text = "\n".join(history_lines) or "(无历史对话)"

    # 用 ThreadPoolExecutor 并行检索（兼容多问题场景，此处为单个问题）
    retrieved = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            law_client.search,
            question,
            ["regulation", "judicial_interpretation", "case"],
        )
        try:
            response = future.result()
            retrieved.extend(response.get("results", []))
        except Exception:
            pass

    # 去重检索结果（基于 citation_id，与案件模式一致）
    unique = {
        item.get("citation_id"): item
        for item in retrieved
        if item.get("citation_id")
        and item.get("document_id")
        and item.get("title")
        and item.get("quote")
    }

    # 无有效引用 → 不强行下结论，遵循「检索不足则说明证据不足」约束
    if not unique:
        research_results.append(ResearchResult(
            issue_id="consult_default",
            conclusion="未检索到可验证的法律引用，建议咨询专业律师获取具体意见。",
            uncertainties=["Law RAG 未返回有效引用"],
            citations_valid=False,
        ))
        return {
            "legal_research": research_results,
            "current_node": "research_law",
            "degradations": all_degradations,
            "trace": all_traces,
            "prompt_versions": {**state.prompt_versions, **prompt_manager.versions},
        }

    # 构建检索上下文
    context = "\n".join(
        f"[{cid}] {item['title']} {item.get('location', '')}: {item['quote']}"
        for cid, item in unique.items()
    )
    # 构建消息列表：携带多轮对话历史 + 本轮问题 + 仅基于检索结果的约束
    messages = [
        {"role": "system", "content": prompt_manager.load("research_agent")},
        {"role": "user", "content": (
            f"对话历史（供参考上下文）:\n{history_text}\n\n"
            f"用户问题: {question}\n\n仅可使用以下检索结果：\n{context}"
        )},
    ]
    # 调用 LLM 进行法律研究分析
    output, _, degradations, trace = llm_structured_call(
        "research_law", messages, ResearchOutput, state, fallback_data=None,
    )
    all_degradations.extend(degradations)
    all_traces.append(trace)

    # 过滤有效的引用 ID
    selected_ids = [
        citation_id for citation_id in (output.citation_ids if output else [])
        if citation_id in unique
    ]
    citations = [_citation(unique[citation_id]) for citation_id in selected_ids]
    citations_valid = bool(citations)
    conclusion = (
        output.conclusion
        if output and citations_valid
        else "存在检索结果，但未能形成带有效引用的确定性法律结论。"
    )
    research_results.append(ResearchResult(
        issue_id="consult_default",
        conclusion=conclusion,
        application_conditions=output.application_conditions if output else [],
        favorable_sources=citations,
        uncertainties=output.uncertainties if output else ["模型输出降级"],
        citations_valid=citations_valid,
    ))

    return {
        "legal_research": research_results,
        "current_node": "research_law",
        "degradations": all_degradations,
        "trace": all_traces,
        "prompt_versions": {**state.prompt_versions, **prompt_manager.versions},
    }