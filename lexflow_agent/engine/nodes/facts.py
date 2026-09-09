# 模块文档字符串：事实时间线 Agent - LLM 结构化提取
"""事实时间线 Agent - LLM 结构化提取"""

# 启用 Python 3.10+ 的新特性支持
from __future__ import annotations

# 导入可选类型提示
from typing import Optional

# 导入 Pydantic 基础模型和字段装饰器
from pydantic import BaseModel, Field

# 导入案件智能体状态
from lexflow_agent.engine.state import CaseAgentState
# 导入事实相关模型
from lexflow_agent.engine.models.facts import Fact, FactSource, FactStatus
# 导入 LLM 结构化调用辅助函数
from lexflow_agent.engine.nodes.llm_helper import llm_structured_call
# 导入 Prompt 管理器
from lexflow_agent.engine.prompts.manager import prompt_manager
# 导入 LangSmith traceable 装饰器
from lexflow_agent.engine.tracing import trace_node


# 事实输出模型
class FactOutput(BaseModel):
    fact_id: str                               # 事实 ID
    statement: str                             # 事实陈述
    occurred_at: Optional[str] = None          # 发生时间
    fact_type: str = ""                        # 事实类型
    status: str = "confirmed"                  # 事实状态
    confidence: float = 0.0                    # 置信度
    sources: list[dict] = Field(default_factory=list)  # 来源列表
    conflicts: list[str] = Field(default_factory=list) # 冲突列表
    questions: list[str] = Field(default_factory=list) # 问题列表


# 事实列表输出模型
class FactListOutput(BaseModel):
    facts: list[FactOutput] = Field(default_factory=list)  # 事实列表


# 验证事实来源的辅助函数
def _validated_sources(state: CaseAgentState, raw_sources: list[dict]) -> list[FactSource]:
    """Keep only citations that point to an existing page and quote its text."""
    # 构建文档页面映射表
    pages = {}
    if state.case_input:
        for document in state.case_input.documents:
            for page in document.pages:
                pages[(document.document_id, page.page)] = page.text

    # 过滤有效的来源
    sources = []
    for source in raw_sources:
        # 获取文档 ID
        document_id = str(source.get("document_id", "")).strip()
        # 获取页码
        try:
            page = int(source.get("page", 1))
        except (TypeError, ValueError):
            continue
        # 获取引用原文
        quote = str(source.get("quote", "")).strip()
        # 获取页面文本
        page_text = pages.get((document_id, page), "")
        # 如果引用为空或不在页面文本中，跳过
        if not quote or not page_text or quote not in page_text:
            continue
        # 添加有效来源
        sources.append(FactSource(document_id=document_id, page=page, quote=quote))
    return sources


# 事实提取节点函数
@trace_node("extract_facts")
def extract_facts_node(state: CaseAgentState) -> dict:
    # 构建文档文本列表
    doc_texts = []
    if state.case_input:
        for doc in state.case_input.documents:
            # 取每篇文档前 10 页，每页截取前 500 字符
            pages = "\n".join(f"[第{p.page}页] {p.text[:500]}" for p in doc.pages[:10])
            doc_texts.append(f"--- {doc.document_id} ({doc.file_name}) ---\n{pages}")

    # 加载事实提取 Prompt
    system_prompt = prompt_manager.load("fact_agent")
    # 构建消息列表
    messages = [
        {"role": "system", "content": system_prompt},  # 系统提示
        {"role": "user", "content": f"从以下劳动争议案件材料中提取全部案件事实：\n\n客户诉求: {state.case_input.client_goal if state.case_input else ''}\n\n材料正文:\n{chr(10).join(doc_texts)}"},  # 用户输入
    ]

    # 调用 LLM 进行结构化事实提取
    result, degraded, degradations, trace = llm_structured_call(
        "extract_facts", messages, FactListOutput, state,
        fallback_data=FactListOutput(facts=[]),  # 降级时返回空事实列表
    )

    # 处理提取的事实
    facts = []
    if result and result.facts:
        for f_out in result.facts:
            # 转换事实状态
            try:
                status = FactStatus(f_out.status)
            except ValueError:
                status = FactStatus.INFERRED
            # 验证事实来源
            sources = _validated_sources(state, f_out.sources or [])
            # 如果没有有效来源，状态设为材料不足
            if not sources:
                status = FactStatus.INSUFFICIENT
            # 构建事实对象
            facts.append(Fact(
                fact_id=f_out.fact_id, statement=f_out.statement,
                fact_type=f_out.fact_type, status=status,
                confidence=min(f_out.confidence, 0.49) if not sources else f_out.confidence,  # 无来源时限制置信度
                sources=sources,
                conflicts=f_out.conflicts, questions=f_out.questions,
            ))

    # 去重事实
    facts = _deduplicate_facts(facts)
    # 返回事实提取结果
    return {
        "facts": facts,                          # 事实列表
        "phase": "fact_review",                  # 下一阶段：事实复核
        "current_node": "extract_facts",         # 当前节点名称
        "degradations": degradations,            # 降级记录
        "trace": state.trace + [trace],          # 追踪信息
        "prompt_versions": {**state.prompt_versions, **prompt_manager.versions},  # Prompt 版本
    }


# 事实去重辅助函数
def _deduplicate_facts(facts: list[Fact]) -> list[Fact]:
    # 如果事实列表为空，直接返回
    if not facts:
        return facts
    # 保留第一个事实
    unique = [facts[0]]
    # 遍历剩余事实
    for f in facts[1:]:
        # 如果前 30 个字符不重复，则添加
        if not any(f.statement[:30] == u.statement[:30] for u in unique):
            unique.append(f)
    return unique
