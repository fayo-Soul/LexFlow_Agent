# 模块文档字符串：证据分析 Agent
"""证据分析 Agent"""

# 启用 Python 3.10+ 的新特性支持
from __future__ import annotations

# 导入 Pydantic 基础模型和字段装饰器
from pydantic import BaseModel, Field

# 导入案件智能体状态
from lexflow_agent.engine.state import CaseAgentState
# 导入证据矩阵相关模型
from lexflow_agent.engine.models.evidence import EvidenceItem, EvidenceSource, EvidenceStrength
# 导入 LLM 结构化调用辅助函数
from lexflow_agent.engine.nodes.llm_helper import llm_structured_call
# 导入 Prompt 管理器
from lexflow_agent.engine.prompts.manager import prompt_manager
# 导入 LangSmith traceable 装饰器
from lexflow_agent.engine.tracing import trace_node


# 证据来源输出模型
class EvSourceOutput(BaseModel):
    evidence_id: str; document_id: str; page: int = 1; evidence_name: str  # ID、文档、页码、名称
    proves: list[str] = Field(default_factory=list)  # 证明内容
    strengths: list[str] = Field(default_factory=list)  # 优势
    risks: list[str] = Field(default_factory=list)  # 风险


# 证据输出模型
class EvidenceOutput(BaseModel):
    issue_id: str; claim: str; fact_to_prove: str  # 焦点 ID、请求、待证事实
    evidence: list[EvSourceOutput] = Field(default_factory=list)  # 证据来源
    status: str = "partial"                    # 证据状态
    missing_materials: list[str] = Field(default_factory=list)  # 缺失材料
    risks: list[str] = Field(default_factory=list)  # 风险


# 证据列表输出模型
class EvidenceListOutput(BaseModel):
    matrix: list[EvidenceOutput] = Field(default_factory=list)  # 证据矩阵


# 证据分析节点函数
@trace_node("analyze_evidence")
def analyze_evidence_node(state: CaseAgentState) -> dict:
    # 构建争议焦点文本
    issues = "\n".join(f"- {i.claim}: {i.description}" for i in state.issues) or "(无)"
    # 构建文档文本
    doc_texts = []
    if state.case_input:
        for doc in state.case_input.documents:
            # 取每篇文档前 5 页，每页截取前 300 字符
            pages = "\n".join(f"[P{p.page}] {p.text[:300]}" for p in doc.pages[:5])
            doc_texts.append(f"--- {doc.document_id} ---\n{pages}")
    # 加载证据分析 Prompt
    system_prompt = prompt_manager.load("evidence_agent")

    # 注入超时保护
    import asyncio
    async def _call():
        # 构建消息列表
        messages = [
            {"role": "system", "content": system_prompt},  # 系统提示
            {"role": "user", "content": f"案件文档:\n{chr(10).join(doc_texts)}\n\n争议焦点:\n{issues}"},  # 用户输入
        ]
        # 调用 LLM 进行证据分析
        result, degraded, degs, trace = llm_structured_call(
            "analyze_evidence", messages, EvidenceListOutput, state,
            fallback_data=EvidenceListOutput(matrix=[]),  # 降级时返回空矩阵
        )
        return result, degraded, degs, trace

    # 用超时链包裹异步调用（超时 300 秒）
    from lexflow_agent.engine.runtime import execute_with_timeout
    result, degraded, degs, trace = asyncio.run(
        execute_with_timeout(_call(), 300)
    )

    # 处理证据矩阵
    matrix = []
    if result and result.matrix:
        for e in result.matrix:
            # 转换证据状态
            try:
                status = EvidenceStrength(e.status)
            except ValueError:
                status = EvidenceStrength.PARTIAL
            # 构建证据项
            matrix.append(EvidenceItem(issue_id=e.issue_id, claim=e.claim,
                fact_to_prove=e.fact_to_prove,
                evidence=[EvidenceSource(**ev.model_dump()) for ev in e.evidence],
                status=status, missing_materials=e.missing_materials, risks=e.risks))

    # 返回证据分析结果
    return {"evidence_matrix": matrix, "current_node": "analyze_evidence",
        "degradations": degs, "trace": state.trace + [trace],
        "prompt_versions": {**state.prompt_versions, **prompt_manager.versions}}