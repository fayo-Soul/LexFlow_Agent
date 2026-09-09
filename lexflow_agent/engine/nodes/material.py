# 模块文档字符串：材料理解节点 - 多文档并行 LLM 分析
"""材料理解节点 - 多文档并行 LLM 分析"""

# 启用 Python 3.10+ 的新特性支持
from __future__ import annotations

# 导入线程池执行器，用于并行处理
from concurrent.futures import ThreadPoolExecutor, as_completed
# 导入可选类型提示
from typing import Optional

# 导入 Pydantic 基础模型和字段装饰器
from pydantic import BaseModel, Field

# 导入案件智能体状态和材料分析模型
from lexflow_agent.engine.state import CaseAgentState, MaterialAnalysis
# 导入 LLM 结构化调用辅助函数
from lexflow_agent.engine.nodes.llm_helper import llm_structured_call
# 导入 Prompt 管理器
from lexflow_agent.engine.prompts.manager import prompt_manager
# 导入 LangSmith traceable 装饰器
from lexflow_agent.engine.tracing import trace_node


# 材料分析输出模型
class MaterialOutput(BaseModel):
    document_type: str = ""                    # 文档类型
    summary: str = ""                          # 文档摘要
    parties: list[str] = Field(default_factory=list)  # 当事人列表
    dates: list[dict] = Field(default_factory=list)   # 日期信息列表
    amounts: list[dict] = Field(default_factory=list) # 金额信息列表
    relevance: float = 0.0                     # 相关性评分
    warnings: list[str] = Field(default_factory=list) # 警告列表


# 构建材料分析 Prompt 的辅助函数
def _build_material_prompt(doc) -> list[dict]:
    # 拼接所有页面文本
    pages_text = "\n".join(f"[第{p.page}页] {p.text}" for p in doc.pages)
    # 加载材料分析 Prompt 模板
    system_prompt = prompt_manager.load("material_agent")
    # 构建消息列表
    return [
        {"role": "system", "content": system_prompt},  # 系统提示
        {"role": "user", "content": f"分析以下劳动争议案件文档：\n\n文档ID: {doc.document_id}\n文件名: {doc.file_name}\n\n文档正文:\n{pages_text}"},  # 用户输入
    ]


# 材料分析节点函数
@trace_node("analyze_materials")
def analyze_materials_node(state: CaseAgentState) -> dict:
    # 获取案件输入
    case_input = state.case_input
    # 如果没有输入或没有文档，返回警告
    if not case_input or not case_input.documents:
        return {"warnings": state.warnings + ["没有待分析文档"]}

    # 初始化分析结果、警告、降级记录和追踪列表
    analyses = list(state.material_analyses)
    warnings = list(state.warnings)
    all_degradations = list(state.degradations)
    all_traces = list(state.trace)

    # 定义单个文档分析函数
    def _analyze_one(doc) -> Optional[MaterialOutput]:
        # 构建 Prompt
        messages = _build_material_prompt(doc)
        # 调用 LLM 进行结构化分析
        result, degraded, degs, trace = llm_structured_call(
            "analyze_materials", messages, MaterialOutput, state,
            fallback_data=None,  # 无降级数据
        )
        # 收集降级记录和追踪信息
        all_degradations.extend(degs)
        all_traces.append(trace)
        return result

    # 使用线程池并行分析所有文档
    with ThreadPoolExecutor(max_workers=5) as executor:
        # 提交所有分析任务
        futures = {executor.submit(_analyze_one, doc): doc for doc in case_input.documents}
        # 收集分析结果
        for future in as_completed(futures):
            doc = futures[future]
            try:
                output = future.result()
                # 如果分析结果为空，记录警告
                if output is None:
                    warnings.append(f"文档 {doc.document_id} 分析降级")
                else:
                    # 将分析结果添加到列表中
                    analyses.append(MaterialAnalysis(
                        document_id=doc.document_id,
                        document_type=output.document_type or "other",
                        summary=output.summary,
                        parties=output.parties,
                        dates=output.dates,
                        amounts=output.amounts,
                        relevance=output.relevance,
                        warnings=output.warnings,
                    ))
            except Exception as e:
                # 捕获异常并记录警告
                warnings.append(f"文档 {doc.document_id} 分析异常: {e}")

    # 返回分析结果和状态更新
    return {
        "material_analyses": analyses,           # 材料分析结果列表
        "phase": "facts",                        # 下一阶段：事实提取
        "current_node": "analyze_materials",     # 当前节点名称
        "warnings": warnings,                    # 警告列表
        "degradations": all_degradations,        # 降级记录列表
        "trace": all_traces,                     # 追踪信息列表
        "prompt_versions": {**state.prompt_versions, **prompt_manager.versions},  # Prompt 版本
    }