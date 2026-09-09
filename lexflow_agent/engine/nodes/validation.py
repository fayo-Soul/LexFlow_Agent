# 模块文档字符串：输入校验节点 - 确定性代码
"""输入校验节点 - 确定性代码"""

# 启用 Python 3.10+ 的新特性支持
from __future__ import annotations

# 导入案件智能体状态类
from lexflow_agent.engine.state import CaseAgentState
# 导入案件输入模型
from lexflow_agent.engine.models.case import CaseInput
# 导入工作流状态枚举
from lexflow_agent.engine.models.enums import WorkflowStatus
# 导入 LangSmith traceable 装饰器
from lexflow_agent.engine.tracing import trace_node


# 输入校验节点函数
@trace_node("validate_input", run_type="tool")
def validate_input_node(state: CaseAgentState) -> dict:
    """输入校验节点 - 确保无效输入不进入LLM节点"""
    # 获取案件输入
    case_input = state.case_input
    # 如果输入为空，返回失败状态
    if not case_input:
        return {
            "status": WorkflowStatus.FAILED,     # 状态设为失败
            "errors": ["案件输入为空"],           # 错误信息
            "current_node": "validate_input",    # 当前节点名称
        }

    # 初始化错误和警告列表
    errors = []
    warnings = []

    # 校验案由是否支持
    if case_input.case_type.value not in ("labor_dispute",):
        errors.append(f"不支持的案由: {case_input.case_type.value}")

    # 校验文档 ID 唯一性
    doc_ids = [d.document_id for d in case_input.documents]
    if len(doc_ids) != len(set(doc_ids)):
        errors.append("文档 ID 存在重复")

    # 校验页码和正文
    for doc in case_input.documents:
        # 检查页码是否重复
        page_nums = [p.page for p in doc.pages]
        if len(page_nums) != len(set(page_nums)):
            errors.append(f"文档 {doc.document_id} 存在重复页码")
        # 检查每页正文是否为空
        for page in doc.pages:
            if not page.text.strip():
                warnings.append(f"文档 {doc.document_id} 第{page.page}页正文为空")

    # 校验客户诉求不能为空
    if not case_input.client_goal.strip():
        errors.append("客户诉求不能为空")

    # 校验材料体积限制
    total_chars = sum(len(p.text) for d in case_input.documents for p in d.pages)
    if total_chars > 500_000:
        errors.append(f"材料总字符数 {total_chars} 超过限制 500,000")
    if len(case_input.documents) > 20:
        errors.append(f"文档数 {len(case_input.documents)} 超过限制 20")

    # 如果存在错误，返回失败状态
    if errors:
        return {
            "status": WorkflowStatus.FAILED,     # 状态设为失败
            "errors": errors,                    # 错误列表
            "warnings": warnings,                # 警告列表
            "current_node": "validate_input",    # 当前节点名称
        }

    # 校验通过，返回运行状态
    return {
        "status": WorkflowStatus.RUNNING,        # 状态设为运行中
        "phase": "material",                     # 当前阶段：材料分析
        "current_node": "validate_input",        # 当前节点名称
        "warnings": state.warnings + warnings,   # 合并已有警告和新警告
    }