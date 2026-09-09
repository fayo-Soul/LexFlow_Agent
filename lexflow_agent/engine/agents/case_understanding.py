# 模块文档字符串：案件理解智能体，负责并行材料审查和事实提取
"""Case Understanding Agent: parallel material review and fact extraction."""

# 从 langgraph.graph 导入图构建所需的 END 常量和 StateGraph 类
from langgraph.graph import END, StateGraph

# 从工作流状态枚举模块导入 WorkflowStatus 枚举类
from lexflow_agent.engine.models.enums import WorkflowStatus
# 从事实节点模块导入事实提取节点函数
from lexflow_agent.engine.nodes.facts import extract_facts_node
# 从材料节点模块导入材料分析节点函数
from lexflow_agent.engine.nodes.material import analyze_materials_node
# 从验证节点模块导入输入验证节点函数
from lexflow_agent.engine.nodes.validation import validate_input_node
# 从状态模块导入案件智能体状态类
from lexflow_agent.engine.state import CaseAgentState


# 定义验证后的路由函数，决定下一步执行哪个节点
def _after_validation(state: CaseAgentState) -> str:
    # 如果状态为失败或存在错误，则返回 END 结束流程
    return END if state.status == WorkflowStatus.FAILED or state.errors else "analyze_materials"


# 定义构建案件理解智能体的工厂函数
def build_case_understanding_agent():
    # 创建基于 CaseAgentState 的状态图构建器
    builder = StateGraph(CaseAgentState)
    # 添加起始节点，设置当前智能体标识为案件理解智能体
    builder.add_node("start", lambda state: {"current_agent": "case_understanding_agent"})
    # 添加输入验证节点，用于验证用户输入的有效性
    builder.add_node("validate_input", validate_input_node)
    # 添加材料分析节点，用于分析案件相关材料
    builder.add_node("analyze_materials", analyze_materials_node)
    # 添加事实提取节点，用于从材料中提取关键事实
    builder.add_node("extract_facts", extract_facts_node)
    # 设置图的入口点为起始节点
    builder.set_entry_point("start")
    # 添加从起始节点到输入验证节点的有向边
    builder.add_edge("start", "validate_input")
    # 添加从验证节点出发的条件边，根据验证结果决定下一步
    builder.add_conditional_edges(
        "validate_input",           # 条件边的源节点
        _after_validation,          # 路由函数，决定下一步走向
        {"analyze_materials": "analyze_materials", END: END},  # 路由映射：成功则进入材料分析，失败则结束
    )
    # 添加从材料分析节点到事实提取节点的有向边
    builder.add_edge("analyze_materials", "extract_facts")
    # 添加从事实提取节点到结束节点的有向边
    builder.add_edge("extract_facts", END)
    # 编译并返回构建好的状态图
    return builder.compile()