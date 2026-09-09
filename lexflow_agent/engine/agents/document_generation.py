# 模块文档字符串：文档生成智能体，负责起草和两阶段质量审查
"""Document Generation Agent: drafting and two-stage quality review.

图结构：
start → clarify_input → [if missing data] → wait_clarification → END (等待澄清)
                       → [if ready] → draft_documents → deterministic_review
                                                       → semantic_review → END
"""

# 从 langgraph.graph 导入图构建所需的 END 常量和 StateGraph 类
from langgraph.graph import END, StateGraph

# 从确定性审查节点模块导入确定性审查节点函数
from lexflow_agent.engine.nodes.deterministic_review import deterministic_review_node
# 从起草节点模块导入文档起草节点函数
from lexflow_agent.engine.nodes.drafting import draft_documents_node
# 从语义审查节点模块导入语义审查节点函数
from lexflow_agent.engine.nodes.semantic_review import semantic_review_node
# 从状态模块导入案件智能体状态类
from lexflow_agent.engine.state import CaseAgentState
# 从工作流状态枚举导入
from lexflow_agent.engine.models.enums import WorkflowStatus


# ★新增：澄清检查节点
def clarify_input_node(state: CaseAgentState) -> dict:
    """检查文档生成所需的前置数据是否齐全

    如果 facts 或 strategy 缺失，返回澄清询问而不是失败。
    """
    missing: list[dict] = []

    if not state.facts:
        missing.append({
            "question_id": "missing_facts",
            "question": "请提供案件的基本事实信息：什么时间、发生了什么事、涉及哪些人？",
            "field": "facts",
            "expected_type": "text",
        })

    if not state.strategy:
        missing.append({
            "question_id": "missing_strategy",
            "question": "请提供案件策略或分析结论：诉讼请求、法律依据、证据方向？",
            "field": "strategy",
            "expected_type": "text",
        })

    if missing:
        return {
            "status": WorkflowStatus.WAITING_CLARIFICATION,
            "current_node": "clarify_input",
            "phase": "clarification",
            "clarification_questions": missing,
            "clarification_answers": {},
            "warnings": state.warnings + [
                "文书生成 Agent 缺少前置数据，等待用户澄清"
            ],
        }

    return {
        "current_node": "clarify_input",
        "phase": "drafting",
    }


def _after_clarify(state: CaseAgentState) -> str:
    """澄清检查后的路由：缺数据则等待，齐备则起草"""
    if state.status == WorkflowStatus.WAITING_CLARIFICATION:
        return "wait_clarification"
    return "draft_documents"


def _wait_clarification(state: CaseAgentState) -> dict:
    """等待澄清节点 — 标记为等待用户回答"""
    return {
        "status": WorkflowStatus.WAITING_CLARIFICATION,
        "current_node": "wait_clarification",
        "phase": "clarification",
    }


# 定义构建文档生成智能体的工厂函数
def build_document_generation_agent():
    # 创建基于 CaseAgentState 的状态图构建器
    builder = StateGraph(CaseAgentState)
    # 添加起始节点，设置当前智能体标识为文档生成智能体
    builder.add_node("start", lambda state: {"current_agent": "document_generation_agent"})
    # ★新增：澄清检查节点
    builder.add_node("clarify_input", clarify_input_node)
    # ★新增：等待用户澄清节点
    builder.add_node("wait_clarification", _wait_clarification)
    # 添加文档起草节点，用于生成法律文书初稿
    builder.add_node("draft_documents", draft_documents_node)
    # 添加确定性审查节点，进行格式和规则层面的审查
    builder.add_node("deterministic_review", deterministic_review_node)
    # 添加语义审查节点，进行语义和内容层面的审查
    builder.add_node("semantic_review", semantic_review_node)
    # 设置图的入口点为起始节点
    builder.set_entry_point("start")
    # 添加从起始节点到澄清检查节点的有向边
    builder.add_edge("start", "clarify_input")
    # ★条件边：根据澄清结果决定下一步
    builder.add_conditional_edges(
        "clarify_input",
        _after_clarify,
        {"draft_documents": "draft_documents", "wait_clarification": "wait_clarification"},
    )
    # 等待澄清后结束（前端通过 resume 继续）
    builder.add_edge("wait_clarification", END)
    # 添加从文档起草节点到确定性审查节点的有向边
    builder.add_edge("draft_documents", "deterministic_review")
    # 添加从确定性审查节点到语义审查节点的有向边
    builder.add_edge("deterministic_review", "semantic_review")
    # 添加从语义审查节点到结束节点的有向边
    builder.add_edge("semantic_review", END)
    # 编译并返回构建好的状态图
    return builder.compile()