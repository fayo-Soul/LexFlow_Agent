"""Four-agent LangGraph orchestrator - 四 Agent 工作流编排器

本模块使用 LangGraph 构建和编译法律案件分析的工作流图。
工作流包含四个主要 Agent：
1. case_understanding_agent - 案件理解（材料分析、事实提取、争议焦点识别）
2. legal_reseat - 法律研究（法条检索、案例检索）
3. case_strategy_agent - 案件策略（证据分析、策略生成）
4. document_generation_agent - 文rch_agen书生成（起诉状、答辩状等）

工作流支持：
- 条件分支（根据状态决定下一步）
- 循环迭代（支持修订和重新生成）
- 人工干预（当自动处理失败时）
- 检查点保存（支持断点续传）
"""

from __future__ import annotations  # 启用未来类型注解

import sqlite3  # SQLite 数据库（用于检查点持久化）

from langgraph.checkpoint.sqlite import SqliteSaver  # LangGraph SQLite 检查点保存器
from langgraph.graph import END, StateGraph  # LangGraph 图构建工具

from lexflow_agent.config.settings import settings  # 全局配置
from lexflow_agent.engine.agents import (  # 导入四个 Agent 构建函数
    build_case_strategy_agent,
    build_case_understanding_agent,
    build_document_generation_agent,
    build_legal_research_agent,
)
from lexflow_agent.engine.models.enums import WorkflowStatus  # 工作流状态枚举
from lexflow_agent.engine.state import CaseAgentState  # 全局状态模型


def _mark_agent(name: str, phase: str):
    """创建标记节点 - 用于记录当前执行的 Agent 和阶段
    
    Args:
        name: Agent 名称
        phase: 当前阶段名称
    
    Returns:
        一个节点函数，执行时更新 state 的 current_agent 和 phase 字段
    """
    def node(state: CaseAgentState) -> dict:
        return {"current_agent": name, "phase": phase}
    return node


def _after_understanding(state: CaseAgentState) -> str:
    """案件理解后的路由决策 - 决定下一步是继续还是结束
    
    检查案件理解阶段是否成功：
    - 如果失败或有错误，直接结束工作流
    - 如果成功，进入事实复核阶段
    
    Args:
        state: 当前工作流状态
    
    Returns:
        下一个节点名称（"fact_review" 或 END）
    """
    if state.status == WorkflowStatus.FAILED or state.errors:
        return END
    return "fact_review"


def _after_document_review(state: CaseAgentState) -> str:
    """文书复核后的路由决策 - 决定是结束、重试还是人工干预
    
    复杂的条件判断逻辑：
    1. 检查文书数量是否 >= 3 篇
    2. 检查文书内容是否完整（非空、无"生成失败"标记）
    3. 检查确定性复核是否通过
    4. 如果上述任一不满足，进入人工干预
    
    如果复核通过：
    - 语义复核通过或无复核 → 结束
    - 达到最大修订次数 → 人工干预
    - 否则根据错误类型路由到对应 Agent 重新生成
    
    Args:
        state: 当前工作流状态
    
    Returns:
        下一个节点名称（Agent 名、"human_intervention" 或 END）
    """
    if (
        len(state.drafts) < 3  # 文书数量不足
        or any(not draft.full_text.strip() or "生成失败" in draft.full_text for draft in state.drafts)  # 文书内容不完整
        or not state.deterministic_review  # 无确定性复核结果
        or not state.deterministic_review.passed  # 确定性复核未通过
    ):
        return "human_intervention"
    review = state.semantic_review
    if not review or review.passed:  # 无语义复核或复核通过
        return END
    if state.revision_count >= state.max_revision_count:  # 达到最大修订次数
        return "human_intervention"

    # 根据错误类型决定路由到哪个 Agent
    target = "case_understanding_agent"  # 默认路由
    for finding in review.findings:
        if finding.severity != "error":  # 只处理错误级别的发现
            continue
        if finding.node in {"research_law"}:  # 法律研究错误 → 重新研究
            target = "legal_research_agent"
        elif finding.node in {"generate_strategy", "analyze_evidence"}:  # 策略错误 → 重新生成策略
            target = "case_strategy_agent"
        elif finding.node in {"draft_documents"}:  # 文书错误 → 重新生成文书
            target = "document_generation_agent"
        break
    return target


def _increment_revision(state: CaseAgentState) -> dict:
    """增加修订计数 - 在重新生成前调用
    
    Args:
        state: 当前工作流状态
    
    Returns:
        状态更新字典，将 revision_count 加 1
    """
    return {"revision_count": state.revision_count + 1}


def _human_intervention(state: CaseAgentState) -> dict:
    """人工干预节点 - 将状态设置为等待人工处理
    
    当自动处理无法继续时（如文书质量不达标、超过修订次数等），
    工作流进入人工干预状态，等待用户手动介入。
    
    Args:
        state: 当前工作流状态
    
    Returns:
        状态更新字典，设置 status、current_node 和 phase
    """
    return {
        "status": WorkflowStatus.WAITING_HUMAN_INTERVENTION,  # 等待人工干预状态
        "current_node": "human_intervention",  # 当前节点
        "phase": "human_intervention",  # 当前阶段
    }


def _build_graph() -> StateGraph:
    """构建 LangGraph 工作流图 - 定义所有节点和边
    
    工作流图结构：
    1. 入口：case_understanding_agent（案件理解）
    2. 条件分支：理解成功后进入 fact_review，失败则结束
    3. 顺序执行：fact_review → legal_research → case_strategy → strategy_review
    4. 条件分支：document_generation 后根据复核结果决定下一步
       - 通过 → END
       - 失败 → increment_revision → 路由到对应 Agent 重新生成
       - 超过修订次数 → human_intervention → END
    
    Returns:
        编译好的 StateGraph 对象
    """
    builder = StateGraph(CaseAgentState)  # 创建图构建器
    # 添加所有节点
    builder.add_node("case_understanding_agent", build_case_understanding_agent())  # 案件理解 Agent
    builder.add_node("fact_review", _mark_agent("case_strategy_agent", "fact_review"))  # 事实复核标记节点
    builder.add_node("legal_research_agent", build_legal_research_agent())  # 法律研究 Agent
    builder.add_node("case_strategy_agent", build_case_strategy_agent())  # 案件策略 Agent
    builder.add_node("strategy_review", _mark_agent("case_strategy_agent", "strategy_review"))  # 策略复核标记节点
    builder.add_node("document_generation_agent", build_document_generation_agent())  # 文书生成 Agent
    builder.add_node("increment_revision", _increment_revision)  # 修订计数节点
    builder.add_node("human_intervention", _human_intervention)  # 人工干预节点

    # 设置入口点
    builder.set_entry_point("case_understanding_agent")
    
    # 案件理解后的条件分支
    builder.add_conditional_edges(
        "case_understanding_agent",
        _after_understanding,
        {"fact_review": "fact_review", END: END},  # 成功→fact_review，失败→END
    )
    
    # 顺序执行链路
    builder.add_edge("fact_review", "legal_research_agent")  # 事实复核后进入法律研究
    builder.add_edge("legal_research_agent", "case_strategy_agent")  # 法律研究后进入策略生成
    builder.add_edge("case_strategy_agent", "strategy_review")  # 策略生成后进入策略复核
    builder.add_edge("strategy_review", "document_generation_agent")  # 策略复核后进入文书生成
    
    # 文书生成后的条件分支
    builder.add_conditional_edges(
        "document_generation_agent",
        _after_document_review,
        {
            "case_understanding_agent": "increment_revision",  # 重新理解 → 先增加修订计数
            "legal_research_agent": "increment_revision",  # 重新研究 → 先增加修订计数
            "case_strategy_agent": "increment_revision",  # 重新策略 → 先增加修订计数
            "document_generation_agent": "increment_revision",  # 重新生成 → 先增加修订计数
            "human_intervention": "human_intervention",  # 人工干预
            END: END,  # 完成
        },
    )
    
    # 修订计数后的条件分支（实际路由到对应 Agent）
    builder.add_conditional_edges(
        "increment_revision",
        _after_document_review,
        {
            "case_understanding_agent": "case_understanding_agent",  # 路由到案件理解
            "legal_research_agent": "legal_research_agent",  # 路由到法律研究
            "case_strategy_agent": "case_strategy_agent",  # 路由到策略生成
            "document_generation_agent": "document_generation_agent",  # 路由到文书生成
            "human_intervention": "human_intervention",  # 路由到人工干预
            END: END,  # 完成
        },
    )
    
    # 人工干预后结束
    builder.add_edge("human_intervention", END)
    return builder


_checkpoint_connection: sqlite3.Connection | None = None  # 全局检查点数据库连接


def compile_graph():
    """编译工作流图 - 创建可执行的 LangGraph 应用
    
    功能：
    1. 建立 SQLite 数据库连接（用于检查点持久化）
    2. 编译 StateGraph 为可执行应用
    3. 配置检查点保存器
    4. 设置中断点（在关键节点前暂停，允许人工审核）
    
    Returns:
        编译好的 LangGraph 应用对象
    """
    global _checkpoint_connection
    # 从配置中获取数据库路径（移除 "sqlite:///" 前缀）
    db_path = settings.CHECKPOINT_DB_URL.removeprefix("sqlite:///")
    # 建立数据库连接（check_same_thread=False 允许多线程访问）
    _checkpoint_connection = sqlite3.connect(db_path, check_same_thread=False)
    # 编译图并配置检查点和中断点
    return _build_graph().compile(
        checkpointer=SqliteSaver(_checkpoint_connection),  # SQLite 检查点保存器
        interrupt_before=["legal_research_agent", "document_generation_agent"],  # 在研究和生成前中断（允许人工审核）
    )


_app = None  # 全局单例应用对象


def get_graph():
    """获取或创建 LangGraph 应用单例
    
    使用懒加载模式，首次调用时编译图，后续调用返回缓存的应用对象。
    
    Returns:
        编译好的 LangGraph 应用对象
    """
    global _app
    if _app is None:
        _app = compile_graph()
    return _app


def delete_checkpoint(thread_id: str) -> None:
    """删除指定线程的检查点数据
    
    当 API 运行完成或被删除时，清理对应的 LangGraph 检查点数据，
    避免数据库膨胀。
    
    Args:
        thread_id: 线程 ID（对应 API run_id）
    """
    graph = get_graph()
    graph.checkpointer.delete_thread(thread_id)