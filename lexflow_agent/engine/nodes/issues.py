# 模块文档字符串：争议焦点 Agent
"""争议焦点 Agent"""

# 启用 Python 3.10+ 的新特性支持
from __future__ import annotations

# 导入 Pydantic 基础模型和字段装饰器
from pydantic import BaseModel, Field

# 导入案件智能体状态和争议焦点模型
from lexflow_agent.engine.state import CaseAgentState, Issue
# 导入 LLM 结构化调用辅助函数
from lexflow_agent.engine.nodes.llm_helper import llm_structured_call
# 导入 Prompt 管理器
from lexflow_agent.engine.prompts.manager import prompt_manager
# 导入 LangSmith traceable 装饰器
from lexflow_agent.engine.tracing import trace_node


# 争议焦点输出模型
class IssueOutput(BaseModel):
    issue_id: str; claim: str; description: str  # ID、请求、描述
    facts_to_prove: list[str] = Field(default_factory=list)  # 需证明的事实
    priority: int = 0                            # 优先级
    research_questions: list[str] = Field(default_factory=list)  # 研究问题
    warning: str = ""                            # 警告信息


# 争议焦点列表输出模型
class IssueListOutput(BaseModel):
    issues: list[IssueOutput] = Field(default_factory=list)  # 争议焦点列表


# 默认争议焦点（降级时使用）
_DEFAULT = [IssueOutput(issue_id="default_001", claim="劳动关系确认",
    description="确认双方是否存在劳动关系", priority=5,
    research_questions=["劳动关系的认定标准是什么？"])]


# 争议焦点识别节点函数
@trace_node("identify_issues")
def identify_issues_node(state: CaseAgentState) -> dict:
    # 构建已确认事实的文本
    facts = "\n".join(f"- {f.statement}" for f in state.facts) or "(无确认事实)"
    # 加载争议焦点 Prompt
    system_prompt = prompt_manager.load("issue_agent")
    # 构建消息列表
    messages = [
        {"role": "system", "content": system_prompt},  # 系统提示
        {"role": "user", "content": f"客户诉求: {state.case_input.client_goal if state.case_input else ''}\n确认事实:\n{facts}"},  # 用户输入
    ]
    # 调用 LLM 进行争议焦点识别
    result, degraded, degs, trace = llm_structured_call(
        "identify_issues", messages, IssueListOutput, state,
        fallback_data=IssueListOutput(issues=_DEFAULT),  # 降级时使用默认焦点
    )
    # 转换输出为内部 Issue 模型
    issues = [Issue(issue_id=i.issue_id, claim=i.claim, description=i.description,
        facts_to_prove=i.facts_to_prove, priority=i.priority,
        research_questions=i.research_questions, warning=i.warning)
        for i in (result.issues if result else [])]
    # 返回争议焦点识别结果
    return {"issues": issues, "phase": "evidence", "current_node": "identify_issues",
        "degradations": degs, "trace": state.trace + [trace],
        "prompt_versions": {**state.prompt_versions, **prompt_manager.versions}}