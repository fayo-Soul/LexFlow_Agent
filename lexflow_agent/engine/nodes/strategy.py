# 模块文档字符串：案件策略 Agent
"""案件策略 Agent"""

# 启用 Python 3.10+ 的新特性支持
from __future__ import annotations

# 导入 Pydantic 基础模型和字段装饰器
from pydantic import BaseModel, Field

# 导入案件智能体状态
from lexflow_agent.engine.state import CaseAgentState
# 导入风险等级和工作流状态枚举
from lexflow_agent.engine.models.enums import RiskLevel, WorkflowStatus
# 导入策略结果和策略方案模型
from lexflow_agent.engine.models.strategy import StrategyResult, StrategyPlan
# 导入 LLM 结构化调用辅助函数
from lexflow_agent.engine.nodes.llm_helper import llm_structured_call
# 导入 Prompt 管理器
from lexflow_agent.engine.prompts.manager import prompt_manager
# 导入 LangSmith traceable 装饰器
from lexflow_agent.engine.tracing import trace_node


# 策略方案输出模型
class PlanOutput(BaseModel):
    name: str = ""; description: str           # 名称、描述
    claims: list[str] = Field(default_factory=list)  # 诉讼请求
    legal_basis: list[str] = Field(default_factory=list)  # 法律依据
    evidence_summary: list[str] = Field(default_factory=list)  # 证据摘要
    risk_level: str = "insufficient"           # 风险等级
    opponent_defenses: list[str] = Field(default_factory=list)  # 对方抗辩
    counter_paths: list[str] = Field(default_factory=list)  # 应对路径


# 策略输出模型
class StrategyOutput(BaseModel):
    primary_plan: PlanOutput                   # 主要方案
    alternative_plans: list[PlanOutput] = Field(default_factory=list)  # 备选方案
    risks: list[dict] = Field(default_factory=list)  # 风险列表
    pending_confirmations: list[str] = Field(default_factory=list)  # 待确认事项


# 降级策略构建函数
def _fallback_strategy(state: CaseAgentState) -> StrategyResult:
    """Build a conservative, non-LLM strategy without inventing legal authority."""
    # 获取有证据支持的请求
    supported_claims = [item.claim for item in state.evidence_matrix if item.claim]
    # 获取已验证的法律依据
    verified_basis = [
        citation.title
        for research in state.legal_research
        if research.citations_valid
        for citation in research.favorable_sources
        if citation.title
    ]
    # 返回保守的降级策略
    return StrategyResult(
        primary_plan=StrategyPlan(
            name="保守处理方案（降级）",
            description="基于已确认事实整理现有请求；因策略模型不可用，具体请求、金额及法律依据须由律师确认。",
            claims=supported_claims,
            legal_basis=list(dict.fromkeys(verified_basis)),
            evidence_summary=[
                f"{item.claim}: {item.status.value}" for item in state.evidence_matrix
            ],
            risk_level=RiskLevel.INSUFFICIENT,
        ),
        risks=[{
            "type": "model_degradation",
            "level": "high",
            "desc": "策略模型不可用，未生成确定性诉讼结论",
        }],
        pending_confirmations=["请律师确认请求、金额、法律依据和证据缺口"],
    )


# 策略生成节点函数
@trace_node("generate_strategy")
def generate_strategy_node(state: CaseAgentState) -> dict:
    # 构建事实、证据矩阵和法律研究的文本
    facts = "\n".join(f"- {f.statement}" for f in state.facts) or "(无)"
    ev = "\n".join(f"- {e.claim}: {e.status.value}" for e in state.evidence_matrix) or "(无)"
    research = "\n".join(f"- {r.conclusion}" for r in state.legal_research) or "(无)"

    # 加载策略生成 Prompt
    system_prompt = prompt_manager.load("strategy_agent")
    # 构建消息列表
    messages = [
        {"role": "system", "content": system_prompt},  # 系统提示
        {"role": "user", "content": f"确认事实:\n{facts}\n\n证据矩阵:\n{ev}\n\n法律研究:\n{research}"},  # 用户输入
    ]
    # 调用 LLM 生成策略
    result, degraded, degs, trace = llm_structured_call(
        "generate_strategy", messages, StrategyOutput, state, fallback_data=None,
    )

    # 处理策略结果
    strategy = None
    if result:
        # 定义方案解析函数
        def parse_plan(p: PlanOutput) -> StrategyPlan:
            # 转换风险等级
            try:
                rl = RiskLevel(p.risk_level) if p.risk_level in ("low","medium","high","insufficient") else RiskLevel.INSUFFICIENT
            except ValueError:
                rl = RiskLevel.INSUFFICIENT
            # 构建策略方案
            return StrategyPlan(name=p.name, description=p.description, claims=p.claims,
                legal_basis=p.legal_basis, evidence_summary=p.evidence_summary,
                risk_level=rl, opponent_defenses=p.opponent_defenses, counter_paths=p.counter_paths)
        # 构建策略结果
        strategy = StrategyResult(
            primary_plan=parse_plan(result.primary_plan),
            alternative_plans=[parse_plan(p) for p in result.alternative_plans],
            risks=result.risks, pending_confirmations=result.pending_confirmations,
        )
    else:
        # 如果 LLM 调用失败，使用降级策略
        strategy = _fallback_strategy(state)

    # 返回策略生成结果
    return {"strategy": strategy, "status": WorkflowStatus.WAITING_STRATEGY_REVIEW,
        "current_node": "generate_strategy", "degradations": degs,
        "trace": state.trace + [trace],
        "prompt_versions": {**state.prompt_versions, **prompt_manager.versions}}