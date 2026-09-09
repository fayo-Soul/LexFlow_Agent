# 模块文档字符串：策略模型
"""策略模型"""

# 启用 Python 3.10+ 的新特性支持
from __future__ import annotations

# 导入 Pydantic 基础模型和字段装饰器
from pydantic import BaseModel, Field

# 导入风险等级枚举
from .enums import RiskLevel


# 策略方案模型
class StrategyPlan(BaseModel):
    name: str = ""                               # 方案名称
    description: str                             # 方案描述
    claims: list[str] = Field(default_factory=list)  # 诉讼请求列表
    legal_basis: list[str] = Field(default_factory=list)  # 法律依据列表
    evidence_summary: list[str] = Field(default_factory=list)  # 证据摘要
    risk_level: RiskLevel = RiskLevel.INSUFFICIENT  # 风险等级
    opponent_defenses: list[str] = Field(default_factory=list)  # 对方可能的抗辩
    counter_paths: list[str] = Field(default_factory=list)  # 应对路径


# 策略结果模型
class StrategyResult(BaseModel):
    primary_plan: StrategyPlan                   # 主要策略方案
    alternative_plans: list[StrategyPlan] = Field(default_factory=list)  # 备选策略方案列表
    risks: list[dict] = Field(default_factory=list)  # 风险列表 [{"type":"jurisdiction","level":"medium","desc":""}]
    pending_confirmations: list[str] = Field(default_factory=list)  # 待确认事项列表
    approved: bool = False                       # 是否已批准