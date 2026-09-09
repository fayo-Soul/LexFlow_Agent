# 模块文档字符串：争议焦点模型
"""争议焦点模型"""

# 启用 Python 3.10+ 的新特性支持
from __future__ import annotations

# 导入 Pydantic 基础模型和字段装饰器
from pydantic import BaseModel, Field


# 争议焦点模型
class Issue(BaseModel):
    issue_id: str                                # 争议焦点 ID
    claim: str = Field(..., description="客户的具体请求")  # 客户的具体请求
    description: str = Field(..., description="争议焦点描述")  # 争议焦点描述
    facts_to_prove: list[str] = Field(default_factory=list)  # 需要证明的事实列表
    priority: int = Field(default=0, ge=0, le=5)  # 优先级（0-5）
    research_questions: list[str] = Field(default_factory=list)  # 研究问题列表
    warning: str = ""                            # 警告信息