# 模块文档字符串：事实模型
"""事实模型"""

# 启用 Python 3.10+ 的新特性支持
from __future__ import annotations

# 导入日期和时间类型
from datetime import date, datetime
# 导入可选类型提示
from typing import Optional

# 导入 Pydantic 基础模型和字段装饰器
from pydantic import BaseModel, Field

# 导入事实状态枚举
from .enums import FactStatus


# 事实来源模型
class FactSource(BaseModel):
    document_id: str                             # 文档 ID
    page: int = Field(..., ge=1)                 # 页码（从 1 开始）
    quote: str = Field(..., min_length=1)        # 引用原文


# 事实模型
class Fact(BaseModel):
    fact_id: str                                 # 事实 ID
    statement: str                               # 事实陈述
    occurred_at: Optional[date] = None           # 发生日期
    fact_type: str = ""  # termination, salary, attendance, etc.  # 事实类型
    status: FactStatus = FactStatus.CONFIRMED    # 事实状态
    confidence: float = Field(default=0.0, ge=0, le=1)  # 置信度（0-1）
    sources: list[FactSource] = Field(default_factory=list)  # 事实来源列表
    conflicts: list[str] = Field(default_factory=list)  # 冲突信息列表
    questions: list[str] = Field(default_factory=list)  # 待确认问题列表
    created_at: datetime = Field(default_factory=datetime.now)  # 创建时间