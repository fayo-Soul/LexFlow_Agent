# 模块文档字符串：复核模型
"""复核模型"""

# 启用 Python 3.10+ 的新特性支持
from __future__ import annotations

# 导入日期时间模块
from datetime import datetime
# 导入可选类型提示
from typing import Optional

# 导入 Pydantic 基础模型和字段装饰器
from pydantic import BaseModel, Field


# 复核发现项模型
class ReviewFinding(BaseModel):
    severity: str  # error | warning | info      # 严重程度
    node: str = ""                               # 关联节点
    field: str = ""                              # 关联字段
    description: str                             # 问题描述
    suggestion: str = ""                         # 修复建议


# 复核结果模型
class ReviewResult(BaseModel):
    findings: list[ReviewFinding] = Field(default_factory=list)  # 发现项列表
    passed: bool = True                          # 是否通过复核
    reviewed_at: datetime = Field(default_factory=datetime.now)  # 复核时间


# 降级记录模型
class DegradationRecord(BaseModel):
    node: str                                    # 发生降级的节点
    reason: str                                  # 降级原因
    level: int = Field(..., ge=1, le=5)          # 降级级别（1-5）
    action: str                                  # 采取的降级动作
    timestamp: datetime = Field(default_factory=datetime.now)  # 记录时间


# 幻觉检测记录模型
class HallucinationRecord(BaseModel):
    node: str                                    # 发生幻觉的节点
    assertion: str                               # 幻觉断言内容
    matched: bool = False                        # 是否与原文匹配
    detected_at: datetime = Field(default_factory=datetime.now)  # 检测时间