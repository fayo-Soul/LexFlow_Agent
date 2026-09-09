# 模块文档字符串：统一响应模型
"""统一响应模型"""

# 启用 Python 3.10+ 的新特性支持
from __future__ import annotations

# 导入类型提示工具
from typing import Any, Optional

# 导入 Pydantic 基础模型和字段装饰器
from pydantic import BaseModel, Field

# 导入 API 状态枚举
from .enums import ApiStatus
# 导入降级记录模型
from .review import DegradationRecord


# API 错误模型
class ApiError(BaseModel):
    code: str                                    # 错误代码
    message: str                                 # 错误消息
    details: Optional[Any] = None                # 错误详情


# API 统一响应模型
class ApiResponse(BaseModel):
    status: ApiStatus = ApiStatus.SUCCESS        # 响应状态
    data: Optional[dict[str, Any]] = None        # 响应数据
    error: Optional[ApiError] = None             # 错误信息
    warnings: list[str] = Field(default_factory=list)  # 警告列表
    degradations: list[DegradationRecord] = Field(default_factory=list)  # 降级记录列表
    trace_id: str = ""                           # 追踪 ID


# 节点追踪模型
class NodeTrace(BaseModel):
    node: str                                    # 节点名称
    status: str  # success | degraded | failed | skipped  # 节点状态
    start_time: str = ""                         # 开始时间
    end_time: str = ""                           # 结束时间
    duration_ms: float = 0.0                     # 耗时（毫秒）
    prompt_version: str = ""                     # Prompt 版本
    model: str = ""                              # 使用的模型
    token_usage: int = 0                         # Token 使用量
    retry_count: int = 0                         # 重试次数
    error: str = ""                              # 错误信息