# 模块文档字符串：核心输入模型 - 案件输入
"""核心输入模型 - 案件输入"""

# 启用 Python 3.10+ 的新特性支持
from __future__ import annotations

# 导入日期时间模块
from datetime import datetime
# 导入可选类型提示
from typing import Optional

# 导入 Pydantic 基础模型和字段装饰器
from pydantic import BaseModel, Field

# 导入案件类型和客户角色枚举
from .enums import CaseType, ClientRole


# 页面内容模型
class PageContent(BaseModel):
    page: int = Field(..., ge=1, description="页码")  # 页码（从 1 开始）
    text: str = Field(..., min_length=1, description="页面正文")  # 页面正文内容


# 案件文档模型
class CaseDocument(BaseModel):
    document_id: str = Field(..., pattern=r"^doc_\w+$")  # 文档 ID（格式：doc_xxx）
    file_name: str                               # 文件名
    pages: list[PageContent] = Field(..., min_length=1)  # 页面内容列表


# 案件输入模型
class CaseInput(BaseModel):
    """案件分析请求的输入模型"""
    case_id: str                                 # 案件 ID
    case_type: CaseType                          # 案件类型
    client_role: ClientRole                      # 客户角色
    client_goal: str = Field(..., min_length=1)  # 客户诉求
    documents: list[CaseDocument] = Field(..., min_length=1)  # 案件文档列表
    idempotency_key: Optional[str] = None        # 幂等键（防止重复请求）
    created_at: datetime = Field(default_factory=datetime.now)  # 创建时间
