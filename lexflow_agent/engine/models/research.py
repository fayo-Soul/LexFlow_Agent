# 模块文档字符串：法律研究模型
"""法律研究模型"""

# 启用 Python 3.10+ 的新特性支持
from __future__ import annotations

# 导入 Pydantic 基础模型和字段装饰器
from pydantic import BaseModel, Field


# 法条引用模型
class Citation(BaseModel):
    citation_id: str                             # 引用 ID
    document_id: str                             # 文档 ID
    chunk_id: str = ""                           # 文本块 ID
    title: str                                   # 文档标题
    location: str = ""                           # 位置信息
    quote: str = ""                              # 引用原文


# 法律研究结果模型
class ResearchResult(BaseModel):
    issue_id: str                                # 关联的争议焦点 ID
    conclusion: str                              # 研究结论
    application_conditions: list[str] = Field(default_factory=list)  # 适用条件列表
    favorable_sources: list[Citation] = Field(default_factory=list)  # 有利引用列表
    unfavorable_sources: list[Citation] = Field(default_factory=list)  # 不利引用列表
    neutral_sources: list[Citation] = Field(default_factory=list)  # 中性引用列表
    uncertainties: list[str] = Field(default_factory=list)  # 不确定性列表
    citations_valid: bool = False                # 引用是否有效