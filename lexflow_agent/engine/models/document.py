# 模块文档字符串：文书模型
"""文书模型"""

# 启用 Python 3.10+ 的新特性支持
from __future__ import annotations

# 导入 Pydantic 基础模型和字段装饰器
from pydantic import BaseModel, Field


# 文书段落模型
class DraftSection(BaseModel):
    title: str                                   # 段落标题
    content: str                                 # 段落内容
    fact_mappings: list[str] = Field(default_factory=list)  # 关联事实列表
    citation_mappings: list[str] = Field(default_factory=list)  # 关联法条引用列表


# 文书起草结果模型
class DraftResult(BaseModel):
    document_type: str  # arbitration_application | defense_statement | evidence_list  # 文书类型
    sections: list[DraftSection] = Field(default_factory=list)  # 文书段落列表
    full_text: str = ""                          # 完整文书文本
    placeholders: list[str] = Field(default_factory=list)  # 未替换的占位符
    reviewed: bool = False                       # 是否已复核