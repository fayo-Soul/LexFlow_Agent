# 模块文档字符串：证据矩阵模型
"""证据矩阵模型"""

# 启用 Python 3.10+ 的新特性支持
from __future__ import annotations

# 导入 Pydantic 基础模型和字段装饰器
from pydantic import BaseModel, Field

# 导入证据强度枚举
from .enums import EvidenceStrength


# 证据来源模型
class EvidenceSource(BaseModel):
    evidence_id: str                             # 证据 ID
    document_id: str                             # 文档 ID
    page: int = Field(..., ge=1)                 # 页码（从 1 开始）
    evidence_name: str                           # 证据名称
    proves: list[str] = Field(default_factory=list)  # 证明内容列表
    strengths: list[str] = Field(default_factory=list)  # 优势列表
    risks: list[str] = Field(default_factory=list)  # 风险列表


# 证据项模型
class EvidenceItem(BaseModel):
    issue_id: str                                # 关联的争议焦点 ID
    claim: str                                   # 请求内容
    fact_to_prove: str                           # 需要证明的事实
    evidence: list[EvidenceSource] = Field(default_factory=list)  # 证据来源列表
    status: EvidenceStrength = EvidenceStrength.INSUFFICIENT  # 证据状态
    missing_materials: list[str] = Field(default_factory=list)  # 缺失材料列表
    risks: list[str] = Field(default_factory=list)  # 风险列表