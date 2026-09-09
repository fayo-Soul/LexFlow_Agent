# 模块文档字符串：枚举定义
"""枚举定义"""

# 启用 Python 3.10+ 的新特性支持
from __future__ import annotations

# 导入枚举基类
from enum import Enum


# 工作流状态枚举
class WorkflowStatus(str, Enum):
    PENDING = "pending"                          # 待处理
    RUNNING = "running"                          # 运行中
    WAITING_FACT_REVIEW = "waiting_fact_review"  # 等待事实复核
    WAITING_STRATEGY_REVIEW = "waiting_strategy_review"  # 等待策略复核
    WAITING_CLARIFICATION = "waiting_clarification"  # 等待澄清（单Agent缺少数据时追问）
    WAITING_HUMAN_INTERVENTION = "waiting_human_intervention"  # 等待人工干预
    COMPLETED = "completed"                      # 已完成
    FAILED = "failed"                            # 失败
    CANCELLED = "cancelled"                      # 已取消


# 案件类型枚举
class CaseType(str, Enum):
    LABOR_DISPUTE = "labor_dispute"              # 劳动争议


# 客户角色枚举
class ClientRole(str, Enum):
    EMPLOYEE = "employee"                        # 劳动者
    EMPLOYER = "employer"                        # 用人单位


# 事实状态枚举
class FactStatus(str, Enum):
    CONFIRMED = "confirmed"                      # 已确认
    INFERRED = "inferred"                        # 推断
    CONFLICTING = "conflicting"                  # 冲突
    INSUFFICIENT = "insufficient"                # 材料不足


# 证据强度枚举
class EvidenceStrength(str, Enum):
    FULL = "full"                                # 充分
    PARTIAL = "partial"                          # 部分
    MISSING = "missing"                          # 缺失
    CONFLICTING = "conflicting"                  # 冲突
    INSUFFICIENT = "insufficient"                # 不足


# 风险等级枚举
class RiskLevel(str, Enum):
    LOW = "low"                                  # 低风险
    MEDIUM = "medium"                            # 中风险
    HIGH = "high"                                # 高风险
    INSUFFICIENT = "insufficient"                # 不足


# 意图类型枚举
class IntentType(str, Enum):
    CASE_ANALYSIS = "case_analysis"              # 案件分析
    CASE_STATUS_QUERY = "case_status_query"      # 案件状态查询
    LEGAL_QUESTION = "legal_question"            # 法律问题
    DELETE_REQUEST = "delete_request"            # 删除请求
    CANCEL_REQUEST = "cancel_request"            # 取消请求
    UNKNOWN = "unknown"                          # 未知


# 文档类型枚举
class DocumentType(str, Enum):
    CONTRACT = "contract"                        # 合同
    PAYSLIP = "payslip"                          # 工资单
    ATTENDANCE = "attendance"                    # 考勤记录
    TERMINATION_NOTICE = "termination_notice"    # 解除通知
    CHAT_RECORD = "chat_record"                  # 聊天记录
    OTHER = "other"                              # 其他


# 降级级别枚举
class DegradationLevel(int, Enum):
    RETRY = 1                                    # 重试
    NODE_FALLBACK = 2                            # 节点降级
    MODEL_FAILOVER = 3                           # 模型切换
    CIRCUIT_BREAK = 4                            # 熔断
    SYSTEM_FALLBACK = 5                          # 系统降级


# 恢复操作枚举
class ResumeAction(str, Enum):
    APPROVE_FACTS = "approve_facts"              # 批准事实
    REJECT_FACTS = "reject_facts"                # 拒绝事实
    APPROVE_STRATEGY = "approve_strategy"        # 批准策略
    REJECT_STRATEGY = "reject_strategy"          # 拒绝策略
    PROVIDE_INFO = "provide_additional_information"  # 提供补充信息
    SUBMIT_CLARIFICATION = "submit_clarification"  # 提交澄清回答
    CANCEL = "cancel"                            # 取消


# API 状态枚举
class ApiStatus(str, Enum):
    SUCCESS = "success"                          # 成功
    ERROR = "error"                              # 错误
    DEGRADED = "degraded"                        # 降级