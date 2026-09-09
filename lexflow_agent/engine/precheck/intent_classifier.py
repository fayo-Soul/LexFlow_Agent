# 模块文档字符串：意图分类器 - 规则 + 轻量模型两层
"""意图分类器 - 规则 + 轻量模型两层"""

# 启用 Python 3.10+ 的新特性支持
from __future__ import annotations

# 导入可选类型提示
from typing import Optional

# 导入案件输入模型
from lexflow_agent.engine.models.case import CaseInput
# 导入意图类型枚举
from lexflow_agent.engine.models.enums import IntentType


# 意图结果类
class IntentResult:
    def __init__(self, intent: IntentType, confidence: float):
        self.intent = intent                     # 意图类型
        self.confidence = confidence             # 置信度


# 意图分类器类
class IntentClassifier:
    """意图分类器"""

    CONFIDENCE_THRESHOLD = 0.6                   # 置信度阈值

    def classify(self, request: CaseInput) -> IntentResult:
        """先规则匹配，不满足阈值则返回 UNKNOWN"""
        # 获取并清理客户诉求
        goal = (request.client_goal or "").strip().lower()

        # 状态查询意图检测
        if any(kw in goal for kw in ["进度", "状态", "查进度", "进行到"]):
            return IntentResult(IntentType.CASE_STATUS_QUERY, 0.95)

        # 删除意图检测
        if any(kw in goal for kw in ["删除任务", "取消", "撤销"]):
            return IntentResult(IntentType.DELETE_REQUEST, 0.95)

        # 纯法律问题意图检测（无材料时）
        if not request.documents and "?" in goal or "吗" in goal or "什么" in goal:
            return IntentResult(IntentType.LEGAL_QUESTION, 0.85)

        # 默认：案件分析意图
        if request.documents and request.client_goal:
            return IntentResult(IntentType.CASE_ANALYSIS, 0.8)

        # 兜底：未知意图
        return IntentResult(IntentType.UNKNOWN, 0.0)


# 创建全局意图分类器实例
intent_classifier = IntentClassifier()