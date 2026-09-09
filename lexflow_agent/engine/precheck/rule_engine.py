# 模块文档字符串：规则过滤引擎 - 责任链模式
"""规则过滤引擎 - 责任链模式"""

# 启用 Python 3.10+ 的新特性支持
from __future__ import annotations

# 导入正则表达式模块
import re
# 导入抽象基类
from abc import ABC, abstractmethod
# 导入可选类型提示
from typing import Optional

# 导入案件输入模型
from lexflow_agent.engine.models.case import CaseInput
# 导入意图类型枚举
from lexflow_agent.engine.models.enums import IntentType


# 规则结果类
class RuleResult:
    def __init__(self, passed: bool = True, code: str = "",
                 message: str = "", log_audit: bool = False):
        self.passed = passed                     # 是否通过
        self.code = code                         # 错误代码
        self.message = message                   # 错误消息
        self.log_audit = log_audit               # 是否需要审计日志


# 规则抽象基类
class Rule(ABC):
    @abstractmethod
    def evaluate(self, request: CaseInput) -> RuleResult:
        pass


# 案由校验规则
class CaseTypeRule(Rule):
    SUPPORTED = {"labor_dispute"}                # 支持的案由集合

    def evaluate(self, request: CaseInput) -> RuleResult:
        # 获取案由值
        case_type = request.case_type.value if hasattr(request.case_type, 'value') else str(request.case_type)
        # 检查案由是否在支持列表中
        if case_type not in self.SUPPORTED:
            return RuleResult(
                passed=False, code="UNSUPPORTED_CASE_TYPE",
                message=f"案由 {case_type} 不在首期支持列表中",
            )
        return RuleResult()


# 材料非空校验规则
class MaterialNonEmptyRule(Rule):
    def evaluate(self, request: CaseInput) -> RuleResult:
        # 检查文档列表是否为空
        if not request.documents:
            return RuleResult(
                passed=False, code="EMPTY_DOCUMENTS",
                message="案件材料不能为空",
            )
        return RuleResult()


# 诉求非空校验规则
class GoalNonEmptyRule(Rule):
    def evaluate(self, request: CaseInput) -> RuleResult:
        # 获取并清理客户诉求
        goal = (request.client_goal or "").strip()
        # 检查诉求是否为空
        if not goal:
            return RuleResult(
                passed=False, code="EMPTY_GOAL",
                message="客户诉求不能为空",
            )
        return RuleResult()


# 问候语过滤规则
class GreetingRule(Rule):
    PATTERNS = {
        "你好", "您好", "hello", "hi", "hey",
        "测试", "test", "ping", "help", "帮助",
        "你是谁", "你能做什么",
    }

    def evaluate(self, request: CaseInput) -> RuleResult:
        # 将诉求转为小写
        goal = request.client_goal.strip().lower()
        # 检查是否为问候语
        if goal in self.PATTERNS:
            return RuleResult(
                passed=False, code="GREETING_NOT_SUPPORTED",
                message="您好，本系统用于劳动争议案件分析。如需使用，请输入案件具体信息（案由、诉求、材料）。",
            )
        return RuleResult()


# 体积限制规则
class VolumeLimitRule(Rule):
    MAX_CHARS = 500_000                          # 最大字符数
    MAX_DOCS = 20                                # 最大文档数

    def evaluate(self, request: CaseInput) -> RuleResult:
        # 计算总字符数
        total_chars = sum(
            len(p.text) for doc in request.documents for p in doc.pages
        )
        # 检查字符数是否超限
        if total_chars > self.MAX_CHARS:
            return RuleResult(
                passed=False, code="VOLUME_EXCEEDED",
                message=f"材料总字符数 {total_chars} 超过限制 {self.MAX_CHARS}",
            )
        # 检查文档数是否超限
        if len(request.documents) > self.MAX_DOCS:
            return RuleResult(
                passed=False, code="TOO_MANY_DOCUMENTS",
                message=f"文档数 {len(request.documents)} 超过限制 {self.MAX_DOCS}",
            )
        return RuleResult()


# 恶意参数检测规则
class MaliciousParamRule(Rule):
    INJECTION = [
        r"<script.*?>.*?</script>",              # XSS 攻击
        r"'.*?(OR|or|AND|and).*?'.*?=",          # SQL 注入
        r"__(proto|class|define)__",             # 原型链污染
        r"union.*select",                        # SQL UNION 注入
        r"drop\s+table",                         # SQL DROP 注入
    ]

    def evaluate(self, request: CaseInput) -> RuleResult:
        # 拼接需要检查的文本
        text_to_check = request.client_goal + " " + " ".join(
            doc.file_name for doc in request.documents
        )
        # 检查是否包含恶意模式
        for pattern in self.INJECTION:
            if re.search(pattern, text_to_check, re.IGNORECASE):
                return RuleResult(
                    passed=False, code="MALICIOUS_INPUT",
                    message="请求参数包含非法内容", log_audit=True,
                )
        return RuleResult()


# 规则引擎类
class RuleEngine:
    """规则过滤引擎"""

    def __init__(self):
        # 初始化规则链
        self._rules: list[Rule] = [
            CaseTypeRule(),                      # 案由校验
            MaterialNonEmptyRule(),              # 材料非空
            GoalNonEmptyRule(),                  # 诉求非空
            GreetingRule(),                      # 问候语过滤
            VolumeLimitRule(),                   # 体积限制
            MaliciousParamRule(),                # 恶意参数检测
        ]

    def check(self, request: CaseInput) -> Optional[RuleResult]:
        """按顺序执行所有规则，返回首个拦截结果或 None"""
        for rule in self._rules:
            result = rule.evaluate(request)
            if not result.passed:
                return result                    # 返回首个失败的规则结果
        return None                              # 所有规则通过


# 创建全局规则引擎实例
rule_engine = RuleEngine()