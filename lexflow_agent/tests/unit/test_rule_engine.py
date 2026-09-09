"""单元测试: 规则过滤引擎"""

import pytest
from pydantic import ValidationError

from lexflow_agent.engine.models.case import CaseInput, CaseType, ClientRole, CaseDocument, PageContent
from lexflow_agent.engine.precheck.rule_engine import (
    CaseTypeRule, MaterialNonEmptyRule, GoalNonEmptyRule,
    GreetingRule, VolumeLimitRule, RuleEngine,
)


def _make_case(goal="请求赔偿", doc_text="test", case_type="labor_dispute"):
    doc = CaseDocument(
        document_id="doc_001", file_name="t.txt",
        pages=[PageContent(page=1, text=doc_text)],
    )
    try:
        ct = CaseType(case_type)
    except ValueError:
        ct = CaseType.LABOR_DISPUTE
    return CaseInput(
        case_id="c1", case_type=ct,
        client_role=ClientRole.EMPLOYEE, client_goal=goal,
        documents=[doc],
    )


class TestCaseTypeRule:
    def test_supported(self):
        rule = CaseTypeRule()
        result = rule.evaluate(_make_case())
        assert result.passed

    def test_unsupported(self):
        rule = CaseTypeRule()
        doc = CaseDocument(document_id="doc_001", file_name="t.txt",
            pages=[PageContent(page=1, text="test")])
        ci = CaseInput(case_id="c1", case_type=CaseType.LABOR_DISPUTE,
            client_role=ClientRole.EMPLOYEE, client_goal="请求", documents=[doc])
        # 手动设一个不支持的 case_type 字符串
        object.__setattr__(ci, 'case_type', 'criminal')
        result = rule.evaluate(ci)
        assert not result.passed
        assert result.code == "UNSUPPORTED_CASE_TYPE"


class TestGoalNonEmptyRule:
    def test_empty_goal_fails(self):
        rule = GoalNonEmptyRule()
        doc = CaseDocument(document_id="doc_001", file_name="t.txt",
            pages=[PageContent(page=1, text="test")])
        ci = CaseInput(case_id="c1", case_type=CaseType.LABOR_DISPUTE,
            client_role=ClientRole.EMPLOYEE, client_goal="x", documents=[doc])
        # 手动设空字符串以绕过 min_length
        ci.client_goal = ""
        result = rule.evaluate(ci)
        assert not result.passed

    def test_non_empty_passes(self):
        rule = GoalNonEmptyRule()
        result = rule.evaluate(_make_case(goal="请求赔偿"))
        assert result.passed


class TestGreetingRule:
    def test_greeting_blocked(self):
        rule = GreetingRule()
        for g in ["你好", "hello", "测试"]:
            result = rule.evaluate(_make_case(goal=g))
            assert not result.passed, f"'{g}' should be blocked"

    def test_normal_goal_passes(self):
        rule = GreetingRule()
        result = rule.evaluate(_make_case(goal="请求支付拖欠工资"))
        assert result.passed


class TestRuleEngine:
    def test_valid_passes(self):
        result = RuleEngine().check(_make_case())
        assert result is None

    def test_invalid_filters(self):
        doc = CaseDocument(document_id="doc_001", file_name="t.txt",
            pages=[PageContent(page=1, text="test")])
        ci = CaseInput(case_id="c1", case_type=CaseType.LABOR_DISPUTE,
            client_role=ClientRole.EMPLOYEE, client_goal="x", documents=[doc])
        ci.client_goal = ""
        result = RuleEngine().check(ci)
        assert result is not None
        assert result.code == "EMPTY_GOAL"

    def test_greeting_blocked(self):
        result = RuleEngine().check(_make_case(goal="你好"))
        assert result is not None
        assert result.code == "GREETING_NOT_SUPPORTED"
