from app.query_planner import QueryPlan, QueryPlanner


class FakeClassifier:
    def __init__(self, domain=None, confidence=0.0):
        self.domain = domain
        self.confidence = confidence

    def predict_with_confidence(self, query):
        return self.domain, self.confidence


def test_high_certainty_rule_returns_fixed_response_without_models():
    planner = QueryPlanner(
        classifier=FakeClassifier("劳动法", 0.99),
        llm=lambda prompt: (_ for _ in ()).throw(AssertionError("不应调用 LLM")),
    )

    plan = planner.plan("你好")

    assert plan.intent == "greeting"
    assert plan.scope == ()


def test_case_request_is_routed_to_case_by_rule():
    planner = QueryPlanner(classifier=FakeClassifier(), llm=None)

    plan = planner.plan("请查找竞业限制相关案例和裁判观点")

    assert plan.intent == "legal_research"
    assert plan.scope == ("case",)
    assert plan.task_type == "similar_case_search"
    assert plan.route_source == "rule"


def test_high_confidence_bert_avoids_llm_and_uses_public_legal_sources():
    planner = QueryPlanner(
        classifier=FakeClassifier("劳动法", 0.91),
        llm=lambda prompt: (_ for _ in ()).throw(AssertionError("不应调用 LLM")),
    )

    plan = planner.plan("公司拖欠工资应当如何处理")

    assert plan.legal_domains == ("劳动法",)
    assert plan.scope == ("regulation", "judicial_interpretation")
    assert plan.route_source == "bert"


def test_low_confidence_bert_uses_structured_llm_route():
    planner = QueryPlanner(
        classifier=FakeClassifier(None, 0.42),
        llm=lambda prompt: """
        ```json
        {
          "intent": "legal_research",
          "legal_domains": ["民法"],
          "requested_sources": ["regulation", "case"],
          "task_type": "legal_analysis",
          "confidence": 0.86,
          "needs_clarification": false
        }
        ```
        """,
    )

    plan = planner.plan("这种安排是否有效，通常应当如何判断")

    assert plan.scope == ("regulation", "case")
    assert plan.legal_domains == ("民法",)
    assert plan.route_source == "llm"


def test_invalid_llm_output_falls_back_to_safe_public_scope():
    planner = QueryPlanner(
        classifier=FakeClassifier(None, 0.2),
        llm=lambda prompt: "无法输出结构化结果",
    )

    plan = planner.plan("帮我看看这个问题")

    assert plan == QueryPlan(
        intent="legal_research",
        scope=("regulation", "judicial_interpretation"),
        confidence=0.2,
        route_source="fallback",
    )


def test_explicit_scope_has_priority_over_automatic_scope():
    planner = QueryPlanner(
        classifier=FakeClassifier("劳动法", 0.95),
        llm=None,
    )

    plan = planner.plan(
        "请查找竞业限制案例",
        explicit_scope=["regulation"],
    )

    assert plan.scope == ("regulation",)
    assert plan.route_source == "explicit"
