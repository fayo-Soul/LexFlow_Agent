"""集成测试 - 案件分析全链路"""

import pytest
from pydantic import ValidationError

from lexflow_agent.engine.models.case import CaseInput, CaseType, ClientRole, CaseDocument, PageContent
from lexflow_agent.engine.models.enums import WorkflowStatus
from lexflow_agent.engine.state import CaseAgentState
from lexflow_agent.engine.nodes.validation import validate_input_node
from lexflow_agent.engine.nodes.deterministic_review import deterministic_review_node
from lexflow_agent.engine.cache.semantic_cache import semantic_cache
from lexflow_agent.engine.precheck.rule_engine import RuleEngine
from lexflow_agent.engine.precheck.intent_classifier import IntentClassifier


@pytest.fixture
def sample_case_input():
    docs = [
        CaseDocument(
            document_id="doc_001", file_name="劳动合同.pdf",
            pages=[PageContent(page=1, text="甲方：公司A，乙方：员工B，劳动合同期限2020-01-01至2025-12-31")],
        ),
        CaseDocument(
            document_id="doc_002", file_name="解除通知书.pdf",
            pages=[PageContent(page=1, text="经公司研究决定，自2025-03-10起解除与员工B的劳动合同")],
        ),
    ]
    return CaseInput(
        case_id="test_integration_001", case_type=CaseType.LABOR_DISPUTE,
        client_role=ClientRole.EMPLOYEE, client_goal="请求支付违法解除赔偿金",
        documents=docs,
    )


class TestWorkflowIntegration:
    def test_validation_passes(self, sample_case_input):
        state = CaseAgentState(
            run_id="run_test_001", case_id="test_001",
            case_input=sample_case_input, status=WorkflowStatus.PENDING,
        )
        result = validate_input_node(state)
        assert "errors" not in result or not result["errors"]
        assert result.get("status") in (WorkflowStatus.RUNNING, "running")

    def test_validation_empty_goal_fails(self):
        doc = CaseDocument(document_id="doc_001", file_name="t.pdf",
            pages=[PageContent(page=1, text="test")])
        with pytest.raises(ValidationError):
            CaseInput(case_id="test_002", case_type=CaseType.LABOR_DISPUTE,
                client_role=ClientRole.EMPLOYEE, client_goal="", documents=[doc])

    def test_rule_engine_passes(self, sample_case_input):
        result = RuleEngine().check(sample_case_input)
        assert result is None

    def test_rule_engine_blocks_greeting(self, sample_case_input):
        greeting = CaseInput(
            case_id="test_003", case_type=CaseType.LABOR_DISPUTE,
            client_role=ClientRole.EMPLOYEE, client_goal="你好", documents=sample_case_input.documents,
        )
        result = RuleEngine().check(greeting)
        assert result is not None
        assert result.code == "GREETING_NOT_SUPPORTED"

    def test_intent_classifier(self, sample_case_input):
        result = IntentClassifier().classify(sample_case_input)
        assert result.intent.value == "case_analysis"
        assert result.confidence >= 0.6

    def test_intent_classifier_status_query(self, sample_case_input):
        status_req = CaseInput(
            case_id="test_004", case_type=CaseType.LABOR_DISPUTE,
            client_role=ClientRole.EMPLOYEE, client_goal="查进度", documents=sample_case_input.documents,
        )
        result = IntentClassifier().classify(status_req)
        assert result.intent.value == "case_status_query"

    def test_deterministic_review_empty_state(self):
        state = CaseAgentState(run_id="run_005", case_id="test_005")
        result = deterministic_review_node(state)
        review = result.get("deterministic_review")
        assert review is not None
        assert review.passed

    def test_semantic_cache(self):
        semantic_cache.clear()
        semantic_cache.set("劳动法第几条", {"answer": "第X条"}, scope="test")
        cached = semantic_cache.get("劳动法第几条", scope="test")
        assert cached is not None
        assert cached["answer"] == "第X条"
        missed = semantic_cache.get("不存在的查询", scope="test")
        assert missed is None


class TestStateMigration:
    def test_state_enum_values(self):
        assert WorkflowStatus.PENDING.value == "pending"
        assert WorkflowStatus.RUNNING.value == "running"
        assert WorkflowStatus.COMPLETED.value == "completed"
        assert WorkflowStatus.FAILED.value == "failed"
        assert WorkflowStatus.CANCELLED.value == "cancelled"

    def test_state_initial(self):
        state = CaseAgentState(run_id="r1", case_id="c1")
        assert state.status == WorkflowStatus.PENDING
        assert len(state.trace) == 0
        assert len(state.degradations) == 0
