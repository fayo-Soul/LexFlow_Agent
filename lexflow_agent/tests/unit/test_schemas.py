"""单元测试: Schema 校验"""

import pytest
from pydantic import ValidationError

from lexflow_agent.engine.models.case import CaseInput, CaseType, ClientRole, CaseDocument, PageContent
from lexflow_agent.engine.models.facts import Fact, FactStatus, FactSource
from lexflow_agent.engine.models.evidence import EvidenceItem, EvidenceSource, EvidenceStrength
from lexflow_agent.engine.models.enums import WorkflowStatus


class TestCaseInput:
    def test_valid_input(self):
        doc = CaseDocument(
            document_id="doc_001", file_name="test.pdf",
            pages=[PageContent(page=1, text="test")],
        )
        case = CaseInput(
            case_id="case_001", case_type=CaseType.LABOR_DISPUTE,
            client_role=ClientRole.EMPLOYEE, client_goal="请求赔偿",
            documents=[doc],
        )
        assert case.case_id == "case_001"

    def test_empty_goal_fails(self):
        with pytest.raises(ValidationError):
            CaseInput(
                case_id="case_001", case_type=CaseType.LABOR_DISPUTE,
                client_role=ClientRole.EMPLOYEE, client_goal="",
                documents=[],
            )

    def test_empty_documents_fails(self):
        with pytest.raises(ValidationError):
            CaseInput(
                case_id="case_001", case_type=CaseType.LABOR_DISPUTE,
                client_role=ClientRole.EMPLOYEE, client_goal="请求",
                documents=[],
            )

    def test_negative_page_fails(self):
        with pytest.raises(ValidationError):
            PageContent(page=0, text="test")


class TestFactModel:
    def test_valid_fact(self):
        fact = Fact(
            fact_id="fact_001", statement="公司解除合同",
            status=FactStatus.CONFIRMED, confidence=0.95,
            sources=[FactSource(document_id="doc_001", page=1, quote="解除合同")],
        )
        assert fact.status == FactStatus.CONFIRMED

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            Fact(fact_id="f1", statement="test", confidence=1.5)

    def test_empty_sources_allowed_for_inferred(self):
        fact = Fact(
            fact_id="f1", statement="推断",
            status=FactStatus.INFERRED, confidence=0.5,
        )
        assert len(fact.sources) == 0


class TestEvidenceModel:
    def test_valid_evidence(self):
        item = EvidenceItem(
            issue_id="issue_001", claim="赔偿金",
            fact_to_prove="公司违法解除",
            status=EvidenceStrength.PARTIAL,
        )
        assert item.status == EvidenceStrength.PARTIAL


class TestWorkflowStatus:
    def test_status_values(self):
        assert WorkflowStatus.PENDING.value == "pending"
        assert WorkflowStatus.RUNNING.value == "running"
        assert WorkflowStatus.COMPLETED.value == "completed"
        assert WorkflowStatus.FAILED.value == "failed"
