import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from lexflow_agent.api.routes.case_runs import (
    CreateRunRequest,
    _apply_graph_result,
    _idempotency,
    _runs,
    delete_run,
)
from lexflow_agent.engine.models.enums import WorkflowStatus
from lexflow_agent.engine.state import CaseAgentState
from lexflow_agent.api.app import app
from lexflow_agent.config.settings import settings
from lexflow_agent.engine.run_store import run_store


def test_graph_interrupt_sets_fact_review_status():
    run = CaseAgentState(run_id="r1", case_id="c1")
    _apply_graph_result(run, {"current_node": "extract_facts"}, ("identify_issues",))
    assert run.status == WorkflowStatus.WAITING_FACT_REVIEW
    assert run.completed_at is None


def test_graph_interrupt_sets_strategy_review_status():
    run = CaseAgentState(run_id="r2", case_id="c2")
    _apply_graph_result(run, {"current_node": "generate_strategy"}, ("draft_documents",))
    assert run.status == WorkflowStatus.WAITING_STRATEGY_REVIEW


def test_four_agent_interrupt_names_set_review_statuses():
    fact_run = CaseAgentState(run_id="r-agent-1", case_id="c-agent-1")
    strategy_run = CaseAgentState(run_id="r-agent-2", case_id="c-agent-2")

    _apply_graph_result(fact_run, {}, ("legal_research_agent",))
    _apply_graph_result(strategy_run, {}, ("document_generation_agent",))

    assert fact_run.status == WorkflowStatus.WAITING_FACT_REVIEW
    assert strategy_run.status == WorkflowStatus.WAITING_STRATEGY_REVIEW


def test_graph_without_next_node_completes():
    run = CaseAgentState(run_id="r3", case_id="c3")
    _apply_graph_result(run, {"current_node": "semantic_review"}, ())
    assert run.status == WorkflowStatus.COMPLETED
    assert run.completed_at is not None


async def test_delete_run_removes_idempotency_mapping():
    run = CaseAgentState(run_id="r4", case_id="c4", idempotency_key="key-4")
    _runs[run.run_id] = run
    _idempotency["key-4"] = run.run_id

    await delete_run(run.run_id)

    assert run.run_id not in _runs
    assert "key-4" not in _idempotency


def test_create_request_rejects_invalid_document_id():
    with pytest.raises(ValidationError):
        CreateRunRequest(
            case_id="c5",
            client_goal="请求赔偿",
            documents=[{
                "document_id": "doc1",
                "file_name": "test.txt",
                "pages": [{"page": 1, "text": "测试"}],
            }],
        )


def test_case_routes_require_bearer_token():
    client = TestClient(app)
    assert client.get("/api/v1/case-runs/not-found").status_code == 401
    response = client.get(
        "/api/v1/case-runs/not-found",
        headers={"Authorization": f"Bearer {settings.API_TOKEN}"},
    )
    assert response.status_code == 404


def test_run_store_round_trip():
    run = CaseAgentState(run_id="persist-test", case_id="persist-case")
    run_store.save(run)
    try:
        restored = run_store.load_all()["persist-test"]
        assert restored.case_id == "persist-case"
    finally:
        run_store.delete("persist-test")
