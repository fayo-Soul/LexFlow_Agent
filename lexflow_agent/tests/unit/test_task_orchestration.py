from fastapi.testclient import TestClient

from lexflow_agent.api.app import app
from lexflow_agent.config.settings import settings
from lexflow_agent.engine.orchestrator.models import AgentName, TaskMode, TaskRequest
from lexflow_agent.engine.orchestrator.service import Orchestrator
from lexflow_agent.engine.routing.router import route_request


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.API_TOKEN}"}


def test_auto_legal_question_routes_only_to_legal_research():
    request = TaskRequest(
        case_id="route-legal",
        message="违法解除劳动合同的赔偿标准是什么？",
    )

    decision = route_request(request, use_llm=False)

    assert decision.candidate_agents == [AgentName.LEGAL_RESEARCH]


def test_single_agent_mode_does_not_expand_dependencies():
    request = TaskRequest(
        case_id="single-strategy",
        message="制定案件策略",
        mode=TaskMode.SINGLE_AGENT,
        requested_agent=AgentName.CASE_STRATEGY,
        allow_orchestration=False,
    )

    result = Orchestrator().execute(request)

    assert result.status == "input_required"
    assert result.executed_agents == []
    assert "facts" in result.missing_inputs
    assert "legal_research" in result.missing_inputs


def test_full_case_stops_for_fact_review_before_legal_research():
    class FakeRunner:
        def create_state(self, request):
            from lexflow_agent.engine.models.facts import Fact
            from lexflow_agent.engine.state import CaseAgentState

            return CaseAgentState(case_id=request.case_id, facts=[])

        def run(self, agent, state):
            from lexflow_agent.engine.models.facts import Fact

            assert agent == AgentName.CASE_UNDERSTANDING
            state.facts = [
                Fact(
                    fact_id="fact_001",
                    statement="存在劳动关系",
                    status="confirmed",
                )
            ]
            return state

    request = TaskRequest(
        case_id="full-review",
        message="分析完整案件",
        mode=TaskMode.FULL_CASE,
        documents=[{
            "document_id": "doc_case",
            "file_name": "case.txt",
            "pages": [{"page": 1, "text": "案件材料"}],
        }],
    )

    result = Orchestrator(runner=FakeRunner()).execute(request)

    assert result.status == "waiting_fact_review"
    assert result.executed_agents == [AgentName.CASE_UNDERSTANDING]


def test_single_legal_research_api_can_be_called_without_documents(monkeypatch):
    def fake_execute(self, request):
        assert request.mode == TaskMode.SINGLE_AGENT
        assert request.requested_agent == AgentName.LEGAL_RESEARCH
        return self.result_model(
            status="completed",
            route_source="explicit",
            executed_agents=[AgentName.LEGAL_RESEARCH],
            output={"legal_research": []},
        )

    monkeypatch.setattr(Orchestrator, "execute", fake_execute)
    client = TestClient(app)
    response = client.post(
        "/api/v1/agents/legal-research/runs",
        headers=_headers(),
        json={"case_id": "legal-only", "message": "经济补偿如何计算？"},
    )

    assert response.status_code == 200
    assert response.json()["data"]["executed_agents"] == ["legal_research"]


def test_unified_task_endpoint_requires_authentication():
    client = TestClient(app)
    response = client.post(
        "/api/v1/tasks",
        json={"case_id": "no-auth", "message": "分析案件"},
    )

    assert response.status_code == 401
