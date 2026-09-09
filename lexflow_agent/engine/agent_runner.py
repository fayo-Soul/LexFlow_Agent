from __future__ import annotations

from lexflow_agent.engine.agents import (
    build_case_strategy_agent,
    build_case_understanding_agent,
    build_document_generation_agent,
    build_legal_research_agent,
)
from lexflow_agent.engine.models.case import CaseInput
from lexflow_agent.engine.models.evidence import EvidenceItem
from lexflow_agent.engine.models.facts import Fact
from lexflow_agent.engine.models.issues import Issue
from lexflow_agent.engine.models.research import ResearchResult
from lexflow_agent.engine.models.strategy import StrategyResult
from lexflow_agent.engine.orchestrator.models import AgentName, TaskRequest
from lexflow_agent.engine.state import CaseAgentState


_BUILDERS = {
    AgentName.CASE_UNDERSTANDING: build_case_understanding_agent,
    AgentName.LEGAL_RESEARCH: build_legal_research_agent,
    AgentName.CASE_STRATEGY: build_case_strategy_agent,
    AgentName.DOCUMENT_GENERATION: build_document_generation_agent,
}


class AgentRunner:
    def create_state(self, request: TaskRequest) -> CaseAgentState:
        context = request.context
        case_data = {
            "case_id": request.case_id,
            "case_type": request.case_type,
            "client_role": request.client_role,
            "client_goal": request.message,
            "documents": request.documents,
            "idempotency_key": request.idempotency_key,
        }
        # The legacy full-case contract requires documents. A legal-research-only
        # task intentionally has none, so the adapter constructs the internal
        # carrier without weakening the public full-case schema.
        case_input = (
            CaseInput.model_validate(case_data)
            if request.documents
            else CaseInput.model_construct(**case_data)
        )
        return CaseAgentState(
            case_id=request.case_id,
            case_input=case_input,
            facts=[Fact.model_validate(item) for item in context.get("facts", [])],
            issues=[Issue.model_validate(item) for item in context.get("issues", [])],
            legal_research=[
                ResearchResult.model_validate(item)
                for item in context.get("legal_research", [])
            ],
            evidence_matrix=[
                EvidenceItem.model_validate(item)
                for item in context.get("evidence_matrix", [])
            ],
            strategy=(
                StrategyResult.model_validate(context["strategy"])
                if context.get("strategy")
                else None
            ),
        )

    def run(self, agent: AgentName, state: CaseAgentState) -> CaseAgentState:
        result = _BUILDERS[agent]().invoke(state)
        return CaseAgentState.model_validate(result)
