from lexflow_agent.engine.models.enums import RiskLevel
from lexflow_agent.engine.nodes.strategy import _fallback_strategy
from lexflow_agent.engine.state import CaseAgentState


def test_strategy_fallback_is_explicit_and_non_authoritative():
    strategy = _fallback_strategy(CaseAgentState(run_id="r", case_id="c"))

    assert strategy.primary_plan.risk_level == RiskLevel.INSUFFICIENT
    assert strategy.primary_plan.legal_basis == []
    assert strategy.pending_confirmations
    assert strategy.risks[0]["type"] == "model_degradation"
