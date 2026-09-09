#!/usr/bin/env python3
"""Run the deterministic contract gate as a LangSmith offline experiment."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langsmith import Client

from lexflow_agent.config.settings import settings
from lexflow_agent.engine.models.case import CaseInput
from lexflow_agent.engine.nodes.validation import validate_input_node
from lexflow_agent.engine.state import CaseAgentState


def target(inputs: dict) -> dict:
    case = CaseInput.model_validate(inputs)
    result = validate_input_node(CaseAgentState(
        run_id=f"eval-{case.case_id}",
        case_id=case.case_id,
        case_input=case,
    ))
    return {
        "valid": not result.get("errors"),
        "phase": result.get("phase"),
        "case_id": case.case_id,
    }


def contract_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    annotations = (reference_outputs or {}).get("annotations", {})
    return {
        "key": "contract_and_reference_completeness",
        "score": int(bool(outputs.get("valid") and annotations.get("facts"))),
    }


def main() -> None:
    client = Client(api_key=settings.LANGSMITH_API_KEY, api_url=settings.LANGSMITH_ENDPOINT)
    results = client.evaluate(
        target,
        data="lexflow-labor-dispute-mvp",
        evaluators=[contract_evaluator],
        experiment_prefix=f"lexflow-{settings.APP_VERSION}",
        description="30-case contract and annotation completeness gate",
        metadata={"deployment": "vm-192.168.88.100", "agent_count": 4},
        max_concurrency=4,
    )
    print(results)


if __name__ == "__main__":
    main()
