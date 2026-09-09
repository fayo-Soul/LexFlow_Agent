#!/usr/bin/env python3
"""Verify the expected LexFlow assets and telemetry exist in LangSmith."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langsmith import Client

from lexflow_agent.config.settings import settings


def main() -> None:
    client = Client(api_key=settings.LANGSMITH_API_KEY, api_url=settings.LANGSMITH_ENDPOINT)
    prompts = [
        "lexflow-material-agent",
        "lexflow-fact-agent",
        "lexflow-issue-agent",
        "lexflow-research-agent",
        "lexflow-evidence-agent",
        "lexflow-strategy-agent",
        "lexflow-drafting-agent",
        "lexflow-semantic-review-agent",
    ]
    for prompt in prompts:
        client.pull_prompt(prompt)

    dataset = client.read_dataset(dataset_name="lexflow-labor-dispute-mvp")
    examples = list(client.list_examples(dataset_id=dataset.id))
    assert len(examples) == 30
    queue = list(client.list_annotation_queues(name="lexflow-production-review"))[0]

    runs = list(client.list_runs(project_name=settings.LANGSMITH_PROJECT, limit=100))
    run_types = {run.run_type for run in runs}
    names = {run.name for run in runs}
    metadata = [run.extra.get("metadata", {}) for run in runs]
    assert "llm" in run_types
    assert "tool" in run_types or "law_rag.search" in names
    assert any(item.get("thread_id") for item in metadata)
    assert any(item.get("deployment") == "vm-192.168.88.100" for item in metadata)

    root = next(run for run in runs if run.parent_run_id is None)
    feedback = client.create_feedback(
        root.id,
        "global_check",
        score=1,
        comment="Automated global verification passed",
        source_info={"deployment": "vm-192.168.88.100"},
    )
    client.add_runs_to_annotation_queue(queue.id, run_ids=[root.id])
    queued = list(client.list_runs_from_annotation_queue(queue.id, limit=100))
    assert any(item.id == root.id for item in queued)

    print({
        "prompts": len(prompts),
        "dataset_examples": len(examples),
        "run_types": sorted(run_types),
        "thread_metadata": True,
        "feedback_id": str(feedback.id),
        "annotation_queue": str(queue.id),
        "deployment_metadata": True,
    })


if __name__ == "__main__":
    main()
