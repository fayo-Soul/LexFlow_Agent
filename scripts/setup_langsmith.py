#!/usr/bin/env python3
"""Idempotently provision LexFlow assets in the configured LangSmith workspace."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lexflow_agent.config.settings import settings
DATASET_NAME = "lexflow-labor-dispute-mvp"
QUEUE_NAME = "lexflow-production-review"


def main() -> None:
    client = Client(api_key=settings.LANGSMITH_API_KEY, api_url=settings.LANGSMITH_ENDPOINT)

    prompt_urls = {}
    prompt_dir = ROOT / "lexflow_agent" / "engine" / "prompts"
    for path in sorted(prompt_dir.glob("*_agent.txt")):
        name = f"lexflow-{path.stem.replace('_', '-')}"
        prompt = ChatPromptTemplate.from_messages([
            ("system", path.read_text(encoding="utf-8")),
            ("human", "{input}"),
        ])
        prompt_urls[name] = client.push_prompt(
            name,
            object=prompt,
            description=f"LexFlow {path.stem} production prompt",
            tags=["lexflow", "labor-dispute", "four-agent"],
        )

    datasets = list(client.list_datasets(dataset_name=DATASET_NAME))
    dataset = datasets[0] if datasets else client.create_dataset(
        DATASET_NAME,
        description="LexFlow 劳动争议四 Agent 离线评测集",
        metadata={"application": "lexflow-agent", "version": settings.APP_VERSION},
    )
    existing = {example.metadata.get("case_id") for example in client.list_examples(dataset_id=dataset.id)}
    cases = json.loads(
        (ROOT / "data" / "eval" / "dev_set" / "sample_cases.json").read_text(encoding="utf-8")
    )
    for case in cases:
        if case["case_id"] in existing:
            continue
        client.create_example(
            dataset_id=dataset.id,
            inputs={key: value for key, value in case.items() if key != "annotations"},
            outputs={"annotations": case.get("annotations", {})},
            metadata={"case_id": case["case_id"], "domain": "labor_dispute"},
        )

    queues = list(client.list_annotation_queues(name=QUEUE_NAME))
    queue = queues[0] if queues else client.create_annotation_queue(
        name=QUEUE_NAME,
        description="低分、降级、人工介入与线上用户反馈的统一审核队列",
        rubric_instructions="核对事实来源、法律引用、策略风险和文书可用性。",
    )

    print(json.dumps({
        "project": settings.LANGSMITH_PROJECT,
        "prompts": prompt_urls,
        "dataset": {"id": str(dataset.id), "name": DATASET_NAME, "examples": len(cases)},
        "annotation_queue": {"id": str(queue.id), "name": QUEUE_NAME},
        "deployment": "langgraph.json",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
