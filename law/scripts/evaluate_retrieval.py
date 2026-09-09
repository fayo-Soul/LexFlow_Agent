#!/usr/bin/env python3
"""Evaluate curated legal retrieval without paying LLM generation latency."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.evaluate_rag import calculate_recall_at_k


def percentile(values: list[float], value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(len(ordered) * value) - 1)
    return ordered[index]


def evaluate(dataset: Path, output: Path, min_recall: float, max_p95_ms: float):
    from main import LegalResearchService

    pairs = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    service = LegalResearchService()
    results = []
    recalls = []
    latencies = []
    for pair in pairs:
        started = time.perf_counter()
        response = service.search(
            pair["question"],
            scope=["regulation", "judicial_interpretation"],
        )
        elapsed = (time.perf_counter() - started) * 1000
        citations = response["citations"]
        recall = calculate_recall_at_k(citations, pair.get("golden_docs", []))
        recalls.append(recall or 0.0)
        latencies.append(elapsed)
        results.append(
            {
                "question": pair["question"],
                "golden_docs": pair.get("golden_docs", []),
                "retrieved": [
                    f"{item['title']} {item.get('location', '')}".strip()
                    for item in citations
                ],
                "recall_at_k": recall,
                "latency_ms": round(elapsed, 1),
            }
        )
    average_recall = sum(recalls) / len(recalls) if recalls else 0.0
    p95 = percentile(latencies, 0.95)
    gates = {
        "average_recall": average_recall >= min_recall,
        "p95_latency": p95 <= max_p95_ms,
        "all_queries_return_evidence": all(item["retrieved"] for item in results),
    }
    report = {
        "dataset": str(dataset),
        "samples": len(pairs),
        "metrics": {
            "average_recall_at_k": round(average_recall, 4),
            "p50_latency_ms": round(percentile(latencies, 0.50), 1),
            "p95_latency_ms": round(p95, 1),
        },
        "quality_gates": gates,
        "passed": all(gates.values()),
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False))
    print(f"passed={report['passed']}, report={output}")
    return report


def main():
    parser = argparse.ArgumentParser(description="精选知识库检索质量评测")
    parser.add_argument(
        "--dataset",
        default="data/eval/curated_business_v1.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--output",
        default="data/eval/curated_retrieval_report.json",
        type=Path,
    )
    parser.add_argument("--min-recall", default=0.80, type=float)
    parser.add_argument("--max-p95-ms", default=5000, type=float)
    args = parser.parse_args()
    report = evaluate(
        args.dataset,
        args.output,
        args.min_recall,
        args.max_p95_ms,
    )
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
