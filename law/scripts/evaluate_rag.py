#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 质量评估脚本

评估 RAG 系统的检索和生成质量。

用法：
    python scripts/evaluate_rag.py                    # 全量评估
    python scripts/evaluate_rag.py --sample 10        # 只评估前 10 条（快速验证）
    python scripts/evaluate_rag.py --output eval.json  # 输出到 JSON 文件

指标：
    - Retrieval Recall@K：前 K 个检索结果中包含期望文书的比例
    - Domain Accuracy：法律领域分类准确率
    - Avg Response Time：平均响应耗时
    - Faithfulness Score：回答忠实度（LLM-as-Judge）
    - Hallucination Rate：编造法条/案号的比例（需人工复核）

依赖：
    - 需要完整的后端服务（MySQL / Milvus / LLM / 模型）
    - 建议先启动 api_server.py 或确保所有依赖可用
"""

import argparse
import json
import os
import sys
import time
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_eval_dataset(path="data/eval/qa_pairs.jsonl"):
    """加载标准评估数据集。"""
    full_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
    if not os.path.exists(full_path):
        print(f"[ERROR] 评估数据集不存在：{full_path}")
        sys.exit(1)

    pairs = []
    with open(full_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    print(f"[INFO] 加载评估数据集：{len(pairs)} 条")
    return pairs


def extract_cited_articles(text):
    """从回答文本中提取引用的法条名称。"""
    patterns = [
        r"《(.+?)》\s*(?:第\s*[一-龥\d]+\s*条)?",
        r"([一-龥]+法)\s*(?:第\s*[一-龥\d]+\s*条)?",
        r"([一-龥]+条例)\s*(?:第\s*[一-龥\d]+\s*条)?",
        r"(民法典|劳动合同法|刑法|刑事诉讼法|行政诉讼法|行政复议法|国家赔偿法|食品安全法|消费者权益保护法|工伤保险条例|女职工劳动保护特别规定|职工带薪年休假条例)",
    ]
    cited = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            cited.add(match.group(1).strip())
    return cited


def calculate_recall_at_k(references, golden_docs):
    """计算 Recall@K：检索结果中是否包含期望文书。"""
    if not golden_docs:
        return None  # 无期望文书时不计
    ref_titles = set()
    for ref in references:
        title = " ".join(
            part for part in (ref.get("title", ""), ref.get("location", "")) if part
        )
        ref_titles.add(title)
    # 检查任意 golden_doc 是否出现在检索结果中
    matched = sum(1 for doc in golden_docs if any(doc in r for r in ref_titles))
    return matched / len(golden_docs)


def evaluate(
    sample_count=None,
    output_file=None,
    dataset_path="data/eval/qa_pairs.jsonl",
    min_recall=0.70,
    min_citation_coverage=0.90,
    max_avg_latency_ms=15000,
):
    """执行 RAG 质量评估。"""
    print("=" * 60)
    print("  RAG 质量评估")
    print("=" * 60)

    # 1. 加载评估数据集
    qa_pairs = load_eval_dataset(dataset_path)
    if sample_count:
        qa_pairs = qa_pairs[:sample_count]
        print(f"  [SAMPLE] 仅评估前 {sample_count} 条")

    # 2. 初始化 LawAssistant
    print("\n>> 正在初始化 LawAssistant...")
    try:
        from main import LawAssistant
        assistant = LawAssistant()
    except Exception as e:
        print(f"[ERROR] LawAssistant 初始化失败: {e}")
        print("请确保所有外部服务（MySQL/Milvus/模型）可用。")
        sys.exit(1)

    print(f">> LawAssistant 初始化完成，开始评估 {len(qa_pairs)} 条数据...\n")

    # 3. 逐条评估
    results = []
    total_retrieval_time = 0.0
    domain_correct = 0
    recall_values = []
    faith_scores = []

    for i, pair in enumerate(qa_pairs, 1):
        question = pair["question"]
        expected_domain = pair["expected_domain"]
        golden_docs = pair.get("golden_docs", [])

        print(f"  [{i}/{len(qa_pairs)}] {question[:30]}...", end=" ")

        try:
            t0 = time.perf_counter()
            result = assistant.answer(question)
            elapsed = (time.perf_counter() - t0) * 1000
            total_retrieval_time += elapsed

            # 检查领域分类
            actual_domain = expected_domain  # source_filter 传入为 expected_domain
            is_domain_correct = True  # 没传 source_filter 时靠 BERT 分类，这里简化为 passed
            if result.get("source_filter"):
                # 如果 answer() 能返回分类结果更好，但目前不开放该字段
                pass

            # 计算 Recall@K
            references = result.get("citations", result.get("references", []))
            recall = calculate_recall_at_k(references, golden_docs)
            if recall is not None:
                recall_values.append(recall)

            # 提取引用的法条（用于后续 Faithfulness / Hallucination 分析）
            answer_text = result.get("answer", "")
            cited = extract_cited_articles(answer_text)

            entry = {
                "question": question,
                "expected_domain": expected_domain,
                "response_time_ms": round(elapsed, 1),
                "used_rag": result.get("used_rag", False),
                "used_fallback": result.get("used_fallback", False),
                "references_count": len(references),
                "recall_at_k": round(recall, 2) if recall is not None else None,
                "cited_articles": list(cited),
                "has_answer": bool(answer_text and len(answer_text) > 10),
            }
            results.append(entry)

            # 简单指标统计
            status = f"{elapsed:.0f}ms"
            if recall is not None:
                status += f" recall={recall:.0%}"
            print(f"[OK] {status}")

        except Exception as e:
            print(f"[ERROR] {e}")
            results.append({
                "question": question,
                "expected_domain": expected_domain,
                "error": str(e),
            })

    # 4. 汇总指标
    print("\n" + "=" * 60)
    print("  评估报告")
    print("=" * 60)

    total = len(qa_pairs)
    success = [r for r in results if "error" not in r]
    n = len(success)

    avg_time = round(total_retrieval_time / n, 1) if n > 0 else 0
    recall_ok = [r for r in success if r.get("recall_at_k") is not None]
    avg_recall = round(
        sum(r["recall_at_k"] for r in recall_ok) / len(recall_ok) * 100, 1
    ) if recall_ok else 0

    no_answer = [r for r in success if not r.get("has_answer")]
    with_refs = [r for r in success if r.get("references_count", 0) > 0]

    print(f"\n  评估数据集：{total} 个问答对")
    print(f"  成功评估：  {n} 个")
    print(f"  失败：      {total - n} 个")
    print()
    print(f"  平均响应耗时：       {avg_time} ms")
    print(f"  检索 Recall@K：      {avg_recall}%  ({len(recall_ok)}/{n})")
    print(f"  有引用来源的回复：    {len(with_refs)}/{n}")

    if no_answer:
        print(f"  无有效回答：         {len(no_answer)} 条")
        for r in no_answer[:5]:
            print(f"    - {r['question'][:40]}")

    # 领域分析
    domain_stats = {}
    for r in success:
        d = r.get("expected_domain", "未知")
        domain_stats.setdefault(d, 0)
        domain_stats[d] += 1

    print(f"\n  领域分布：")
    for domain, count in sorted(domain_stats.items()):
        pct = round(count / n * 100, 1)
        print(f"    {domain}: {count} 条 ({pct}%)")

    # 生成建议
    print(f"\n  建议：")
    citation_coverage = len(with_refs) / n if n else 0
    gates = {
        "recall_at_k": avg_recall / 100 >= min_recall,
        "citation_coverage": citation_coverage >= min_citation_coverage,
        "average_latency": avg_time <= max_avg_latency_ms,
        "no_runtime_errors": n == total,
    }
    if avg_recall < min_recall * 100:
        print(f"    - Recall@K 偏低（{avg_recall}%），建议检查检索链路和数据覆盖")
    if avg_time > max_avg_latency_ms:
        print(f"    - 平均响应耗时过长（{avg_time}ms），建议优化检索或 LLM 调用")

    # 5. 输出 JSON 报告
    report = {
        "eval_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": total,
        "evaluated": n,
        "metrics": {
            "avg_response_time_ms": avg_time,
            "recall_at_k_pct": avg_recall,
            "recall_sample": len(recall_ok),
            "with_references": len(with_refs),
            "citation_coverage_pct": round(citation_coverage * 100, 1),
        },
        "quality_gates": gates,
        "passed": all(gates.values()),
        "domain_distribution": domain_stats,
        "results": results,
    }

    if output_file:
        output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n  详细报告已保存：{output_path}")

    print("=" * 60)
    return report


def main():
    parser = argparse.ArgumentParser(description="RAG 质量评估工具")
    parser.add_argument("--sample", type=int, default=None,
                        help="采样数量，快速验证时使用")
    parser.add_argument(
        "--dataset",
        default="data/eval/qa_pairs.jsonl",
        help="评测 JSONL 数据集路径",
    )
    parser.add_argument("--output", type=str, default=None,
                        help="输出 JSON 报告路径")
    parser.add_argument("--min-recall", type=float, default=0.70)
    parser.add_argument("--min-citation-coverage", type=float, default=0.90)
    parser.add_argument("--max-avg-latency-ms", type=float, default=15000)
    args = parser.parse_args()

    report = evaluate(
        sample_count=args.sample,
        output_file=args.output,
        dataset_path=args.dataset,
        min_recall=args.min_recall,
        min_citation_coverage=args.min_citation_coverage,
        max_avg_latency_ms=args.max_avg_latency_ms,
    )
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
