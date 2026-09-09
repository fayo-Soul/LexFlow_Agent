#!/usr/bin/env python3
"""端到端工作流测试 - 模拟完整案件分析流程"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/tmp/lexflow-venv/lib/python3.10/site-packages')

from lexflow_agent.engine.models.case import CaseInput, CaseType, ClientRole, CaseDocument, PageContent
from lexflow_agent.engine.models.enums import WorkflowStatus
from lexflow_agent.engine.state import CaseAgentState
from lexflow_agent.engine.nodes.validation import validate_input_node


def load_demo_case(case_id: str = "demo_001") -> dict:
    """加载评测数据中的某个案件"""
    for candidate in [
        Path(__file__).parent.parent / "data" / "eval" / "dev_set" / "sample_cases.json",
        Path(__file__).parent.parent / "lexflow_agent" / "data" / "eval" / "dev_set" / "sample_cases.json",
    ]:
        if candidate.exists():
            path = candidate
            break
    else:
        print(f"[ERROR] 评测数据不存在")
        return {}
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)
    for c in cases:
        if c["case_id"] == case_id:
            return c
    print(f"[ERROR] 未找到案件: {case_id}")
    return {}


def test_pipeline_from_input(case_data: dict) -> bool:
    """从 CaseInput 到 validation，验证工作流基本可执行"""
    print(f"\n{'='*60}")
    print(f"测试案件: {case_data.get('case_id', 'unknown')}")
    print(f"诉求: {case_data.get('client_goal', '')[:40]}...")
    print(f"{'='*60}")

    # 1. 构造输入
    case_input = CaseInput(
        case_id=case_data["case_id"],
        case_type=CaseType(case_data["case_type"]),
        client_role=ClientRole(case_data["client_role"]),
        client_goal=case_data["client_goal"],
        documents=[
            CaseDocument(
                document_id=d["document_id"],
                file_name=d["file_name"],
                pages=[PageContent(page=p["page"], text=p["text"]) for p in d["pages"]],
            ) for d in case_data["documents"]
        ],
    )

    print(f"  文档数: {len(case_input.documents)}")
    total_pages = sum(len(d.pages) for d in case_input.documents)
    print(f"  总页数: {total_pages}")
    total_chars = sum(len(p.text) for d in case_input.documents for p in d.pages)
    print(f"  总字符: {total_chars}")

    # 2. 验证
    state = CaseAgentState(
        run_id=f"test_{case_data['case_id']}",
        case_id=case_data["case_id"],
        case_input=case_input,
        status=WorkflowStatus.PENDING,
    )
    result = validate_input_node(state)

    if result.get("status") == WorkflowStatus.FAILED:
        print(f"  [FAIL] 校验失败: {result.get('errors', [])}")
        return False

    print(f"  [PASS] 校验通过")
    print(f"  状态: {result.get('status', 'unknown')}")
    print(f"  阶段: {result.get('phase', 'unknown')}")

    # 3. 验证标注数据
    annotations = case_data.get("annotations", {})
    if annotations:
        print(f"  标注事实: {len(annotations.get('facts', []))} 条")
        print(f"  标注争点: {annotations.get('issues', [])}")
        print(f"  适用法律: {len(annotations.get('applicable_laws', []))} 条")

    return True


def run_all():
    """执行所有评测数据中的案件测试"""
    for candidate in [
        Path(__file__).parent.parent / "data" / "eval" / "dev_set" / "sample_cases.json",
        Path(__file__).parent.parent / "lexflow_agent" / "data" / "eval" / "dev_set" / "sample_cases.json",
    ]:
        if candidate.exists():
            path = candidate
            break
    else:
        print("[ERROR] 评测数据不存在，跳过")
        return

    with open(path, encoding="utf-8") as f:
        cases = json.load(f)

    print(f"共加载 {len(cases)} 个评测案件")
    passed = 0
    failed = 0

    for case in cases:
        if test_pipeline_from_input(case):
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"结果: {passed}/{passed+failed} 通过")
    print(f"{'='*60}")
    return passed == len(cases)


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
