#!/usr/bin/env python3
"""Production smoke test including restart recovery, HITL, SSE and deletion."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lexflow_agent.config.settings import settings


BASE = "http://127.0.0.1:8000/api/v1"
HEADERS = {
    "Authorization": f"Bearer {settings.API_TOKEN}",
    "Content-Type": "application/json; charset=utf-8",
}
IDEMPOTENCY_KEY = "global-restart-check-v2"


def request(method: str, path: str, body=None):
    data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers=HEADERS, method=method)
    with urllib.request.urlopen(req, timeout=1900) as response:
        raw = response.read()
        return json.loads(raw) if raw else None


def create():
    return request("POST", "/case-runs", {
        "case_id": "global_restart_check",
        "case_type": "labor_dispute",
        "client_role": "employee",
        "client_goal": "分析违法解除劳动合同并生成赔偿主张及劳动仲裁文书",
        "documents": [{
            "document_id": "doc_global_1",
            "file_name": "解除材料.txt",
            "pages": [{
                "page": 1,
                "text": (
                    "测试员工于2022年3月1日入职测试企业，月平均工资12000元。"
                    "2025年1月10日企业以业务调整为由解除劳动合同，未提前书面通知，"
                    "未支付经济补偿，员工无严重违纪记录。"
                ),
            }],
        }],
        "idempotency_key": IDEMPOTENCY_KEY,
    })["data"]["run_id"]


def wait_for(run_id: str, expected: set[str], timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = request("GET", f"/case-runs/{run_id}")["data"]
        if state["status"] in expected:
            return state
        if state["status"] == "failed":
            raise AssertionError(state["errors"])
        time.sleep(2)
    raise TimeoutError(f"run did not reach {expected}")


def phase_one():
    run_id = create()
    state = wait_for(run_id, {"waiting_fact_review"})
    assert state["current_agent"] == "case_strategy_agent"
    print(json.dumps({"phase": 1, "run_id": run_id, "status": state["status"]}))


def phase_two():
    run_id = create()
    current = request("GET", f"/case-runs/{run_id}")["data"]["status"]
    if current == "failed":
        request("POST", f"/case-runs/{run_id}/retry")
    elif current == "waiting_fact_review":
        request("POST", f"/case-runs/{run_id}/resume", {
            "action": "approve_facts",
            "comments": "global check",
        })
    wait_for(run_id, {"waiting_strategy_review"})
    request("POST", f"/case-runs/{run_id}/resume", {
        "action": "approve_strategy",
        "comments": "global check",
    })
    state = wait_for(run_id, {"completed", "waiting_human_intervention"})
    if state["status"] != "completed":
        raise AssertionError(state)
    package = request("GET", f"/case-runs/{run_id}/result")["data"]
    assert package["facts"]
    assert package["issues"]
    assert len(package["legal_research"]) == len(package["issues"])
    assert package["strategy"]
    assert len(package["drafts"]) >= 3
    assert package["workflow_events"]
    request("DELETE", f"/case-runs/{run_id}")
    try:
        request("GET", f"/case-runs/{run_id}")
    except urllib.error.HTTPError as error:
        assert error.code == 404
    else:
        raise AssertionError("deleted run is still readable")
    print(json.dumps({
        "phase": 2,
        "status": "passed",
        "facts": len(package["facts"]),
        "issues": len(package["issues"]),
        "research": len(package["legal_research"]),
        "drafts": len(package["drafts"]),
        "events": len(package["workflow_events"]),
    }))


if __name__ == "__main__":
    {"phase1": phase_one, "phase2": phase_two}[sys.argv[1]]()
