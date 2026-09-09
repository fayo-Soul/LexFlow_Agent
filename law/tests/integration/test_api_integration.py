"""API 集成测试（使用 TestClient）。"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """测试健康检查端点（不依赖外部服务）。"""

    def test_health_returns_json(self):
        """验证 /api/v1/health 返回正确格式。"""
        from api_server import app
        client = TestClient(app)
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "ready" in data
        assert "version" in data

    def test_ready_returns_503_when_not_ready(self):
        """未就绪时 /api/v1/ready 应返回 503。"""
        from api_server import app
        client = TestClient(app)
        resp = client.get("/api/v1/ready")
        # 未初始化时 app.state.ready = False → 503
        assert resp.status_code in (200, 503)
        data = resp.json()
        assert "ready" in data


class TestMonitoring:
    """测试监控指标。"""

    def test_metrics_endpoint(self):
        """验证 /metrics 返回监控指标。"""
        from api_server import app
        client = TestClient(app)
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")
        assert "law_rag_http_requests_total" in resp.text
        assert "law_rag_http_errors_total" in resp.text
        assert "law_rag_index_queue_depth" in resp.text

    def test_metrics_count_tracks_requests(self):
        """模拟一次请求后验证计数器自增。"""
        import api_server as asrv
        orig_count = asrv.REQUEST_COUNT
        from fastapi.testclient import TestClient
        client = TestClient(asrv.app)
        client.get("/api/v1/health")
        assert asrv.REQUEST_COUNT == orig_count + 1, "REQUEST_COUNT 应自增"
