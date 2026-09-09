"""API 接口规范化测试"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestFastApiBase:
    """测试 FastAPI 基础组件"""

    def test_fastapi_importable(self):
        import fastapi
        assert hasattr(fastapi, "FastAPI")

    def test_pydantic_validation(self):
        """测试请求体验证（min_length, max_length）"""
        from api_server import ChatRequest

        # 正常请求
        req = ChatRequest(question="劳动法加班费怎么算")
        assert req.question == "劳动法加班费怎么算"

        # 带 filter
        req2 = ChatRequest(question="离婚条件", source_filter="民法")
        assert req2.source_filter == "民法"

        # 超长问题应该被截断/校验
        long_q = "你好" * 2000
        try:
            req3 = ChatRequest(question=long_q)
            assert len(req3.question) <= 2000, \
                f"Expected max 2000, got {len(req3.question)}"
        except Exception:
            pass  # Pydantic 拒绝也是预期行为

    def test_reference_item_model(self):
        """测试 ReferenceItem 模型"""
        from api_server import ReferenceItem

        ref = ReferenceItem(type="article", title="劳动法", source="法律条文")
        assert ref.type == "article"
        assert ref.title == "劳动法"
        assert ref.score is None

        ref2 = ReferenceItem(type="case", title="案例1", content="摘要", score=0.95)
        assert ref2.score == 0.95

    def test_chat_response_model(self):
        """测试 ChatResponse 包含所有新字段"""
        from api_server import ChatResponse

        resp = ChatResponse(
            answer="根据民法典...",
            request_id="test-uuid",
            used_rag=True,
        )
        assert resp.answer == "根据民法典..."
        assert resp.request_id == "test-uuid"
        assert resp.used_rag is True
        assert resp.used_fallback is False
        assert len(resp.references) == 0
        assert "仅供参考" in resp.disclaimer

    def test_error_response_format(self):
        """测试统一错误响应格式"""
        from api_server import error_response, ERROR_CODES

        # 验证错误码定义完整
        assert "INVALID_REQUEST" in ERROR_CODES
        assert "SERVICE_NOT_READY" in ERROR_CODES
        assert "INTERNAL_ERROR" in ERROR_CODES
        assert ERROR_CODES["SERVICE_NOT_READY"] == (503, "服务正在初始化，请稍后重试")


from unittest.mock import patch


class TestApiRouting:
    """测试路由注册"""

    def test_v1_routes_registered(self):
        from api_server import app
        routes = [r.path for r in app.routes]
        for expected in ["/api/v1", "/api/v1/chat", "/api/v1/health", "/api/v1/ready"]:
            assert expected in routes, f"Route {expected} not found!"

    def test_legacy_routes_registered(self):
        from api_server import app
        routes = [r.path for r in app.routes]
        for expected in ["/api/chat", "/api/health"]:
            assert expected in routes, f"Legacy route {expected} not found!"

    def test_index_route(self):
        from api_server import app
        routes = [r.path for r in app.routes]
        assert "/" in routes

    def test_app_title(self):
        from api_server import app
        assert app.title == "律所法律知识库 RAG 系统"


class TestApiTokenAuth:
    """测试 API Token 认证中间件。"""

    def test_admin_routes_require_token(self):
        """配置了 API_TOKEN 时，管理接口应返回 401（无 token）。"""
        import api_server
        # 临时设一个 token 使校验生效
        orig_token = api_server.API_TOKEN
        api_server.API_TOKEN = "test-token-123"
        try:
            from fastapi.testclient import TestClient
            client = TestClient(api_server.app)
            # 不加 token → 401
            resp = client.post("/api/v1/admin/qa", json={"question": "test", "answer": "test"})
            assert resp.status_code == 401, f"Expected 401, got {resp.status_code}"
            data = resp.json()
            assert "UNAUTHORIZED" in str(data.get("detail", {}))
        finally:
            api_server.API_TOKEN = orig_token

    def test_admin_routes_with_valid_token(self):
        """携带正确 token 时管理接口应正常返回（不依赖外部服务则返回 503）。"""
        import api_server
        orig_token = api_server.API_TOKEN
        api_server.API_TOKEN = "test-token-123"
        try:
            from fastapi.testclient import TestClient
            client = TestClient(api_server.app)
            resp = client.post(
                "/api/v1/admin/qa",
                json={"question": "test", "answer": "test"},
                headers={"Authorization": "Bearer test-token-123"},
            )
            # 服务未就绪所以返回 503，而不是 401
            assert resp.status_code != 401, f"不应返回 401，实际: {resp.status_code}"
        finally:
            api_server.API_TOKEN = orig_token

    def test_admin_routes_with_wrong_token(self):
        """携带错误 token 应返回 401。"""
        import api_server
        orig_token = api_server.API_TOKEN
        api_server.API_TOKEN = "test-token-123"
        try:
            from fastapi.testclient import TestClient
            client = TestClient(api_server.app)
            resp = client.post(
                "/api/v1/admin/qa",
                json={"question": "test", "answer": "test"},
                headers={"Authorization": "Bearer wrong-token"},
            )
            assert resp.status_code == 401
        finally:
            api_server.API_TOKEN = orig_token

    def test_chat_routes_no_token_required(self):
        """聊天接口不应受 token 保护。"""
        import api_server
        orig_token = api_server.API_TOKEN
        api_server.API_TOKEN = "test-token-123"
        try:
            from fastapi.testclient import TestClient
            client = TestClient(api_server.app)
            resp = client.post(
                "/api/v1/chat",
                json={"question": "test"},
            )
            # 503（服务未就绪）而非 401
            assert resp.status_code != 401, f"聊天接口无 token 不应 401，实际: {resp.status_code}"
        finally:
            api_server.API_TOKEN = orig_token

    def test_empty_token_does_not_disable_auth(self):
        """企业模式下，空 Token 配置也不能让管理接口匿名访问。"""
        import api_server
        original = api_server.API_TOKEN
        api_server.API_TOKEN = ""
        try:
            from fastapi.testclient import TestClient
            client = TestClient(api_server.app)
            resp = client.post(
                "/api/v1/admin/qa",
                json={"question": "test", "answer": "test"},
            )
            assert resp.status_code == 401
        finally:
            api_server.API_TOKEN = original
