import io

from app.knowledge_base import KnowledgeBaseManager
from app.rate_limit import DistributedRateLimiter
from base.session_manager import SessionManager
from scripts.evaluate_rag import calculate_recall_at_k


class FakePipeline:
    def __init__(self, store):
        self.store = store
        self.key = None

    def incr(self, key):
        self.key = key
        self.store[key] = self.store.get(key, 0) + 1
        return self

    def expire(self, key, seconds):
        return self

    def execute(self):
        return [self.store[self.key], True]


class FakeRedis:
    def __init__(self):
        self.store = {}

    def pipeline(self, transaction=True):
        return FakePipeline(self.store)


def test_document_lifecycle_defaults_to_draft_and_can_be_published(tmp_path):
    manager = KnowledgeBaseManager(tmp_path / "kb")
    item = manager.create(
        "memo.md",
        io.BytesIO("内部研究".encode("utf-8")),
        "internal",
    )

    assert item["lifecycle_status"] == "draft"
    manager.update_lifecycle(item["id"], "pending_review", "admin")
    published = manager.update_lifecycle(item["id"], "published", "admin")

    assert published["lifecycle_status"] == "published"
    assert published["reviewed_by"] == "admin"


def test_distributed_limiter_uses_shared_counter():
    redis = FakeRedis()
    first = DistributedRateLimiter(redis, limit=2, window_seconds=60)
    second = DistributedRateLimiter(redis, limit=2, window_seconds=60)

    assert first.allow("user") is True
    assert second.allow("user") is True
    assert first.allow("user") is False


def test_session_ids_are_signed_and_tampering_creates_new_session(monkeypatch):
    monkeypatch.setattr("base.session_manager._get_redis_client", lambda: None)
    manager = SessionManager()

    session_id, _ = manager.get_or_create()
    replacement, _ = manager.get_or_create(session_id + "tampered")

    assert "." in session_id
    assert replacement != session_id + "tampered"


def test_recall_uses_document_title_and_article_location():
    recall = calculate_recall_at_k(
        [{"title": "中华人民共和国民法典", "location": "第一千零八十四条"}],
        ["民法典 第一千零八十四条"],
    )

    assert recall == 1.0
