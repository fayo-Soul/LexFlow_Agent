import io

import pytest

from app.intent import route_query
from app.auth import authenticate_bearer
from app.knowledge_base import KnowledgeBaseError, KnowledgeBaseManager
from case_rag.sparse import stable_token_id


@pytest.mark.parametrize(
    ("query", "intent"),
    [
        ("这个系统能帮我查询什么？", "product_help"),
        ("你能帮我干什么", "product_help"),
        ("你可以帮我做什么？", "product_help"),
        ("你都会什么", "product_help"),
        ("你好", "greeting"),
        ("谢谢", "courtesy"),
        ("今天天气怎么样？", "out_of_scope"),
        ("违法解除劳动合同后能否继续履行？", "legal_research"),
    ],
)
def test_query_intent_routing(query, intent):
    assert route_query(query).intent == intent


def test_uploaded_filename_cannot_escape_staging_directory(tmp_path):
    manager = KnowledgeBaseManager(tmp_path / "knowledge_base")

    with pytest.raises(KnowledgeBaseError, match="文件名"):
        manager.create(
            file_name="../../escape.md",
            file_object=io.BytesIO(b"content"),
            document_type="internal",
        )


def test_upload_size_limit_is_enforced(tmp_path):
    manager = KnowledgeBaseManager(tmp_path / "knowledge_base", max_file_size=4)

    with pytest.raises(KnowledgeBaseError, match="文件大小"):
        manager.create(
            file_name="large.txt",
            file_object=io.BytesIO(b"12345"),
            document_type="internal",
        )


@pytest.mark.parametrize(
    ("file_name", "content"),
    [
        ("fake.pdf", b"plain text"),
        ("fake.docx", b"plain text"),
        ("bad.txt", b"\xff\xfe"),
    ],
)
def test_uploaded_content_must_match_declared_format(tmp_path, file_name, content):
    manager = KnowledgeBaseManager(tmp_path / "knowledge_base")

    with pytest.raises(KnowledgeBaseError):
        manager.create(
            file_name=file_name,
            file_object=io.BytesIO(content),
            document_type="internal",
        )


def test_case_sparse_token_id_is_stable():
    assert stable_token_id("法") == stable_token_id("法")
    assert stable_token_id("法") != stable_token_id("律")


def test_researcher_and_admin_tokens_have_distinct_roles():
    admin = authenticate_bearer("Bearer admin-token", "admin-token", "research-token")
    researcher = authenticate_bearer(
        "Bearer research-token", "admin-token", "research-token"
    )

    assert admin.role == "admin"
    assert researcher.role == "researcher"
    assert authenticate_bearer("Bearer wrong", "admin-token", "research-token") is None
