import io

import pytest

from app.knowledge_base import KnowledgeBaseError, KnowledgeBaseManager


def test_document_lifecycle(tmp_path):
    manager = KnowledgeBaseManager(tmp_path / "knowledge_base")

    created = manager.create(
        file_name="research.md",
        file_object=io.BytesIO("# 股权转让研究".encode("utf-8")),
        document_type="internal",
        title="股权转让研究",
    )

    assert created["processing_status"] == "uploaded"
    assert manager.get(created["id"])["title"] == "股权转让研究"

    updated = manager.update_status(created["id"], "indexed", chunk_count=3)
    assert updated["processing_status"] == "indexed"
    assert updated["chunk_count"] == 3

    assert manager.delete(created["id"]) is True
    assert manager.get(created["id"]) is None


def test_duplicate_document_is_rejected(tmp_path):
    manager = KnowledgeBaseManager(tmp_path / "knowledge_base")
    content = "相同资料".encode("utf-8")
    manager.create("one.txt", io.BytesIO(content), "regulation")

    with pytest.raises(KnowledgeBaseError, match="文件已存在"):
        manager.create("two.txt", io.BytesIO(content), "regulation")


def test_unsupported_document_type_is_rejected(tmp_path):
    manager = KnowledgeBaseManager(tmp_path / "knowledge_base")

    with pytest.raises(KnowledgeBaseError, match="不支持的文档类型"):
        manager.create("file.txt", io.BytesIO(b"content"), "agent")
