import json
from pathlib import Path

import pytest

from scripts.ingest_curated_knowledge import load_records, to_documents


def test_curated_knowledge_has_complete_business_metadata():
    path = Path("data/curated/legal_knowledge_v1.jsonl")
    records = load_records(path)

    assert len(records) >= 10
    assert all(item["effective_status"] == "effective" for item in records)
    assert all(item["source_url"].startswith("https://") for item in records)
    assert all(item["business_summary"] for item in records)
    assert len({item["document_id"] for item in records}) == len(records)


def test_curated_documents_keep_citation_location_and_business_context():
    records = load_records(Path("data/curated/legal_knowledge_v1.jsonl"))
    document = to_documents(records[:1])[0]

    assert document.metadata["law_name"] == "中华人民共和国民法典"
    assert document.metadata["article_number"] == "第一千零八十四条"
    assert document.metadata["lifecycle_status"] == "published"
    assert "条文原文：" in document.metadata["parent_content"]
    assert "业务适用：" in document.metadata["parent_content"]


def test_curated_loader_rejects_incomplete_record(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"document_id": "missing"}), encoding="utf-8")

    with pytest.raises(ValueError, match="缺少字段"):
        load_records(path)
