#!/usr/bin/env python3
"""Build the small, business-focused curated legal collection."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.documents import Document


REQUIRED_FIELDS = {
    "document_id",
    "law_name",
    "article_number",
    "source",
    "document_type",
    "effective_status",
    "version",
    "source_url",
    "content",
    "business_summary",
}


def load_records(path: Path) -> list[dict]:
    records = []
    seen = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            missing = REQUIRED_FIELDS - item.keys()
            if missing:
                raise ValueError(f"第 {line_number} 行缺少字段: {sorted(missing)}")
            key = (item["law_name"], item["article_number"], item["version"])
            if key in seen:
                raise ValueError(f"第 {line_number} 行存在重复条款版本: {key}")
            seen.add(key)
            records.append(item)
    if not records:
        raise ValueError("精选知识库不能为空")
    return records


def to_documents(records: list[dict]) -> list[Document]:
    documents = []
    for item in records:
        scenarios = "；".join(item.get("business_scenarios", []))
        parent_content = (
            f"《{item['law_name']}》{item['article_number']}\n"
            f"条文原文：{item['content']}\n"
            f"业务适用：{item['business_summary']}"
        )
        search_text = (
            f"{item['law_name']} {item['article_number']} "
            f"业务场景：{scenarios} {parent_content}"
        )
        metadata = {
            **item,
            "title": item["law_name"],
            "location": item["article_number"],
            "parent_id": item["document_id"],
            "parent_content": parent_content,
            "chunk_id": f"{item['document_id']}-v1",
            "timestamp": item.get("effective_date", ""),
            "access_level": "public",
            "lifecycle_status": "published",
            "file_path": str(Path("data/curated/legal_knowledge_v1.jsonl")),
        }
        documents.append(Document(page_content=search_text, metadata=metadata))
    return documents


def ingest(path: Path, collection: str, drop_existing: bool = True) -> int:
    from rag_qa.core.vector_store import VectorStore

    records = load_records(path)
    store = VectorStore(collection_name=collection)
    if drop_existing and store.client.has_collection(collection):
        store.client.drop_collection(collection)
        store._create_or_load_collection()
    documents = to_documents(records)
    store.add_documents(documents, batch_size=50)
    return len(documents)


def main():
    parser = argparse.ArgumentParser(description="构建精选法律业务知识库")
    parser.add_argument(
        "--input",
        default="data/curated/legal_knowledge_v1.jsonl",
    )
    parser.add_argument(
        "--collection",
        default="fzt_legal_curated_v1",
    )
    parser.add_argument("--no-drop", action="store_true")
    args = parser.parse_args()
    count = ingest(
        Path(args.input),
        args.collection,
        drop_existing=not args.no_drop,
    )
    print(f"精选知识库入库完成：collection={args.collection}, records={count}")


if __name__ == "__main__":
    main()
