"""检索证据的引用构建与校验。"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable


REFERENCE_PATTERN = re.compile(r"\[(法规|司法解释|案例|内部资料)\d+\]")


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}-{digest}"


def _score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_regulation_citation(document: Any, index: int) -> dict[str, Any]:
    metadata = getattr(document, "metadata", {}) or {}
    quote = (getattr(document, "page_content", "") or "").strip()
    source_type = metadata.get("document_type") or "regulation"
    label = {
        "judicial_interpretation": "司法解释",
        "internal": "内部资料",
    }.get(source_type, "法规")
    if source_type == "internal":
        title = metadata.get("title") or metadata.get("law_name")
    else:
        title = metadata.get("law_name") or metadata.get("title")
    title = title or metadata.get("source") or "法律资料"
    location = metadata.get("article_number") or metadata.get("location") or metadata.get("section") or ""
    document_id = metadata.get("document_id") or _stable_id("doc", f"{title}:{metadata.get('file_path', '')}")
    chunk_id = metadata.get("chunk_id") or metadata.get("id") or _stable_id("chunk", f"{document_id}:{quote}")

    return {
        "citation_id": f"{label}{index}",
        "document_id": str(document_id),
        "chunk_id": str(chunk_id),
        "source_type": source_type,
        "title": str(title),
        "location": str(location),
        "quote": quote,
        "source": str(metadata.get("source", "")),
        "version": str(metadata.get("version", "")),
        "effective_status": str(metadata.get("effective_status", "unknown")),
        "score": _score(metadata.get("rerank_score", metadata.get("score"))),
        "metadata": {
            key: value
            for key, value in {
                "issuer": metadata.get("issuer"),
                "publish_date": metadata.get("publish_date"),
                "effective_date": metadata.get("effective_date"),
                "promulgation_date": metadata.get("promulgation_date"),
                "source_url": metadata.get("source_url"),
                "file_path": metadata.get("file_path"),
            }.items()
            if value not in (None, "")
        },
    }


def build_case_citation(case: dict[str, Any], index: int) -> dict[str, Any]:
    quote = (
        case.get("child_chunk")
        or case.get("court_view")
        or case.get("ruling")
        or case.get("fact")
        or ""
    ).strip()
    title = case.get("title") or "案例资料"
    case_number = case.get("case_number") or ""
    document_id = case.get("document_id") or case.get("id") or _stable_id(
        "case", f"{title}:{case_number}"
    )
    chunk_id = case.get("chunk_id") or _stable_id("chunk", f"{document_id}:{quote}")

    return {
        "citation_id": f"案例{index}",
        "document_id": str(document_id),
        "chunk_id": str(chunk_id),
        "source_type": "case",
        "title": str(title),
        "location": str(case_number),
        "quote": quote,
        "source": str(case.get("source", "")),
        "version": str(case.get("version", "")),
        "effective_status": "not_applicable",
        "score": _score(case.get("rerank_score", case.get("score"))),
        "metadata": {
            key: value
            for key, value in {
                "case_number": case_number,
                "court": case.get("court"),
                "judgment_date": case.get("judgment_date"),
                "cause": case.get("cause") or case.get("domain"),
            }.items()
            if value not in (None, "")
        },
    }


def build_citations(article_docs: Iterable[Any], cases: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    citations = []
    regulation_index = 0
    interpretation_index = 0
    internal_index = 0
    for document in article_docs:
        document_type = (getattr(document, "metadata", {}) or {}).get("document_type")
        if document_type == "judicial_interpretation":
            interpretation_index += 1
            index = interpretation_index
        elif document_type == "internal":
            internal_index += 1
            index = internal_index
        else:
            regulation_index += 1
            index = regulation_index
        citations.append(build_regulation_citation(document, index))

    citations.extend(build_case_citation(case, index) for index, case in enumerate(cases, 1))
    return citations


def build_evidence_context(citations: Iterable[dict[str, Any]]) -> str:
    sections = []
    for citation in citations:
        metadata = citation.get("metadata") or {}
        details = [
            citation["title"],
            citation.get("location", ""),
            metadata.get("court", ""),
            metadata.get("case_number", ""),
        ]
        heading = " | ".join(str(item) for item in details if item)
        sections.append(
            f"[{citation['citation_id']}] {heading}\n"
            f"原文：{citation.get('quote', '')}"
        )
    return "\n\n".join(sections)


def validate_citations(answer: str, citations: Iterable[dict[str, Any]]) -> tuple[bool, list[str]]:
    available = {item["citation_id"] for item in citations}
    referenced = {match.group(0)[1:-1] for match in REFERENCE_PATTERN.finditer(answer or "")}
    unknown = sorted(referenced - available)
    if unknown:
        return False, unknown
    if available and not referenced:
        return False, []
    return True, []
