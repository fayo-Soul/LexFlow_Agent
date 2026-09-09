from app.citations import (
    build_case_citation,
    build_regulation_citation,
    validate_citations,
)
from main import LegalResearchService


class FakeDocument:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata or {}


def test_regulation_citation_is_traceable():
    document = FakeDocument(
        "当事人一方不履行合同义务的，应当承担违约责任。",
        {
            "document_id": "law-001",
            "chunk_id": "law-001-577",
            "law_name": "中华人民共和国民法典",
            "article_number": "第五百七十七条",
            "source": "民法",
            "effective_status": "effective",
        },
    )

    citation = build_regulation_citation(document, 1)

    assert citation["citation_id"] == "法规1"
    assert citation["document_id"] == "law-001"
    assert citation["chunk_id"] == "law-001-577"
    assert citation["location"] == "第五百七十七条"
    assert citation["quote"] == document.page_content


def test_legacy_regulation_without_document_type_uses_safe_default():
    document = FakeDocument(
        "旧集合中的法律条文",
        {"document_type": None, "source": "民法"},
    )

    citation = build_regulation_citation(document, 1)

    assert citation["source_type"] == "regulation"
    assert citation["citation_id"] == "法规1"


def test_internal_document_uses_internal_label_and_managed_title():
    document = FakeDocument(
        "律所内部研究内容",
        {
            "document_type": "internal",
            "title": "股权转让内部研究",
            "law_name": "uploaded-file",
        },
    )

    citation = build_regulation_citation(document, 1)

    assert citation["citation_id"] == "内部资料1"
    assert citation["title"] == "股权转让内部研究"


def test_case_citation_uses_real_case_metadata():
    citation = build_case_citation(
        {
            "id": "case-001",
            "title": "某公司诉王某劳动合同纠纷案",
            "case_number": "（2025）某民终1号",
            "court": "某中级人民法院",
            "child_chunk": "法院认为，用人单位应当依法支付补偿。",
            "rerank_score": 0.91,
        },
        1,
    )

    assert citation["citation_id"] == "案例1"
    assert citation["document_id"] == "case-001"
    assert citation["location"] == "（2025）某民终1号"
    assert citation["metadata"]["court"] == "某中级人民法院"


def test_unknown_citation_is_rejected():
    citations = [
        {
            "citation_id": "法规1",
            "document_id": "law-001",
            "chunk_id": "chunk-001",
            "source_type": "regulation",
            "title": "中华人民共和国民法典",
            "location": "第五百七十七条",
            "quote": "原文",
        }
    ]

    valid, unknown = validate_citations(
        "结论依据如下。[法规1] 另有案例支持。[案例9]",
        citations,
    )

    assert valid is False
    assert unknown == ["案例9"]


def test_answer_without_reference_is_rejected_when_evidence_exists():
    citations = [
        {
            "citation_id": "法规1",
            "document_id": "law-001",
            "chunk_id": "chunk-001",
            "source_type": "regulation",
            "title": "中华人民共和国民法典",
            "location": "第五百七十七条",
            "quote": "原文",
        }
    ]

    valid, unknown = validate_citations("这是一个没有引用的结论。", citations)

    assert valid is False
    assert unknown == []


def test_automatic_domain_classification_is_not_used_as_hard_filter():
    service = LegalResearchService.__new__(LegalResearchService)
    service.rag_system = type(
        "FakeRagSystem",
        (),
        {"query_classifier": type("Classifier", (), {"predict": lambda self, query: "其他法"})()},
    )()
    captured = {}

    def fake_search(query, source_filter=None, scope=None, filters=None):
        captured["source_filter"] = source_filter
        return {
            "citations": [],
            "retrieval": {
                "strategy": "hybrid",
                "retrieved_count": 0,
                "reranked_count": 0,
            },
        }

    service.search = fake_search

    result = service.answer("违法解除劳动合同后能否要求继续履行？")

    assert captured["source_filter"] is None
    assert result["classified_domain"] == "其他法"
