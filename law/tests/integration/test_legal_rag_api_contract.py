def test_legal_rag_routes_are_registered():
    from api_server import app

    routes = {route.path for route in app.routes}

    assert "/api/v1/search" in routes
    assert "/api/v1/ask" in routes
    assert "/api/v1/documents" in routes
    assert "/api/v1/documents/{document_id}" in routes
    assert "/api/v1/documents/{document_id}/reindex" in routes


def test_product_identity_is_law_firm_rag():
    from api_server import app

    assert app.title == "律所法律知识库 RAG 系统"


def test_frontend_entry_is_not_cached():
    from api_server import index

    response = index()

    assert response.headers["cache-control"] == "no-store"


def test_ask_request_defaults_to_hybrid_retrieval_and_automatic_scope():
    from api_server import AskRequest

    request = AskRequest(question="违法解除劳动合同的法律后果是什么？")

    assert request.retrieval_mode == "hybrid"
    assert request.scope is None


def test_citation_response_has_traceability_fields():
    from api_server import CitationItem

    citation = CitationItem(
        citation_id="法规1",
        document_id="law-001",
        chunk_id="chunk-001",
        source_type="regulation",
        title="中华人民共和国民法典",
        location="第五百七十七条",
        quote="当事人一方不履行合同义务……",
    )

    assert citation.document_id == "law-001"
    assert citation.chunk_id == "chunk-001"
    assert citation.citation_id == "法规1"
