from lexflow_agent.engine.models.case import (
    CaseDocument,
    CaseInput,
    CaseType,
    ClientRole,
    PageContent,
)
from lexflow_agent.engine.nodes import facts as facts_node
from lexflow_agent.engine.nodes.facts import FactListOutput, FactOutput, _validated_sources
from lexflow_agent.engine.state import CaseAgentState


def _state() -> CaseAgentState:
    case_input = CaseInput(
        case_id="case-source",
        case_type=CaseType.LABOR_DISPUTE,
        client_role=ClientRole.EMPLOYEE,
        client_goal="请求赔偿",
        documents=[
            CaseDocument(
                document_id="doc_source",
                file_name="source.txt",
                pages=[PageContent(page=1, text="公司于2025年1月10日解除劳动合同。")],
            )
        ],
    )
    return CaseAgentState(run_id="run-source", case_id="case-source", case_input=case_input)


def test_fact_source_rejects_empty_or_unmatched_quotes():
    sources = _validated_sources(
        _state(),
        [
            {"document_id": "doc_source", "page": 1, "quote": ""},
            {"document_id": "doc_source", "page": 1, "quote": "材料中不存在"},
        ],
    )
    assert sources == []


def test_fact_source_accepts_exact_page_quote():
    sources = _validated_sources(
        _state(),
        [{"document_id": "doc_source", "page": 1, "quote": "2025年1月10日解除劳动合同"}],
    )
    assert len(sources) == 1
    assert sources[0].document_id == "doc_source"


def test_extract_facts_keeps_validated_source(monkeypatch):
    def fake_call(*args, **kwargs):
        return (
            FactListOutput(
                facts=[
                    FactOutput(
                        fact_id="fact_1",
                        statement="公司解除劳动合同",
                        confidence=0.9,
                        sources=[
                            {
                                "document_id": "doc_source",
                                "page": 1,
                                "quote": "2025年1月10日解除劳动合同",
                            }
                        ],
                    )
                ]
            ),
            False,
            [],
            None,
        )

    monkeypatch.setattr(facts_node, "llm_structured_call", fake_call)

    result = facts_node.extract_facts_node(_state())

    assert result["facts"][0].sources[0].quote == "2025年1月10日解除劳动合同"
