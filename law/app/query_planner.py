"""分层查询路由：确定性规则、BERT 领域分类、低置信度 LLM 降级。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Iterable

from app.intent import route_query


PUBLIC_SCOPE = ("regulation", "judicial_interpretation")
SUPPORTED_SCOPE = {"regulation", "judicial_interpretation", "case", "internal"}
CASE_PHRASES = ("案例", "判例", "裁判观点", "类似裁判", "相似案件")


@dataclass(frozen=True)
class QueryPlan:
    intent: str
    scope: tuple[str, ...] = PUBLIC_SCOPE
    legal_domains: tuple[str, ...] = ()
    task_type: str = "legal_analysis"
    confidence: float = 1.0
    needs_clarification: bool = False
    answer: str = ""
    route_source: str = "rule"


class QueryPlanner:
    def __init__(
        self,
        classifier,
        llm: Callable[[str], str] | None,
        bert_confidence_threshold: float = 0.7,
    ):
        self.classifier = classifier
        self.llm = llm
        self.bert_confidence_threshold = bert_confidence_threshold

    def plan(
        self,
        query: str,
        explicit_scope: Iterable[str] | None = None,
    ) -> QueryPlan:
        route = route_query(query)
        if route.intent != "legal_research":
            return QueryPlan(
                intent=route.intent,
                scope=(),
                answer=route.answer,
                route_source="rule",
            )

        if explicit_scope is not None:
            scope = self._normalize_scope(explicit_scope)
            return QueryPlan(
                intent="legal_research",
                scope=scope or PUBLIC_SCOPE,
                route_source="explicit",
            )

        rule_plan = self._legal_rule_plan(query)
        if rule_plan:
            return rule_plan

        domain, confidence = self._bert_result(query)
        if domain and confidence >= self.bert_confidence_threshold:
            return QueryPlan(
                intent="legal_research",
                scope=PUBLIC_SCOPE,
                legal_domains=(domain,),
                confidence=confidence,
                route_source="bert",
            )

        llm_plan = self._llm_plan(query)
        if llm_plan:
            return llm_plan

        return QueryPlan(
            intent="legal_research",
            scope=PUBLIC_SCOPE,
            confidence=confidence,
            route_source="fallback",
        )

    @staticmethod
    def _normalize_scope(scope: Iterable[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item for item in scope if item in SUPPORTED_SCOPE))

    @staticmethod
    def _legal_rule_plan(query: str) -> QueryPlan | None:
        wants_case = any(phrase in query for phrase in CASE_PHRASES)
        if not wants_case:
            return None
        return QueryPlan(
            intent="legal_research",
            scope=("case",),
            task_type="similar_case_search",
            route_source="rule",
        )

    def _bert_result(self, query: str) -> tuple[str | None, float]:
        try:
            if hasattr(self.classifier, "predict_with_confidence"):
                return self.classifier.predict_with_confidence(query)
            return self.classifier.predict(query), 1.0
        except Exception:
            return None, 0.0

    def _llm_plan(self, query: str) -> QueryPlan | None:
        if self.llm is None:
            return None
        prompt = f"""
你是法律知识库的查询路由器，只做分类，不回答问题。
请只输出一个 JSON 对象，不要输出解释。字段要求：
- intent: legal_research 或 out_of_scope
- legal_domains: 法律领域字符串数组
- requested_sources: regulation、judicial_interpretation、case、internal 的数组
- task_type: legal_analysis、similar_case_search 或 internal_research
- confidence: 0 到 1
- needs_clarification: 布尔值

用户问题：{query}
"""
        try:
            raw = self.llm(prompt)
            match = re.search(r"\{[\s\S]*\}", raw or "")
            if not match:
                return None
            data = json.loads(match.group(0))
            intent = data.get("intent")
            if intent not in {"legal_research", "out_of_scope"}:
                return None
            if intent == "out_of_scope":
                return QueryPlan(
                    intent=intent,
                    scope=(),
                    confidence=self._confidence(data.get("confidence")),
                    answer="该问题不属于本系统的法律研究范围。本系统主要提供法规、司法解释、案例和律所内部资料检索。",
                    route_source="llm",
                )
            # 内部资料必须由调用方显式选择，以确保 API 能在规划前完成鉴权。
            scope = tuple(
                item
                for item in self._normalize_scope(data.get("requested_sources") or [])
                if item != "internal"
            )
            domains = tuple(
                str(item) for item in data.get("legal_domains") or [] if str(item)
            )
            return QueryPlan(
                intent=intent,
                scope=scope or PUBLIC_SCOPE,
                legal_domains=domains,
                task_type=str(data.get("task_type") or "legal_analysis"),
                confidence=self._confidence(data.get("confidence")),
                needs_clarification=bool(data.get("needs_clarification", False)),
                route_source="llm",
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _confidence(value) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0
