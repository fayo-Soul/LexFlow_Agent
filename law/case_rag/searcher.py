#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
法律案例检索器
从 RAG/rag_qa/searcher.py 适配而来，用于检索 case_law.legal_case_hybrid 中的法律案例。
"""

import torch
from pymilvus import connections, db, Collection, AnnSearchRequest, RRFRanker
from sentence_transformers import SentenceTransformer
from case_rag.config import (
    MILVUS_HOST, MILVUS_PORT, DB_NAME, COLLECTION_NAME,
    EMBEDDING_MODEL_PATH,
)
from case_rag.logger import logger
from case_rag.sparse import stable_token_id


class LegalSearcher:
    def __init__(self):
        logger.info("正在初始化法律案例检索服务...")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL_PATH)

        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        db.using_database(DB_NAME)
        self.collection = Collection(COLLECTION_NAME)
        self.collection.load()
        logger.info("法律案例检索服务初始化完成")

    def _get_query_vectors(self, query_text):
        query_dense = self.embed_model.encode(query_text).tolist()

        lexical_weights = {}
        for word in list(query_text):
            token_id = stable_token_id(word)
            weight_bonus = 5.0 if word in ["强", "奸", "受", "贿", "判", "刑", "彩", "礼"] else 1.0
            lexical_weights[token_id] = lexical_weights.get(token_id, 0.0) + weight_bonus

        total = sum(lexical_weights.values()) if lexical_weights else 1.0
        query_sparse = {int(k): float(v / total) for k, v in lexical_weights.items()}
        return query_dense, query_sparse

    def _build_filter_expr(self, source_filter):
        """根据 source_filter 构建 Milvus 过滤表达式"""
        if not source_filter:
            return ""
        def safe(value):
            value = str(value)
            if not value or any(char in value for char in "\"'[]()=<>"):
                raise ValueError("案例检索过滤条件包含非法字符")
            return value
        if isinstance(source_filter, (list, tuple)):
            quoted = ", ".join([f"'{safe(s)}'" for s in source_filter])
            return f"domain in [{quoted}]"
        return f"domain=='{safe(source_filter)}'"

    def hybrid_search(self, query_text, top_k=15, final_top_n=1, source_filter=None):
        """
        双向量混合检索法律案例。
        返回列表，每个元素包含 title, case_number, child_chunk, fact, ruling, rerank_score。
        source_filter: 按 domain 字段过滤，支持单值字符串或列表。
        """
        if not query_text or not isinstance(query_text, str):
            logger.warning("案例检索查询为空")
            return []

        query_dense, query_sparse = self._get_query_vectors(query_text)
        filter_expr = self._build_filter_expr(source_filter)

        req_dense = AnnSearchRequest(
            [query_dense], "dense_vector", {"metric_type": "COSINE"}, limit=top_k, expr=filter_expr
        )
        req_sparse = AnnSearchRequest(
            [query_sparse], "sparse_vector", {"metric_type": "IP"}, limit=top_k, expr=filter_expr
        )

        res = self.collection.hybrid_search(
            reqs=[req_dense, req_sparse],
            rerank=RRFRanker(),
            limit=top_k,
            output_fields=["title", "case_number", "child_chunk", "fact", "ruling", "domain"]
        )

        if not res:
            return []

        hits = res[0]
        metadata_list = []
        for hit in hits:
            metadata_list.append({
                "title": hit.entity.get("title"),
                "case_number": hit.entity.get("case_number"),
                "child_chunk": hit.entity.get("child_chunk"),
                "fact": hit.entity.get("fact"),
                "ruling": hit.entity.get("ruling"),
                "domain": hit.entity.get("domain")
            })

        if metadata_list:
            q_emb = self.embed_model.encode(query_text, convert_to_tensor=True)
            d_embs = self.embed_model.encode([m["child_chunk"] for m in metadata_list], convert_to_tensor=True)
            scores = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), d_embs, dim=-1).cpu().tolist()
            for idx, score in enumerate(scores):
                metadata_list[idx]["rerank_score"] = float(score)
            metadata_list.sort(key=lambda x: x["rerank_score"], reverse=True)

        return metadata_list[:final_top_n]

    def search_by_articles(self, articles_text, original_query="", top_k=15, final_top_n=2, source_filter=None):
        """
        基于法律条文检索相关案例。
        将法律条文与用户问题拼接作为案例检索输入。
        source_filter: 按 domain 字段过滤案例，通常传入 BERT 分类结果。
        """
        combined_query = articles_text
        if original_query:
            combined_query = f"{original_query}\n\n相关法律条文：\n{articles_text}"
        return self.hybrid_search(combined_query, top_k=top_k, final_top_n=final_top_n, source_filter=source_filter)
