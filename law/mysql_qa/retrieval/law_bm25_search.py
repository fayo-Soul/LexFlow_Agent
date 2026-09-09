#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
法律问答对的 BM25 检索模块
基于 law_qa 表，Redis key 与原有教育 QA 系统隔离
"""

import numpy as np
from rank_bm25 import BM25Okapi
from mysql_qa.utils.preprocess import preprocess_text
from mysql_qa.db.law_mysql_client import LawMySQLClient
from mysql_qa.cache.redis_client import RedisClient
from base import setup_logger


class LawBM25Search:
    def __init__(self, redis_client, mysql_client):
        self.logger = setup_logger("LawBM25Search")
        self.redis_client = redis_client
        self.mysql_client = mysql_client
        self.bm25 = None
        self.questions = None
        self.original_questions = None
        self._load_data()

    def _load_data(self):
        # 使用与原有教育 QA 不同的 Redis key，避免数据冲突
        original_key = "law_qa_original_questions"
        tokenized_key = "law_qa_tokenized_questions"

        self.original_questions = self.redis_client.get_data(original_key)
        tokenized_questions = self.redis_client.get_data(tokenized_key)

        if not self.original_questions or not tokenized_questions:
            self.original_questions = self.mysql_client.fetch_questions()
            if not self.original_questions:
                self.logger.warning("未加载到法律问答问题")
                return
            tokenized_questions = [preprocess_text(q[0]) for q in self.original_questions]
            self.redis_client.set_data(original_key, [q[0] for q in self.original_questions])
            self.redis_client.set_data(tokenized_key, tokenized_questions)

        self.questions = tokenized_questions
        self.bm25 = BM25Okapi(self.questions)
        self.logger.info("法律问答 BM25 模型初始化完成")

    def _softmax(self, scores):
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / np.sum(exp_scores)

    def search(self, query, threshold=0.85):
        if not query or not isinstance(query, str):
            self.logger.error("无效查询")
            return None, False

        # 优先查答案缓存（使用 law_answer 前缀，避免与教育 QA 冲突）
        cached_answer = self.redis_client.get_data(f"law_answer:{query}")
        if cached_answer:
            self.logger.info(f"从 Redis 中获取答案：{cached_answer}")
            return cached_answer, False

        try:
            query_tokens = preprocess_text(query)
            scores = self.bm25.get_scores(query_tokens)
            softmax_scores = self._softmax(scores)
            best_idx = softmax_scores.argmax()
            best_score = softmax_scores[best_idx]

            if best_score >= threshold:
                original_question = self.original_questions[best_idx]
                redis_result = self.redis_client.get_data(f"law_answer:{original_question}")
                if redis_result:
                    answer = redis_result
                    self.logger.info(f"从 Redis 中获取答案：{answer}")
                else:
                    answer = self.mysql_client.fetch_answer(original_question)
                    self.logger.info(f"从 MySQL 中获取答案：{answer}")

                if answer:
                    # 使用 law_answer 前缀缓存，避免与教育 QA 冲突
                    self.redis_client.set_data(f"law_answer:{query}", answer)
                    self.logger.info(f"保存答案到 Redis：{query}")
                    self.logger.info(f"搜索成功，最高分数：{best_score:.3f}")
                    return answer, False

            self.logger.info(f"未找到可靠答案，最高分数：{best_score:.3f}")
            return None, True

        except Exception as e:
            self.logger.error(f"搜索失败：{e}")
            return None, True
