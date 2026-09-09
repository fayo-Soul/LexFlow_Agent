#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
法律问答系统入口
先查 MySQL + Redis（结构化问答对），未命中则返回信号让上层走 RAG
"""

import time
from base import setup_logger
from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.db.law_mysql_client import LawMySQLClient
from mysql_qa.retrieval.law_bm25_search import LawBM25Search


class LawQASystem:
    def __init__(self):
        self.logger = setup_logger("LawQASystem")
        self.redis_client = RedisClient()
        self.mysql_client = LawMySQLClient()
        self.bm25_model = LawBM25Search(self.redis_client, self.mysql_client)
        self.logger.info("法律问答系统初始化成功")

    def query(self, query):
        """
        返回：
            - (answer, False)：命中 MySQL/Redis 问答对，直接返回答案
            - (None, True)：未命中，需要上层调用 RAG
        """
        start_time = time.time()
        self.logger.info(f"开始处理法律问题：{query}")
        answer, need_rag = self.bm25_model.search(query)
        if answer:
            self.logger.info(f"问题：{query}，命中答案")
        else:
            self.logger.info(f"问题：{query}，未命中，需要调用 RAG 系统")
        process_time = time.time() - start_time
        self.logger.info(f"处理完成，耗时：{process_time}秒")
        return answer, need_rag


if __name__ == '__main__':
    law_qa_system = LawQASystem()
    answer, need_rag = law_qa_system.query("离婚之后孩子的抚养权怎么判")
    if answer:
        print(f"命中答案：{answer}")
    else:
        print("未命中，需要走 RAG")
