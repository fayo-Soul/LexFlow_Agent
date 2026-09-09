#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
case_rag 配置
指向根项目下的模型路径和第二个 Milvus 案例库。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from base.config import Config

_cfg = Config()
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 模型路径：复用根项目 rag_qa/models 下的模型（通过 Config 可配置）
MODEL_DIR = os.path.join(PROJECT_ROOT, _cfg.MODEL_DIR) if not os.path.isabs(_cfg.MODEL_DIR) else _cfg.MODEL_DIR
EMBEDDING_MODEL_PATH = _cfg.BGE_M3_DIR
RERANKER_MODEL_PATH = _cfg.BGE_RERANKER_DIR

# Milvus 案例库配置
MILVUS_HOST = _cfg.MILVUS_HOST
MILVUS_PORT = _cfg.MILVUS_PORT
DB_NAME = _cfg.CASE_MILVUS_DATABASE_NAME if hasattr(_cfg, 'CASE_MILVUS_DATABASE_NAME') else "fzt_case"
COLLECTION_NAME = _cfg.CASE_MILVUS_COLLECTION_NAME if hasattr(_cfg, 'CASE_MILVUS_COLLECTION_NAME') else "fzt_case_hybrid"
VECTOR_DIM = 1024
