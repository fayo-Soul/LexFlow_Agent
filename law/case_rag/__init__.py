#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
case_rag: 法律案例检索模块
从 RAG/ 目录适配而来，用于在项目根目录下无冲突地导入。
"""

from case_rag.searcher import LegalSearcher

__all__ = ["LegalSearcher"]
