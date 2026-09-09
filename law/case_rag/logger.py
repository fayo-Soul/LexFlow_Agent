#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
case_rag 日志器
"""

import logging

logger = logging.getLogger("CaseRAG")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    )
    logger.addHandler(handler)
