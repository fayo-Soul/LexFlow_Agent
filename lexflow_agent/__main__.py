#!/usr/bin/env python3
"""LexFlow Agent Engine - 启动入口"""

import sys
import os

# 确保能找到项目根目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from lexflow_agent.config.settings import settings


def main():
    uvicorn.run(
        "lexflow_agent.api.app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )


if __name__ == "__main__":
    main()
