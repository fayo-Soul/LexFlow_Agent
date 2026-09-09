#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导入结构化问答对到 MySQL。

用法：
    python scripts/import_qa.py

流程：
    1. 解析 法律问答.txt → law_qa_pairs.jsonl
    2. 导入 law_qa_pairs.jsonl → MySQL law_qa 表

"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import parse_legal_qa, import_qa_to_mysql


def main():
    print("=" * 50)
    print("  导入结构化问答对")
    print("=" * 50)

    # 第一步：解析
    print("\n>> 第 1 步：解析法律问答.txt → jsonl")
    jsonl_path = parse_legal_qa()
    print(f"  输出文件：{jsonl_path}")

    # 第二步：导入
    print("\n>> 第 2 步：导入 jsonl → MySQL")
    import_qa_to_mysql(jsonl_path)

    print("\n✅ 问答对导入完成")


if __name__ == "__main__":
    main()
