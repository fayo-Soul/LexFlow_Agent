#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导入案例文书到 Milvus 向量库。

用法：
    python scripts/ingest_cases.py              # 重建集合后导入
    python scripts/ingest_cases.py --no-drop    # 保留已有数据

"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(description="导入案例文书到 Milvus")
    parser.add_argument("--no-drop", action="store_true",
                        help="保留已有数据，不重建向量集合")
    args = parser.parse_args()

    from main import ingest_cases

    drop = not args.no_drop
    print("=" * 50)
    print("  导入案例文书到 Milvus")
    print(f"  模式：{'重建集合' if drop else '保留已有数据'}")
    print("=" * 50)

    total = ingest_cases(drop_existing=drop)
    print(f"\n✅ 案例导入完成，共添加 {total} 个文档块")


if __name__ == "__main__":
    main()
