#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依赖服务检查脚本

检查所有外部依赖是否可用，在启动服务前运行：

    python scripts/check_services.py

返回码：
    0 = 全部正常
    1 = 存在警告（非致命）
    2 = 存在错误（无法启动）

"""

import os
import sys

# 确保能找到项目根目录
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
CHECK_RESULTS = []
HAS_ERROR = False
HAS_WARNING = False


def check_ok(name, detail=""):
    CHECK_RESULTS.append(("OK", name, detail))
    print(f"  [OK]      {name}" + (f"  — {detail}" if detail else ""))


def check_warn(name, detail=""):
    global HAS_WARNING
    HAS_WARNING = True
    CHECK_RESULTS.append(("WARN", name, detail))
    print(f"  [WARNING] {name}  — {detail}")


def check_error(name, detail=""):
    global HAS_ERROR
    HAS_ERROR = True
    CHECK_RESULTS.append(("ERROR", name, detail))
    print(f"  [ERROR]   {name}  — {detail}")


# ---------------------------------------------------------------------------
# 检查项
# ---------------------------------------------------------------------------
def check_config():
    """检查配置文件和环境变量。"""
    from base.config import Config
    cfg = Config()

    # API Key
    if cfg.DASHSCOPE_API_KEY and cfg.DASHSCOPE_API_KEY != "change_me":
        check_ok("DashScope API Key", "已配置")
    else:
        check_warn("DashScope API Key", "未配置或为占位值 (change_me)")

    # 客服电话
    if cfg.CUSTOMER_SERVICE_PHONE:
        check_ok("客服电话", cfg.CUSTOMER_SERVICE_PHONE)
    else:
        check_warn("客服电话", "未配置")


def check_mysql():
    """检查 MySQL 连接。"""
    from base.config import Config
    cfg = Config()

    try:
        import pymysql
        conn = pymysql.connect(
            host=cfg.MYSQL_HOST,
            port=int(cfg.MYSQL_PORT),
            user=cfg.MYSQL_USER,
            password=cfg.MYSQL_PASSWORD,
            database=cfg.MYSQL_DATABASE,
            connect_timeout=5,
        )
        conn.ping()
        conn.close()
        check_ok("MySQL", f"{cfg.MYSQL_USER}@{cfg.MYSQL_HOST}:{cfg.MYSQL_PORT}/{cfg.MYSQL_DATABASE}")
    except Exception as e:
        check_warn("MySQL", f"无法连接: {e}")


def check_redis():
    """检查 Redis 连接。"""
    from base.config import Config
    cfg = Config()

    try:
        import redis
        r = redis.Redis(
            host=cfg.REDIS_HOST,
            port=int(cfg.REDIS_PORT),
            password=cfg.REDIS_PASSWORD or None,
            db=int(cfg.REDIS_DB),
            socket_connect_timeout=3,
        )
        r.ping()
        r.close()
        check_ok("Redis", f"{cfg.REDIS_HOST}:{cfg.REDIS_PORT}/{cfg.REDIS_DB}")
    except Exception as e:
        check_warn("Redis", f"无法连接: {e}")


def check_milvus():
    """检查 Milvus 连接。"""
    from base.config import Config
    cfg = Config()

    try:
        from pymilvus import connections
        connections.connect(
            alias="default",
            host=cfg.MILVUS_HOST,
            port=cfg.MILVUS_PORT,
        )
        connections.disconnect("default")
        check_ok("Milvus", f"{cfg.MILVUS_HOST}:{cfg.MILVUS_PORT}")
    except Exception as e:
        check_warn("Milvus", f"无法连接: {e}")


def check_models():
    """检查模型文件是否存在。"""
    from base.config import Config
    _cfg = Config()
    model_dirs = [
        ("BERT 法律分类器", _cfg.BERT_CLASSIFIER_DIR),
        ("BGE-M3 嵌入", _cfg.BGE_M3_DIR),
        ("BGE-Reranker", _cfg.BGE_RERANKER_DIR),
        ("文档分割模型", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      "rag_qa", "models", "nlp_bert_document-segmentation_chinese-base")),
    ]

    project_root = os.path.join(os.path.dirname(__file__), "..")
    for name, path in model_dirs:
        full_path = path if os.path.isabs(path) else os.path.join(project_root, path)
        # 检查关键文件
        key_files = ["config.json", "tokenizer.json"]
        exists = all(os.path.exists(os.path.join(full_path, f)) for f in key_files)
        if exists:
            check_ok(f"模型 {name}", path)
        else:
            check_warn(f"模型 {name}", f"{path} — 部分文件缺失，可能需要先下载")


def check_data_files():
    """检查关键数据文件。"""
    root = os.path.join(os.path.dirname(__file__), "..")

    files = [
        ("法律问答对", "法律问答.txt"),
        ("法律条文目录", "法律条文"),
    ]

    for name, path in files:
        full_path = os.path.join(root, path)
        if os.path.exists(full_path):
            if os.path.isfile(full_path):
                size = os.path.getsize(full_path)
                check_ok(f"数据 {name}", f"{path} ({size / 1024:.0f} KB)")
            else:
                items = os.listdir(full_path)
                check_ok(f"数据 {name}", f"{path} ({len(items)} 项)")
        else:
            check_warn(f"数据 {name}", f"{path} — 未找到")


def check_python_packages():
    """检查关键 Python 包是否已安装。"""
    required = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "pymysql",
        "pymilvus",
    ]

    for pkg in required:
        try:
            __import__(pkg)
            check_ok(f"Python 包 {pkg}")
        except ImportError:
            check_error(f"Python 包 {pkg}", f"{pkg} 未安装，请执行 pip install -r requirements.txt")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  法律咨询 RAG 系统 — 依赖检查")
    print("=" * 60)
    print()

    print(">> 环境变量")
    check_config()

    print()
    print(">> Python 依赖")
    check_python_packages()

    print()
    print(">> 数据文件")
    check_data_files()

    print()
    print(">> 模型文件")
    check_models()

    print()
    print(">> 外部服务连接")
    check_mysql()
    check_redis()
    check_milvus()

    print()
    print("-" * 60)

    ok_count = sum(1 for r in CHECK_RESULTS if r[0] == "OK")
    warn_count = sum(1 for r in CHECK_RESULTS if r[0] == "WARN")
    error_count = sum(1 for r in CHECK_RESULTS if r[0] == "ERROR")

    print(f"  结果: {ok_count} 通过, {warn_count} 警告, {error_count} 错误")

    if HAS_ERROR:
        print()
        print("  [ERROR] 存在错误，请修复后重试")
        sys.exit(2)
    elif HAS_WARNING:
        print()
        print("  [WARN] 存在警告，服务可启动但部分功能受限")
        sys.exit(1)
    else:
        print()
        print("  [OK] 全部检查通过")
        sys.exit(0)


if __name__ == "__main__":
    main()
