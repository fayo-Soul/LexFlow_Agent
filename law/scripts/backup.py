#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据备份脚本

用法：
    python scripts/backup.py                  # 全量备份
    python scripts/backup.py --type mysql     # 仅备份 MySQL
    python scripts/backup.py --type data      # 仅备份原始数据文件
    python scripts/backup.py --output ./backups  # 指定输出目录

备份内容：
    - MySQL 问答库（mysqldump）
    - 原始数据文件（法律条文、案例、问答对）
    - 模型清单（不备份模型文件本身，仅记录路径和版本）
"""

import argparse
import datetime
import os
import shutil
import subprocess
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base.config import Config


def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def backup_mysql(output_dir):
    """使用 mysqldump 备份 MySQL 问答库。"""
    cfg = Config()
    timestamp = get_timestamp()
    filename = f"mysql_law_qa_{timestamp}.sql"
    filepath = os.path.join(output_dir, filename)

    print(f"[备份] MySQL 问答库 -> {filepath}")

    # 构建 mysqldump 命令
    cmd = [
        "mysqldump",
        f"--host={cfg.MYSQL_HOST}",
        f"--port={cfg.MYSQL_PORT}",
        f"--user={cfg.MYSQL_USER}",
        f"--password={cfg.MYSQL_PASSWORD}",
        cfg.MYSQL_DATABASE,
    ]

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, timeout=120)
        if result.returncode == 0:
            size = os.path.getsize(filepath)
            print(f"  [OK] MySQL 备份完成 ({size / 1024:.1f} KB)")
            return filename
        else:
            print(f"  [ERROR] mysqldump 失败: {result.stderr.decode('utf-8', errors='replace')}")
            return None
    except FileNotFoundError:
        print("  [WARN] 未找到 mysqldump 命令，跳过 MySQL 备份")
        return None
    except subprocess.TimeoutExpired:
        print("  [ERROR] mysqldump 超时 (120s)")
        return None


def backup_data_files(output_dir):
    """备份原始数据文件（法律条文、案例、问答对）。"""
    project_root = os.path.join(os.path.dirname(__file__), "..")
    timestamp = get_timestamp()
    backup_dir = os.path.join(output_dir, f"data_files_{timestamp}")
    os.makedirs(backup_dir, exist_ok=True)

    print(f"[备份] 原始数据文件 -> {backup_dir}")

    # 需要备份的目录和文件（src -> dst 相对路径）
    items = [
        ("法律条文", "法律条文"),
        ("case_rag/data/raw", "case_rag/data/raw"),
    ]
    files = [
        ("法律问答.txt", "法律问答.txt"),
        ("law_qa_pairs.jsonl", "law_qa_pairs.jsonl"),
        (".env.example", ".env.example"),
        ("config.ini", "config.ini"),
    ]

    copied = 0
    for name, rel_path in items:
        src = os.path.join(project_root, rel_path)
        dst = os.path.join(backup_dir, rel_path)
        if os.path.exists(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"  [OK] 目录 {name}")
            copied += 1
        else:
            print(f"  [WARN] 目录不存在，跳过: {name}")

    for name, rel_path in files:
        src = os.path.join(project_root, rel_path)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(backup_dir, rel_path))
            print(f"  [OK] 文件 {name}")
            copied += 1
        else:
            print(f"  [WARN] 文件不存在，跳过: {name}")

    # 复制模型清单
    manifest_src = os.path.join(project_root, "models_manifest.json")
    if os.path.exists(manifest_src):
        shutil.copy2(manifest_src, os.path.join(backup_dir, "models_manifest.json"))
        print(f"  [OK] 模型清单 models_manifest.json")

    print(f"  [完成] 共备份 {copied} 项")
    return os.path.basename(backup_dir)


def generate_backup_report(output_dir, mysql_file, data_dir):
    """生成备份报告。"""
    report = {
        "backup_time": datetime.datetime.now().isoformat(),
        "mysql_backup": mysql_file,
        "data_backup": data_dir,
        "models_manifest": "models_manifest.json（仅记录路径，模型文件需单独管理）",
    }
    report_path = os.path.join(output_dir, "backup_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[报告] 备份清单 -> {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="数据备份工具")
    parser.add_argument("--type", choices=["all", "mysql", "data"], default="all",
                        help="备份类型：all（默认）、mysql、data")
    parser.add_argument("--output", default=None,
                        help="备份输出目录，默认 ./backups/")
    args = parser.parse_args()

    project_root = os.path.join(os.path.dirname(__file__), "..")
    output_dir = args.output or os.path.join(project_root, "backups")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 50)
    print("  法律咨询 RAG 系统 — 数据备份")
    print(f"  输出目录: {output_dir}")
    print("=" * 50)

    mysql_file = None
    data_dir = None

    if args.type in ("all", "mysql"):
        print()
        mysql_file = backup_mysql(output_dir)

    if args.type in ("all", "data"):
        print()
        data_dir = backup_data_files(output_dir)

    print()
    report_path = generate_backup_report(output_dir, mysql_file, data_dir)

    print()
    print("=" * 50)
    print("  备份完成")
    print(f"  报告: {report_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
