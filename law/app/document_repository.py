"""Document metadata repositories."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

import pymysql


class JsonDocumentRepository:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = Lock()

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"知识库登记文件损坏或不可读：{exc}") from exc

    def save(self, documents: list[dict[str, Any]]) -> None:
        with self._lock:
            temp_path = self.path.with_suffix(".tmp")
            temp_path.write_text(
                json.dumps(documents, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temp_path.replace(self.path)


class MySQLDocumentRepository:
    """MySQL-backed registry with transaction and row locking."""

    def __init__(self, config, table: str = "rag_documents"):
        if not table.replace("_", "").isalnum():
            raise ValueError("非法数据表名称")
        self.config = config
        self.table = table
        self._create_table()

    def _connect(self):
        return pymysql.connect(
            host=self.config.MYSQL_HOST,
            port=int(self.config.MYSQL_PORT),
            user=self.config.MYSQL_USER,
            password=self.config.MYSQL_PASSWORD,
            database=self.config.MYSQL_DATABASE,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=False,
            connect_timeout=5,
        )

    def _create_table(self) -> None:
        sql = f"""
        CREATE TABLE IF NOT EXISTS `{self.table}` (
            id VARCHAR(64) PRIMARY KEY,
            file_hash CHAR(64) NOT NULL UNIQUE,
            document_type VARCHAR(40) NOT NULL,
            processing_status VARCHAR(24) NOT NULL,
            lifecycle_status VARCHAR(24) NOT NULL DEFAULT 'draft',
            access_level VARCHAR(24) NOT NULL DEFAULT 'internal',
            updated_at VARCHAR(40) NOT NULL,
            payload JSON NOT NULL,
            INDEX idx_rag_documents_type_status
                (document_type, lifecycle_status, processing_status)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql)
            connection.commit()

    def load(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT payload FROM `{self.table}` ORDER BY updated_at")
                rows = cursor.fetchall()
        return [
            json.loads(row["payload"])
            if isinstance(row["payload"], str)
            else row["payload"]
            for row in rows
        ]

    def save(self, documents: list[dict[str, Any]]) -> None:
        upsert = f"""
        INSERT INTO `{self.table}`
            (id, file_hash, document_type, processing_status, lifecycle_status,
             access_level, updated_at, payload)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            file_hash=VALUES(file_hash),
            document_type=VALUES(document_type),
            processing_status=VALUES(processing_status),
            lifecycle_status=VALUES(lifecycle_status),
            access_level=VALUES(access_level),
            updated_at=VALUES(updated_at),
            payload=VALUES(payload)
        """
        with self._connect() as connection:
            try:
                with connection.cursor() as cursor:
                    for item in documents:
                        cursor.execute(
                            upsert,
                            (
                                item["id"],
                                item["file_hash"],
                                item["document_type"],
                                item["processing_status"],
                                item.get("lifecycle_status", "draft"),
                                item.get("access_level", "internal"),
                                item["updated_at"],
                                json.dumps(item, ensure_ascii=False),
                            ),
                        )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def delete(self, document_id: str) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM `{self.table}` WHERE id=%s",
                    (document_id,),
                )
            connection.commit()

    def import_missing(self, documents: list[dict[str, Any]]) -> int:
        existing = {item["id"] for item in self.load()}
        missing = [item for item in documents if item["id"] not in existing]
        if not missing:
            return 0
        self.save(self.load() + missing)
        return len(missing)
