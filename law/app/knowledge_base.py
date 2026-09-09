"""本地文档登记与知识库文件生命周期管理。"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, BinaryIO

from app.document_repository import JsonDocumentRepository


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
DOCUMENT_TYPES = {"regulation", "judicial_interpretation", "case", "internal"}


class KnowledgeBaseError(ValueError):
    pass


class KnowledgeBaseManager:
    def __init__(
        self,
        storage_dir: str | Path = "data/knowledge_base",
        max_file_size: int = 20 * 1024 * 1024,
        repository=None,
    ):
        self.storage_dir = Path(storage_dir)
        self.files_dir = self.storage_dir / "files"
        self.registry_path = self.storage_dir / "documents.json"
        self.repository = repository or JsonDocumentRepository(self.registry_path)
        self.max_file_size = max_file_size
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def _load(self) -> list[dict[str, Any]]:
        try:
            return self.repository.load()
        except (ValueError, OSError) as exc:
            raise KnowledgeBaseError(f"知识库登记文件损坏或不可读：{exc}") from exc

    def _save(self, documents: list[dict[str, Any]]) -> None:
        self.repository.save(documents)

    @staticmethod
    def _digest(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def _validate_content(extension: str, content: bytes) -> None:
        if extension == ".pdf" and not content.startswith(b"%PDF"):
            raise KnowledgeBaseError("文件内容不是有效的 PDF")
        if extension == ".docx" and not content.startswith(b"PK"):
            raise KnowledgeBaseError("文件内容不是有效的 DOCX")
        if extension in {".txt", ".md"}:
            try:
                content.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise KnowledgeBaseError("文本文件必须使用 UTF-8 编码") from exc

    def create(
        self,
        file_name: str,
        file_object: BinaryIO,
        document_type: str,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        extension = Path(file_name).suffix.lower()
        safe_name = Path(file_name).name
        if safe_name != file_name or safe_name in {"", ".", ".."}:
            raise KnowledgeBaseError("文件名包含非法路径")
        if extension not in SUPPORTED_EXTENSIONS:
            raise KnowledgeBaseError(f"不支持的文件格式：{extension or '无扩展名'}")
        if document_type not in DOCUMENT_TYPES:
            raise KnowledgeBaseError(f"不支持的文档类型：{document_type}")

        content = file_object.read()
        if not content:
            raise KnowledgeBaseError("文件内容不能为空")
        if len(content) > self.max_file_size:
            raise KnowledgeBaseError(
                f"文件大小超过限制：最大 {self.max_file_size // (1024 * 1024)} MB"
            )
        self._validate_content(extension, content)
        digest = self._digest(content)

        with self._lock:
            documents = self._load()
            duplicate = next((item for item in documents if item["file_hash"] == digest), None)
            if duplicate:
                raise KnowledgeBaseError(f"文件已存在，document_id={duplicate['id']}")

            document_id = uuid.uuid4().hex
            stored_name = f"{document_id}{extension}"
            file_path = self.files_dir / stored_name
            file_path.write_bytes(content)
            now = datetime.now(timezone.utc).isoformat()
            item = {
                "id": document_id,
                "title": title or Path(file_name).stem,
                "document_type": document_type,
                "file_name": safe_name,
                "file_hash": digest,
                "file_path": str(file_path.resolve()),
                "processing_status": "uploaded",
                "lifecycle_status": "draft",
                "chunk_count": 0,
                "version": str((metadata or {}).get("version", "1.0")),
                "effective_status": str((metadata or {}).get("effective_status", "unknown")),
                "access_level": str((metadata or {}).get("access_level", "internal")),
                "metadata": metadata or {},
                "created_at": now,
                "updated_at": now,
                "error": "",
            }
            documents.append(item)
            self._save(documents)
            return item

    def list(self, document_type: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        documents = self._load()
        if document_type:
            documents = [item for item in documents if item["document_type"] == document_type]
        if status:
            documents = [item for item in documents if item["processing_status"] == status]
        return documents

    def get(self, document_id: str) -> dict[str, Any] | None:
        return next((item for item in self._load() if item["id"] == document_id), None)

    def update_status(
        self,
        document_id: str,
        status: str,
        chunk_count: int | None = None,
        error: str = "",
    ) -> dict[str, Any]:
        with self._lock:
            documents = self._load()
            item = next((doc for doc in documents if doc["id"] == document_id), None)
            if item is None:
                raise KnowledgeBaseError("文档不存在")
            item["processing_status"] = status
            item["error"] = error
            item["updated_at"] = datetime.now(timezone.utc).isoformat()
            if chunk_count is not None:
                item["chunk_count"] = chunk_count
            self._save(documents)
            return item

    def update_lifecycle(
        self,
        document_id: str,
        lifecycle_status: str,
        actor: str,
    ) -> dict[str, Any]:
        allowed = {"draft", "pending_review", "published", "retired"}
        if lifecycle_status not in allowed:
            raise KnowledgeBaseError("非法的文档生命周期状态")
        with self._lock:
            documents = self._load()
            item = next((doc for doc in documents if doc["id"] == document_id), None)
            if item is None:
                raise KnowledgeBaseError("文档不存在")
            item["lifecycle_status"] = lifecycle_status
            item["reviewed_by"] = actor if lifecycle_status == "published" else item.get("reviewed_by", "")
            item["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._save(documents)
            return item

    def delete(self, document_id: str) -> bool:
        with self._lock:
            documents = self._load()
            item = next((doc for doc in documents if doc["id"] == document_id), None)
            if item is None:
                return False
            path = Path(item["file_path"])
            if path.exists():
                path.unlink()
            documents = [doc for doc in documents if doc["id"] != document_id]
            if hasattr(self.repository, "delete"):
                self.repository.delete(document_id)
            else:
                self._save(documents)
            return True

    def staging_directory(self, document_id: str) -> Path:
        item = self.get(document_id)
        if item is None:
            raise KnowledgeBaseError("文档不存在")
        stage = self.storage_dir / "staging" / document_id
        stage.mkdir(parents=True, exist_ok=True)
        target = stage / item["file_name"]
        shutil.copy2(item["file_path"], target)
        return stage

    def cleanup_staging(self, document_id: str) -> None:
        stage = (self.storage_dir / "staging" / document_id).resolve()
        staging_root = (self.storage_dir / "staging").resolve()
        if stage.parent == staging_root and stage.exists():
            shutil.rmtree(stage)
