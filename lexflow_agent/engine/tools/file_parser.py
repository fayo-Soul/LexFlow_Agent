"""文件解析工具 — 将用户上传的原始文件转换为分页纯文本

支持格式:
  - PDF / PNG / JPG → MinerU 解析（子进程调用，输出 Markdown）
  - DOCX           → python-docx 解析（段落 + 表格 + 嵌入图片 OCR）
  - TXT / MD       → 直接读取

架构:
  1. 文件类型检测 → 选择解析策略
  2. 解析为完整文本 → 按段落/换行分页
  3. 输出为 CaseDocument.pages 格式，直接注入 Agent Pipeline

MinerU 说明:
  - 需要独立 Python 环境（conda/miniforge），避免依赖冲突
  - 安装: conda create -n mineru python=3.10 && conda activate mineru &&
          pip install -U "mineru[all]" --extra-index-url https://wheels.myhloli.com -i https://mirrors.aliyun.com/pypi/simple
  - 首次运行需下载模型权重文件
  - 默认输出为 Markdown，保留标题层级、表格、公式
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import BinaryIO, Optional

from lexflow_agent.config.settings import settings
from lexflow_agent.engine.models.case import CaseDocument, PageContent


# ── 配置 ──────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".png", ".jpg", ".jpeg"}
MAX_FILE_SIZE_MB = 50          # 单文件最大体积
MAX_PAGES_PER_DOC = 200        # 单文档最大页数
PAGE_MAX_CHARS = 4000          # 单页最大字符数，超出自动分页


class FileParserError(ValueError):
    """文件解析异常"""
    pass


# ── 工具函数 ──────────────────────────────────────────────

def _generate_document_id(file_name: str, content: bytes) -> str:
    """生成文档 ID: doc_ 前缀 + 文件名哈希"""
    digest = hashlib.sha256(content).hexdigest()[:16]
    safe_name = Path(file_name).stem.replace(" ", "_")[:30]
    return f"doc_{safe_name}_{digest}"


def _validate_file(file_extension: str, content: bytes) -> None:
    """文件合法性校验"""
    if file_extension not in SUPPORTED_EXTENSIONS:
        raise FileParserError(
            f"不支持的文件格式: {file_extension}。"
            f"支持: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise FileParserError(
            f"文件过大: {size_mb:.1f}MB（上限 {MAX_FILE_SIZE_MB}MB）"
        )


def _validate_file_signature(extension: str, content: bytes) -> None:
    """文件头魔数校验，防止扩展名伪装"""
    signatures = {
        ".pdf": b"%PDF",
        ".png": b"\x89PNG",
        ".jpg": b"\xff\xd8\xff",
        ".jpeg": b"\xff\xd8\xff",
        ".docx": b"PK\x03\x04",
        ".doc": b"\xd0\xcf\x11\xe0",   # OLE2 复合文档
    }
    expected = signatures.get(extension)
    if expected and not content.startswith(expected):
        raise FileParserError(
            f"文件头与扩展名不匹配: 扩展名 {extension}，但文件内容不是有效的对应格式"
        )


def _text_to_pages(text: str, max_chars: int = PAGE_MAX_CHARS) -> list[PageContent]:
    """将文本切分为分页结构

    策略:
      1. 先按双换行（段落）切分段落
      2. 逐段拼接，超出 max_chars 时另起一页
      3. 确保不截断段落
    """
    if not text.strip():
        return [PageContent(page=1, text="（文档无文本内容）")]

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    pages = []
    current_page_texts: list[str] = []
    current_length = 0

    for para in paragraphs:
        para_len = len(para)
        if current_length + para_len > max_chars and current_page_texts:
            pages.append(PageContent(
                page=len(pages) + 1,
                text="\n\n".join(current_page_texts)
            ))
            current_page_texts = [para]
            current_length = para_len
        else:
            current_page_texts.append(para)
            current_length += para_len

    if current_page_texts:
        pages.append(PageContent(
            page=len(pages) + 1,
            text="\n\n".join(current_page_texts)
        ))

    if len(pages) > MAX_PAGES_PER_DOC:
        pages = pages[:MAX_PAGES_PER_DOC]
        pages.append(PageContent(
            page=MAX_PAGES_PER_DOC + 1,
            text=f"（文档超过 {MAX_PAGES_PER_DOC} 页上限，后续内容已截断）"
        ))

    return pages


# ── 各格式解析器 ──────────────────────────────────────────

def _parse_with_mineru(file_path: Path) -> str:
    """通过 MinerU 子进程解析 PDF/图片，输出 Markdown 文本

    MinerU 需要独立的 conda 环境，进程中执行:
      conda run -n mineru mineru -p <input> -o <output_dir> -b pipeline

    返回解析后的 Markdown 全文。
    """
    mineru_env = settings.MINERU_CONDA_ENV or "mineru"
    mineru_backend = settings.MINERU_BACKEND or "pipeline"

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "mineru_output"
        output_dir.mkdir()

        cmd = [
            "conda", "run", "-n", mineru_env, "--no-capture-output",
            "mineru",
            "-p", str(file_path),
            "-o", str(output_dir),
            "-b", mineru_backend,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=getattr(settings, "MINERU_TIMEOUT", 300),
            )

            if result.returncode != 0:
                stderr_tail = result.stderr[-500:] if result.stderr else ""
                raise FileParserError(
                    f"MinerU 解析失败 (返回码 {result.returncode}): {stderr_tail}"
                )

            # MinerU 输出目录结构: output_dir/<filename>/auto.md 或 .json
            # 优先读 Markdown
            md_files = list(output_dir.rglob("*.md"))
            if md_files:
                return md_files[0].read_text(encoding="utf-8")

            # 回退读 JSON
            json_files = list(output_dir.rglob("*.json"))
            if json_files:
                import json
                data = json.loads(json_files[0].read_text(encoding="utf-8"))
                return _mineru_json_to_text(data)

            raise FileParserError("MinerU 未生成可读输出文件")

        except subprocess.TimeoutExpired:
            raise FileParserError(
                f"MinerU 解析超时"
                f"（{getattr(settings, 'MINERU_TIMEOUT', 300)}s），"
                f"文件可能过大或过于复杂"
            )
        except FileNotFoundError:
            raise FileParserError(
                f"未找到 MinerU 环境。请先创建 conda 环境: "
                f"conda create -n {mineru_env} python=3.10 && "
                f"conda activate {mineru_env} && "
                f"pip install -U 'mineru[all]' "
                f"--extra-index-url https://wheels.myhloli.com "
                f"-i https://mirrors.aliyun.com/pypi/simple"
            )


def _mineru_json_to_text(data: dict) -> str:
    """MinerU JSON 输出转为纯文本"""
    parts = []
    for block in data.get("blocks", []):
        block_type = block.get("type", "")
        if block_type == "text":
            parts.append(block.get("content", ""))
        elif block_type == "table":
            parts.append(_table_to_text(block.get("content", [])))
        elif block_type == "title":
            parts.append(f"## {block.get('content', '')}")
    return "\n\n".join(parts)


def _table_to_text(table_data: list) -> str:
    """MinerU 表格数据 → Markdown 表格文本"""
    if not table_data:
        return ""
    lines = []
    for row in table_data:
        if isinstance(row, list):
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _parse_docx(file_path: Path) -> str:
    """python-docx 解析 DOCX，提取段落 + 表格文本

    不处理嵌入图片（如需 OCR，建议转 PDF 后走 MinerU）。
    """
    try:
        from docx import Document
    except ImportError:
        raise FileParserError(
            "解析 DOCX 需要 python-docx，请执行: pip install python-docx"
        )

    doc = Document(str(file_path))
    parts = []

    for element in doc.element.body:
        tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

        if tag == "p":  # 段落
            from docx.text.paragraph import Paragraph
            para = Paragraph(element, doc)
            text = para.text.strip()
            if text:
                parts.append(text)

        elif tag == "tbl":  # 表格
            from docx.table import Table
            table = Table(element, doc)
            table_texts = []
            for row in table.rows:
                cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                table_texts.append("| " + " | ".join(cells) + " |")
            if table_texts:
                parts.append("\n".join(table_texts))

    return "\n\n".join(parts) if parts else "（文档无文本内容）"


def _parse_text(file_path: Path) -> str:
    """直接读取 TXT / MD 的文本内容"""
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return text if text.strip() else "（文档无文本内容）"


# ── 主入口 ────────────────────────────────────────────────

class FileParser:
    """文件解析工具

    用法:
        parser = FileParser()
        document = parser.parse(file_path, content_bytes)  # 从路径
        document = parser.parse_from_bytes(file_name, content_bytes)  # 从内存
    """

    def __init__(self):
        self._temp_dir: Optional[Path] = None

    # ── 公共方法 ──────────────────────────────────────────

    def parse(self, file_path: str | Path) -> CaseDocument:
        """从文件路径解析"""
        path = Path(file_path)
        if not path.exists():
            raise FileParserError(f"文件不存在: {path}")
        content = path.read_bytes()
        return self._parse(path.name, content)

    def parse_from_bytes(
        self, file_name: str, content: bytes
    ) -> CaseDocument:
        """从内存字节解析（HTTP 上传场景）"""
        return self._parse(file_name, content)

    # ── 内部实现 ──────────────────────────────────────────

    def _parse(self, file_name: str, content: bytes) -> CaseDocument:
        extension = Path(file_name).suffix.lower()

        _validate_file(extension, content)
        _validate_file_signature(extension, content)

        document_id = _generate_document_id(file_name, content)

        # 需要写入临时文件的格式（MinerU 需要文件路径）
        if extension in {".pdf", ".png", ".jpg", ".jpeg"}:
            text = self._parse_via_tempfile(file_name, extension, content)
        elif extension == ".docx":
            text = self._parse_via_tempfile(file_name, extension, content)
        elif extension in {".txt", ".md"}:
            text = content.decode("utf-8", errors="replace")
        else:
            raise FileParserError(f"未处理的文件格式: {extension}")

        pages = _text_to_pages(text)

        return CaseDocument(
            document_id=document_id,
            file_name=file_name,
            pages=pages,
        )

    def _parse_via_tempfile(
        self, file_name: str, extension: str, content: bytes
    ) -> str:
        """将内容写入临时文件后解析"""
        with tempfile.NamedTemporaryFile(
            suffix=extension, delete=False, prefix="lexflow_parse_"
        ) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            if extension in {".pdf", ".png", ".jpg", ".jpeg"}:
                return _parse_with_mineru(tmp_path)
            elif extension == ".docx":
                return _parse_docx(tmp_path)
            else:
                return _parse_text(tmp_path)
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass


# ── 全局单例 ──────────────────────────────────────────────
file_parser = FileParser()
