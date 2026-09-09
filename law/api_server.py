#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
法律问答系统 Web API 服务

启动方式：
    python api_server.py
    或
    uvicorn api_server:app --host 0.0.0.0 --port 8000

接口：
    POST /api/v1/chat      聊天接口（版本化）
    GET  /api/v1/health    健康检查
    GET  /api/v1/ready     就绪检查
    POST /api/chat         兼容旧路由
    GET  /api/health       兼容旧路由
    GET  /                 前端聊天页面
"""

import os
import sys
import uuid
import logging
import io
import socket
import time
import ipaddress
from urllib.parse import urlparse
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from threading import Lock
from typing import Literal

import uvicorn
from fastapi import FastAPI, Request, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

# 确保能找到项目根目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base import Config, setup_logger
from base.session_manager import get_session_manager
from app.knowledge_base import KnowledgeBaseError, KnowledgeBaseManager
from app.auth import Principal, authenticate_bearer
from app.audit import AuditLogger
from app.document_repository import JsonDocumentRepository, MySQLDocumentRepository
from app.rate_limit import DistributedRateLimiter
from app.task_queue import RedisTaskQueue

# ---------------------------------------------------------------------------
# 全局变量
# ---------------------------------------------------------------------------
assistant = None
logger = setup_logger("api_server")
_cfg = Config()
API_TOKEN = _cfg.API_TOKEN
RESEARCHER_API_TOKEN = _cfg.RESEARCHER_API_TOKEN
knowledge_base = KnowledgeBaseManager(
    max_file_size=_cfg.MAX_UPLOAD_MB * 1024 * 1024
)
audit_logger = AuditLogger()
redis_client = None
rate_limiter = None
index_queue = None


# ---------------------------------------------------------------------------
# 请求 / 响应模型
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="用户法律问题（1-2000 字）",
    )
    source_filter: str | None = Field(
        None,
        description="限定检索领域，如「劳动法」；不指定则自动分类",
    )
    session_id: str | None = Field(
        None,
        description="会话 ID；不传则自动创建新会话",
    )


class ReferenceItem(BaseModel):
    type: str = Field(..., description="引用类型: article | case | qa")
    title: str = Field(..., description="标题")
    source: str = Field("", description="来源")
    content: str = Field("", description="摘要内容")
    score: float | None = Field(None, description="检索得分")


class CitationItem(BaseModel):
    citation_id: str
    document_id: str
    chunk_id: str
    source_type: str
    title: str
    location: str = ""
    quote: str
    source: str = ""
    version: str = ""
    effective_status: str = "unknown"
    score: float | None = None
    metadata: dict = Field(default_factory=dict)


ScopeType = Literal["regulation", "judicial_interpretation", "case", "internal"]
DEFAULT_SCOPE: list[ScopeType] = [
    "regulation", "judicial_interpretation"
]


class SearchFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    effective_status: list[
        Literal["effective", "amended", "repealed", "unknown"]
    ] | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    scope: list[ScopeType] | None = None
    retrieval_mode: str = Field("hybrid", pattern="^(hybrid)$")
    source_filter: str | None = None
    filters: SearchFilters = Field(default_factory=SearchFilters)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    scope: list[ScopeType] | None = None
    retrieval_mode: str = Field("hybrid", pattern="^(hybrid)$")
    source_filter: str | None = None
    filters: SearchFilters = Field(default_factory=SearchFilters)
    session_id: str | None = None


class RetrievalInfo(BaseModel):
    strategy: str = "hybrid"
    retrieved_count: int = 0
    reranked_count: int = 0


class SearchResponse(BaseModel):
    results: list[CitationItem] = Field(default_factory=list)
    retrieval: RetrievalInfo = Field(default_factory=RetrievalInfo)
    trace_id: str


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationItem] = Field(default_factory=list)
    retrieval: RetrievalInfo = Field(default_factory=RetrievalInfo)
    trace_id: str
    session_id: str = ""
    intent: str = "legal_research"
    disclaimer: str = (
        "本回答由系统基于当前知识库资料生成，仅用于法律研究辅助，"
        "不构成正式法律意见。使用前请核对引用原文、资料效力状态及案件具体事实。"
    )


class ChatResponse(BaseModel):
    answer: str
    source_filter: str | None = None
    request_id: str = ""
    session_id: str = ""
    used_rag: bool = False
    used_fallback: bool = False
    references: list[ReferenceItem] = Field(default_factory=list)
    disclaimer: str = "以上内容由 AI 生成，仅供参考，不构成正式法律意见。"


class ErrorResponse(BaseModel):
    code: str
    message: str
    request_id: str


# ---------------------------------------------------------------------------
# 错误码
# ---------------------------------------------------------------------------
ERROR_CODES = {
    "INVALID_REQUEST": (400, "请求参数错误"),
    "SERVICE_NOT_READY": (503, "服务正在初始化，请稍后重试"),
    "RATE_LIMITED": (429, "请求过于频繁"),
    "LLM_ERROR": (502, "大模型调用失败"),
    "RETRIEVAL_ERROR": (502, "知识检索失败"),
    "INTERNAL_ERROR": (500, "系统内部错误"),
    "POLICY_BLOCKED": (403, "数据安全策略禁止该操作"),
}


def error_response(code: str, request_id: str, detail: str | None = None) -> JSONResponse:
    """生成统一格式的错误响应。"""
    http_status, default_msg = ERROR_CODES.get(code, (500, "未知错误"))
    return JSONResponse(
        status_code=http_status,
        content={
            "code": code,
            "message": detail or default_msg,
            "request_id": request_id,
        },
    )


# ---------------------------------------------------------------------------
# 生命周期
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期：启动时初始化法律知识库 RAG 服务。"""
    global assistant, knowledge_base, redis_client, rate_limiter, index_queue
    app.state.ready = False
    logger.info("正在初始化法律助手，请稍候...")
    print("正在初始化法律助手，请稍候...")
    try:
        import redis

        redis_client = redis.Redis(
            host=_cfg.REDIS_HOST,
            port=int(_cfg.REDIS_PORT),
            password=_cfg.REDIS_PASSWORD or None,
            db=int(_cfg.REDIS_DB),
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=5,
        )
        redis_client.ping()
        rate_limiter = DistributedRateLimiter(
            redis_client,
            _cfg.RATE_LIMIT_PER_MINUTE,
        )
        if _cfg.DOCUMENT_REGISTRY_BACKEND == "mysql":
            mysql_repository = MySQLDocumentRepository(_cfg)
            legacy_repository = JsonDocumentRepository(
                knowledge_base.registry_path
            )
            imported = mysql_repository.import_missing(legacy_repository.load())
            if imported:
                logger.info("已从 JSON 迁移 %s 条文档登记记录到 MySQL", imported)
            knowledge_base = KnowledgeBaseManager(
                max_file_size=_cfg.MAX_UPLOAD_MB * 1024 * 1024,
                repository=mysql_repository,
            )
        from main import LegalResearchService
        assistant = LegalResearchService()
        if _cfg.INDEX_QUEUE_ENABLED:
            index_queue = RedisTaskQueue(redis_client, _run_reindex_job)
            index_queue.start()
        app.state.ready = True
        logger.info("法律助手初始化完成，服务已就绪")
        print("法律助手初始化完成，服务已就绪")
    except Exception as e:
        logger.error("LegalResearchService 初始化失败: %s", e)
        print(f"LegalResearchService 初始化失败: {e}", file=sys.stderr)
    yield
    if index_queue:
        index_queue.stop()
    app.state.ready = False
    logger.info("服务正在关闭")


app = FastAPI(
    title="律所法律知识库 RAG 系统",
    description="面向律师、律师助理和律所知识管理人员的法律检索与研究辅助系统",
    version="3.0.0",
    lifespan=lifespan,
)

app.state.ready = False

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cfg.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)


# ---------------------------------------------------------------------------
# 请求日志中间件（记录 path, status_code, cost_ms）
# ---------------------------------------------------------------------------
_rate_lock = Lock()
_rate_windows = defaultdict(deque)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    start = time.perf_counter()
    if request.url.path not in {"/api/v1/health", "/api/v1/ready"}:
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        if rate_limiter:
            allowed = rate_limiter.allow(client_ip)
        else:
            with _rate_lock:
                window = _rate_windows[client_ip]
                while window and now - window[0] >= 60:
                    window.popleft()
                allowed = len(window) < _cfg.RATE_LIMIT_PER_MINUTE
                if allowed:
                    window.append(now)
        if not allowed:
            return error_response("RATE_LIMITED", _generate_request_id())
    response = await call_next(request)
    cost_ms = round((time.perf_counter() - start) * 1000, 1)
    global REQUEST_COUNT, TOTAL_LATENCY_MS, ERROR_COUNT
    REQUEST_COUNT += 1
    TOTAL_LATENCY_MS += cost_ms
    if response.status_code >= 400:
        ERROR_COUNT += 1
    logger.info(
        "[%.1fms] %s %s → %s",
        cost_ms, request.method, request.url.path, response.status_code,
    )
    return response


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _generate_request_id() -> str:
    return str(uuid.uuid4())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z"


# ---------------------------------------------------------------------------
# Token 认证
# ---------------------------------------------------------------------------
def get_principal(authorization: str | None = Header(default=None)) -> Principal | None:
    return authenticate_bearer(authorization, API_TOKEN, RESEARCHER_API_TOKEN)


def require_admin(
    authorization: str | None = Header(default=None),
) -> Principal:
    """管理接口必须使用管理员 Token。"""
    principal = authenticate_bearer(authorization, API_TOKEN, RESEARCHER_API_TOKEN)
    if principal is None:
        raise HTTPException(
            status_code=401,
            detail={"code": "UNAUTHORIZED", "message": "需要有效的 Bearer Token"},
        )
    if principal.role != "admin":
        raise HTTPException(
            status_code=403,
            detail={"code": "FORBIDDEN", "message": "该操作仅允许管理员执行"},
        )
    return principal


verify_token = require_admin


def _require_internal_access(scope, principal):
    scope = scope or []
    if "internal" in scope and principal is None:
        raise HTTPException(
            status_code=401,
            detail={"code": "UNAUTHORIZED", "message": "检索内部资料需要认证"},
        )


def _retrieval_filters(req_filters: SearchFilters, scope, principal) -> dict:
    scope = scope or []
    filters = req_filters.model_dump(exclude_none=True)
    if "internal" in scope:
        filters["access_level"] = (
            ["internal", "restricted"]
            if principal and principal.role == "admin"
            else ["internal"]
        )
        filters["lifecycle_status"] = ["published"]
    return filters


def _llm_is_private() -> bool:
    host = urlparse(_cfg.DASHSCOPE_BASE_URL).hostname or ""
    if host in {"localhost", "host.docker.internal"}:
        return True
    try:
        return ipaddress.ip_address(host).is_private
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# API v1 路由
# ---------------------------------------------------------------------------
@app.get("/api/v1")
def api_v1_root():
    return {
        "service": "律所法律知识库 RAG 系统",
        "version": "3.0",
        "endpoints": {
            "search": "POST /api/v1/search",
            "ask": "POST /api/v1/ask",
            "documents": "GET|POST /api/v1/documents",
            "health": "GET /api/v1/health",
            "ready": "GET /api/v1/ready",
        },
    }


@app.post("/api/v1/search", response_model=SearchResponse)
def search_v1(
    req: SearchRequest,
    principal: Principal | None = Depends(get_principal),
):
    trace_id = _generate_request_id()
    _require_internal_access(req.scope, principal)
    if not app.state.ready:
        return error_response("SERVICE_NOT_READY", trace_id)
    try:
        result = assistant.search(
            req.query,
            source_filter=req.source_filter,
            scope=req.scope,
            filters=_retrieval_filters(req.filters, req.scope, principal),
        )
        audit_logger.record(
            "knowledge.search",
            principal.user_id if principal else "anonymous",
            trace_id=trace_id,
        )
        return SearchResponse(
            results=[CitationItem(**item) for item in result["citations"]],
            retrieval=RetrievalInfo(**result["retrieval"]),
            trace_id=trace_id,
        )
    except Exception as exc:
        logger.error("[%s] 检索失败: %s", trace_id, exc)
        return error_response("RETRIEVAL_ERROR", trace_id)


@app.post("/api/v1/ask", response_model=AskResponse)
def ask_v1(
    req: AskRequest,
    principal: Principal | None = Depends(get_principal),
):
    trace_id = _generate_request_id()
    _require_internal_access(req.scope, principal)
    if (
        "internal" in (req.scope or [])
        and not _cfg.ALLOW_INTERNAL_TO_EXTERNAL_LLM
        and not _llm_is_private()
    ):
        return error_response(
            "POLICY_BLOCKED",
            trace_id,
            "内部资料不得发送至外部大模型；请使用本地模型或由管理员显式批准",
        )
    if not app.state.ready:
        return error_response("SERVICE_NOT_READY", trace_id)

    sm = get_session_manager()
    session_id, _ = sm.get_or_create(req.session_id)
    try:
        result = assistant.answer(
            req.question,
            source_filter=req.source_filter,
            scope=req.scope,
            filters=_retrieval_filters(req.filters, req.scope, principal),
        )
        sm.add_turn(session_id, req.question, result["answer"])
        audit_logger.record(
            "knowledge.ask",
            principal.user_id if principal else "anonymous",
            trace_id=trace_id,
        )
        return AskResponse(
            answer=result["answer"],
            citations=[CitationItem(**item) for item in result.get("citations", [])],
            retrieval=RetrievalInfo(**result.get("retrieval", {})),
            trace_id=trace_id,
            session_id=session_id,
            intent=result.get("intent", "legal_research"),
        )
    except Exception as exc:
        logger.error("[%s] 法律研究问答失败: %s", trace_id, exc)
        return error_response("INTERNAL_ERROR", trace_id)


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat_v1(req: ChatRequest):
    request_id = _generate_request_id()

    if not app.state.ready:
        return error_response("SERVICE_NOT_READY", request_id)

    # 会话管理
    sm = get_session_manager()
    session_id, history = sm.get_or_create(req.session_id)
    sm.add_turn(session_id, req.question, "")  # 先占位，后续更新

    # 构建带上下文的 query（将历史对话压缩为额外输入）
    context_query = req.question
    if history:
        # 取最近 3 轮作为上下文
        recent = history[-6:]
        context_lines = []
        for msg in recent:
            role = "用户" if msg["role"] == "user" else "助手"
            context_lines.append(f"{role}: {msg['content'][:200]}")
        if context_lines:
            context_query = "\n".join(context_lines[-4:]) + f"\n用户: {req.question}"

    try:
        result = assistant.answer(context_query, source_filter=req.source_filter)

        # result 可能为 dict（新格式）或 str（兼容）
        if isinstance(result, dict):
            answer_text = result.get("answer", "")
            # 更新会话历史
            sm.add_turn(session_id, req.question, answer_text)

            return ChatResponse(
                answer=answer_text,
                source_filter=req.source_filter,
                request_id=request_id,
                session_id=session_id,
                used_rag=result.get("used_rag", False),
                used_fallback=result.get("used_fallback", False),
                references=[
                    ReferenceItem(
                        type=ref.get("source_type", "article"),
                        title=ref.get("title", ""),
                        source=ref.get("source", ""),
                        content=ref.get("quote", ""),
                        score=ref.get("score"),
                    )
                    for ref in result.get("citations", result.get("references", []))
                    if ref.get("title")
                ],
                disclaimer="本回答仅用于法律研究辅助，不构成正式法律意见。",
            )
        else:
            answer_text = str(result)
            sm.add_turn(session_id, req.question, answer_text)
            return ChatResponse(
                answer=answer_text,
                source_filter=req.source_filter,
                request_id=request_id,
                session_id=session_id,
                used_rag=False,
                used_fallback=False,
                references=[],
                disclaimer="以上内容由 AI 生成，仅供参考，不构成正式法律意见。",
            )
    except Exception as e:
        logger.error("[%s] 处理聊天请求失败: %s", request_id, e)
        return error_response("INTERNAL_ERROR", request_id)


@app.get("/api/v1/health")
def health_v1():
    return {
        "status": "ok",
        "ready": app.state.ready,
        "version": "3.0.0",
        "timestamp": _now_iso(),
    }


def _tcp_dependency(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=1):
            return True
    except OSError:
        return False


@app.get("/api/v1/ready")
def ready_v1():
    dependencies = {
        "mysql": _tcp_dependency(_cfg.MYSQL_HOST, int(_cfg.MYSQL_PORT)),
        "redis": _tcp_dependency(_cfg.REDIS_HOST, int(_cfg.REDIS_PORT)),
        "milvus": _tcp_dependency(_cfg.MILVUS_HOST, int(_cfg.MILVUS_PORT)),
        "rag": bool(app.state.ready and assistant is not None),
        "index_queue": bool(index_queue is not None or not _cfg.INDEX_QUEUE_ENABLED),
    }
    if not all(dependencies.values()):
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "ready": False,
                "dependencies": dependencies,
            },
        )
    return {"status": "ready", "ready": True, "dependencies": dependencies}


# ---------------------------------------------------------------------------
# Prometheus 指标端点
# ---------------------------------------------------------------------------
REQUEST_COUNT = 0
ERROR_COUNT = 0
TOTAL_LATENCY_MS = 0.0


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """暴露 Prometheus 格式指标。"""
    queue_depth = redis_client.llen(RedisTaskQueue.READY) if redis_client else 0
    processing = redis_client.llen(RedisTaskQueue.PROCESSING) if redis_client else 0
    lines = [
        "# HELP law_rag_http_requests_total Total HTTP requests.",
        "# TYPE law_rag_http_requests_total counter",
        f"law_rag_http_requests_total {REQUEST_COUNT}",
        "# HELP law_rag_http_errors_total Total HTTP error responses.",
        "# TYPE law_rag_http_errors_total counter",
        f"law_rag_http_errors_total {ERROR_COUNT}",
        "# HELP law_rag_http_latency_milliseconds_sum Total HTTP latency.",
        "# TYPE law_rag_http_latency_milliseconds_sum counter",
        f"law_rag_http_latency_milliseconds_sum {TOTAL_LATENCY_MS:.3f}",
        "# HELP law_rag_index_queue_depth Queued indexing jobs.",
        "# TYPE law_rag_index_queue_depth gauge",
        f"law_rag_index_queue_depth {queue_depth}",
        "# HELP law_rag_index_jobs_processing Processing indexing jobs.",
        "# TYPE law_rag_index_jobs_processing gauge",
        f"law_rag_index_jobs_processing {processing}",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 兼容旧路由（指向 v1 实现）
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(
        os.path.join("web", "index.html"),
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat_legacy(req: ChatRequest):
    return chat_v1(req)


@app.get("/api/health")
def health_legacy():
    return health_v1()


# ---------------------------------------------------------------------------
# 知识库文档管理 API
# ---------------------------------------------------------------------------
@app.post("/api/v1/documents", status_code=201)
async def create_document(
    request: Request,
    file_name: str,
    document_type: str,
    title: str | None = None,
    access_level: Literal["internal", "restricted"] = "internal",
    principal: Principal = Depends(require_admin),
):
    """上传法律资料。请求体为原始文件内容。"""
    trace_id = _generate_request_id()
    try:
        content = await request.body()
        item = knowledge_base.create(
            file_name=file_name,
            file_object=io.BytesIO(content),
            document_type=document_type,
            title=title,
            metadata={"access_level": access_level},
        )
        audit_logger.record(
            "document.upload",
            principal.user_id,
            target=item["id"],
            trace_id=trace_id,
        )
        return {"document": item, "trace_id": trace_id}
    except KnowledgeBaseError as exc:
        return error_response("INVALID_REQUEST", trace_id, str(exc))
    except Exception as exc:
        logger.error("[%s] 文档上传失败: %s", trace_id, exc)
        return error_response("INTERNAL_ERROR", trace_id)


@app.get("/api/v1/documents")
def list_documents(
    document_type: str | None = None,
    status: str | None = None,
    principal: Principal = Depends(require_admin),
):
    return {"documents": knowledge_base.list(document_type=document_type, status=status)}


@app.get("/api/v1/documents/{document_id}")
def get_document(
    document_id: str,
    principal: Principal = Depends(require_admin),
):
    item = knowledge_base.get(document_id)
    if item is None:
        raise HTTPException(status_code=404, detail={"code": "DOCUMENT_NOT_FOUND", "message": "文档不存在"})
    return {"document": item}


def _run_reindex(document_id: str, item: dict, actor: str, trace_id: str):
    try:
        knowledge_base.update_status(document_id, "indexing")
        stage_dir = knowledge_base.staging_directory(document_id)
        indexed_item = dict(item)
        indexed_item["stage_dir"] = str(stage_dir)
        chunk_count = assistant.index_document(indexed_item)
        knowledge_base.update_status(document_id, "indexed", chunk_count=chunk_count)
        audit_logger.record(
            "document.reindex",
            actor,
            target=document_id,
            trace_id=trace_id,
        )
    except Exception as exc:
        knowledge_base.update_status(document_id, "failed", error=str(exc))
        audit_logger.record(
            "document.reindex",
            actor,
            target=document_id,
            trace_id=trace_id,
            result="failed",
        )
        logger.error("[%s] 文档重新索引失败: %s", trace_id, exc)
    finally:
        knowledge_base.cleanup_staging(document_id)


def _run_reindex_job(job: dict):
    document_id = job["document_id"]
    item = knowledge_base.get(document_id)
    if item is None:
        raise KnowledgeBaseError("文档不存在")
    _run_reindex(
        document_id,
        item,
        job["actor"],
        job["trace_id"],
    )
    refreshed = knowledge_base.get(document_id)
    if refreshed and refreshed.get("processing_status") == "failed":
        raise RuntimeError(refreshed.get("error") or "索引失败")


@app.post("/api/v1/documents/{document_id}/reindex", status_code=202)
def reindex_document(
    document_id: str,
    principal: Principal = Depends(require_admin),
):
    trace_id = _generate_request_id()
    if not app.state.ready or assistant is None:
        return error_response("SERVICE_NOT_READY", trace_id)

    item = knowledge_base.get(document_id)
    if item is None:
        raise HTTPException(status_code=404, detail={"code": "DOCUMENT_NOT_FOUND", "message": "文档不存在"})
    if item.get("lifecycle_status") != "published":
        raise HTTPException(
            status_code=409,
            detail={"code": "DOCUMENT_NOT_PUBLISHED", "message": "只有已发布文档可以建立检索索引"},
        )
    if index_queue is None:
        return error_response("SERVICE_NOT_READY", trace_id, "索引任务队列未就绪")

    queued = knowledge_base.update_status(document_id, "queued")
    job_id = index_queue.enqueue(
        {
            "document_id": document_id,
            "actor": principal.user_id,
            "trace_id": trace_id,
        }
    )
    return {"document": queued, "job_id": job_id, "trace_id": trace_id}


@app.get("/api/v1/index-jobs/{job_id}")
def get_index_job(
    job_id: str,
    principal: Principal = Depends(require_admin),
):
    if index_queue is None:
        return error_response("SERVICE_NOT_READY", _generate_request_id())
    status = index_queue.status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail={"code": "JOB_NOT_FOUND", "message": "任务不存在"})
    status.pop("payload", None)
    return {"job_id": job_id, **status}


_LIFECYCLE_TRANSITIONS = {
    "draft": {"pending_review"},
    "pending_review": {"draft", "published"},
    "published": {"retired"},
    "retired": set(),
}


class LifecycleRequest(BaseModel):
    status: Literal["draft", "pending_review", "published", "retired"]


@app.post("/api/v1/documents/{document_id}/lifecycle")
def update_document_lifecycle(
    document_id: str,
    req: LifecycleRequest,
    principal: Principal = Depends(require_admin),
):
    item = knowledge_base.get(document_id)
    if item is None:
        raise HTTPException(status_code=404, detail={"code": "DOCUMENT_NOT_FOUND", "message": "文档不存在"})
    current = item.get("lifecycle_status", "draft")
    if req.status not in _LIFECYCLE_TRANSITIONS[current]:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "INVALID_LIFECYCLE_TRANSITION",
                "message": f"不允许从 {current} 变更为 {req.status}",
            },
        )
    updated = knowledge_base.update_lifecycle(
        document_id,
        req.status,
        principal.user_id,
    )
    if req.status == "retired" and item.get("processing_status") == "indexed":
        assistant.delete_document_index(document_id)
        updated = knowledge_base.update_status(document_id, "retired")
    audit_logger.record(
        "document.lifecycle",
        principal.user_id,
        target=document_id,
        trace_id=_generate_request_id(),
    )
    return {"document": updated}


@app.delete("/api/v1/documents/{document_id}")
def delete_document(
    document_id: str,
    principal: Principal = Depends(require_admin),
):
    trace_id = _generate_request_id()
    item = knowledge_base.get(document_id)
    if item is None:
        raise HTTPException(status_code=404, detail={"code": "DOCUMENT_NOT_FOUND", "message": "文档不存在"})
    if item.get("processing_status") == "indexed" and (
        not app.state.ready or assistant is None
    ):
        return error_response(
            "SERVICE_NOT_READY",
            trace_id,
            "检索服务未就绪，暂不能安全删除已索引文档",
        )

    try:
        if item.get("processing_status") == "indexed":
            assistant.delete_document_index(document_id)
        knowledge_base.delete(document_id)
        audit_logger.record(
            "document.delete",
            principal.user_id,
            target=document_id,
            trace_id=trace_id,
        )
        logger.info("[%s] 知识库文档已删除: %s", trace_id, document_id)
        return {"status": "deleted", "document_id": document_id, "trace_id": trace_id}
    except Exception as exc:
        logger.error("[%s] 删除知识库文档失败: %s", trace_id, exc)
        return error_response("INTERNAL_ERROR", trace_id)


# ---------------------------------------------------------------------------
# Admin 管理 API
# ---------------------------------------------------------------------------
class QAItem(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    answer: str = Field(..., min_length=1)
    domain: str = ""
    created_by: str = ""


class QAUpdateItem(BaseModel):
    question: str | None = None
    answer: str | None = None
    domain: str | None = None
    status: str | None = None


@app.post("/api/v1/admin/qa", status_code=201, dependencies=[Depends(verify_token)])
def create_qa(item: QAItem):
    """新增问答对。"""
    if not app.state.ready or assistant is None:
        return error_response("SERVICE_NOT_READY", _generate_request_id())
    try:
        qa_id = assistant.law_qa_system.mysql_client.add_qa(
            question=item.question,
            answer=item.answer,
            domain=item.domain,
            created_by=item.created_by,
        )
        if qa_id:
            return {"id": qa_id, "status": "created"}
        return error_response("INTERNAL_ERROR", _generate_request_id(), "新增失败")
    except Exception as e:
        logger.error("新增问答对失败: %s", e)
        return error_response("INTERNAL_ERROR", _generate_request_id())


@app.put("/api/v1/admin/qa/{qa_id}", dependencies=[Depends(verify_token)])
def update_qa(qa_id: int, item: QAUpdateItem):
    """更新问答对。"""
    if not app.state.ready or assistant is None:
        return error_response("SERVICE_NOT_READY", _generate_request_id())
    try:
        ok = assistant.law_qa_system.mysql_client.update_qa(
            qa_id=qa_id,
            question=item.question,
            answer=item.answer,
            domain=item.domain,
            status=item.status,
        )
        if ok:
            return {"id": qa_id, "status": "updated"}
        return error_response("INTERNAL_ERROR", _generate_request_id(), "更新失败")
    except Exception as e:
        logger.error("更新问答对失败: %s", e)
        return error_response("INTERNAL_ERROR", _generate_request_id())


@app.delete("/api/v1/admin/qa/{qa_id}", dependencies=[Depends(verify_token)])
def delete_qa(qa_id: int, hard: bool = False):
    """删除问答对。hard=true 物理删除；默认软删除。"""
    if not app.state.ready or assistant is None:
        return error_response("SERVICE_NOT_READY", _generate_request_id())
    try:
        ok = assistant.law_qa_system.mysql_client.delete_qa(qa_id=qa_id, hard=hard)
        if ok:
            return {"id": qa_id, "status": "deleted"}
        return error_response("INTERNAL_ERROR", _generate_request_id(), "删除失败")
    except Exception as e:
        logger.error("删除问答对失败: %s", e)
        return error_response("INTERNAL_ERROR", _generate_request_id())


class DeleteSessionRequest(BaseModel):
    session_id: str = Field(..., description="要删除的会话 ID")


@app.post("/api/v1/user/data/delete", dependencies=[Depends(verify_token)])
def delete_user_data(req: DeleteSessionRequest):
    """根据 PIPL 要求，提供用户数据删除接口。"""
    try:
        from base.session_manager import get_session_manager
        sm = get_session_manager()
        sm.clear(req.session_id)
        logger.info("用户数据已删除，session_id=%s", req.session_id)
        return {"status": "deleted", "session_id": req.session_id}
    except Exception as e:
        logger.error("删除用户数据失败: %s", e)
        return error_response("INTERNAL_ERROR", _generate_request_id())


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
