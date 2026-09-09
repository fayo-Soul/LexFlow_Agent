"""LangSmith 追踪工具 — LangChainTracer 回调 + traceable 装饰器封装

Provides:
- get_langsmith_callbacks(): returns a LangChainTracer list for graph.invoke(config)
- trace_node(): decorator factory that wraps an agent node as a @traceable span
- get_tracer_config(): helper to build a LangGraph config dict with tracing enabled
"""

from __future__ import annotations

import os
from typing import Any, Callable

from lexflow_agent.config.settings import settings


# ── 暴露标准 API ────────────────────────────────────────────
# 方便业务节点直接从 langsmith 包导入后按需使用
try:
    from langsmith import traceable  # noqa: F401
except ImportError:
    traceable = None


def _tracing_enabled() -> bool:
    """检查 LangSmith 追踪是否已配置且开启"""
    return bool(
        settings.LANGSMITH_TRACING
        and settings.LANGSMITH_API_KEY
        and settings.LANGSMITH_ENDPOINT
    )


def get_langsmith_callbacks() -> list[Any]:
    """获取 LangSmith 回调列表，用于注入 LangGraph 的 config['callbacks']。

    Returns:
        [LangChainTracer] 如果追踪已启用，否则返回空列表。
    """
    if not _tracing_enabled():
        return []

    try:
        from langchain_classic.callbacks.tracers.langchain import LangChainTracer

        tracer = LangChainTracer(project_name=settings.LANGSMITH_PROJECT)
        return [tracer]
    except Exception:
        return []


def get_tracer_config(*, thread_id: str, extra_tags: list[str] | None = None, extra_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """构建包含 LangSmith tracer 的 LangGraph config。

    Args:
        thread_id: LangGraph thread_id（用于 checkpoint 分组）
        extra_tags: 附加标签（会在 graph config 的 tags 中体现）
        extra_metadata: 附加元数据（LangSmith 面板可见）

    Returns:
        LangGraph config dict，可直接传入 graph.invoke() / graph.stream()。
    """
    cfg: dict[str, Any] = {
        "configurable": {"thread_id": thread_id},
        "tags": extra_tags or [],
        "metadata": extra_metadata or {},
    }

    callbacks = get_langsmith_callbacks()
    if callbacks:
        cfg["callbacks"] = callbacks

    return cfg


def trace_node(name: str | None = None, *, run_type: str = "chain") -> Callable:
    """装饰器工厂：将函数包装为 LangSmith @traceable span。

    用法:
        @trace_node("extract_facts")
        def extract_facts_node(state): ...

    如果 LangSmith 不可用，返回原始函数（安全降级）。

    Args:
        name: span 在 LangSmith 面板中的显示名（默认取函数名）
        run_type: run type — "chain" / "tool" / "llm" / "retriever"

    Returns:
        装饰后的函数，LangSmith 不可用时返回原函数。
    """
    if traceable is None:
        return lambda fn: fn  # 无 langsmith 时，原样返回

    def decorator(fn: Callable) -> Callable:
        wrapped_name = name or fn.__name__
        return traceable(run_type=run_type, name=wrapped_name)(fn)

    return decorator
