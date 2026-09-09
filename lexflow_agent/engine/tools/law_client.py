"""law RAG 客户端封装 - 法律检索服务 HTTP 客户端

本模块封装了与外部法律 RAG（Retrieval-Augmented Generation）服务的通信。
提供两种主要功能：
1. search - 混合检索（法条、案例、文章等）
2. ask - 问答式检索（基于检索结果生成答案）

客户端内置：
- 自动重试（使用 with_retry 装饰器）
- LangSmith 追踪（使用 @traceable 装饰器）
- 健康检查（用于服务可用性检测）
"""

from __future__ import annotations  # 启用未来类型注解

from typing import Optional  # 可选类型提示

import httpx  # HTTP 客户端库
from langsmith import traceable  # LangSmith 追踪装饰器

from lexflow_agent.config.settings import settings  # 全局配置
from lexflow_agent.engine.resilience.retry import with_retry, RetryableError  # 重试机制


class LawRAGClient:
    """law RAG HTTP 客户端 - 封装与法律检索服务的通信
    
    提供法条检索、案例检索、问答等功能。
    客户端内置重试机制和 LangSmith 追踪。
    """

    def __init__(self):
        """初始化客户端 - 从配置加载参数并创建 HTTP 客户端"""
        self.base_url = settings.LAW_RAG_BASE_URL.rstrip("/")  # 基础 URL（去除末尾斜杠）
        self.api_key = settings.LAW_RAG_API_KEY  # API 密钥
        self.search_timeout = settings.LAW_RAG_SEARCH_TIMEOUT  # 搜索超时（秒）
        self.ask_timeout = settings.LAW_RAG_ASK_TIMEOUT  # 问答超时（秒）
        self._client = httpx.Client(timeout=self.search_timeout)  # 同步 HTTP 客户端

    @with_retry(max_retries=2)  # 最多重试 2 次
    @traceable(run_type="tool", name="law_rag.search")  # LangSmith 追踪
    def search(self, query: str, scope: list[str] | None = None, top_k: int = 5) -> dict:
        """调用 law /api/v1/search - 混合检索法条、案例等
        
        Args:
            query: 检索查询词
            scope: 检索范围（如：["laws", "cases"]），None 表示全范围
            top_k: 返回结果数量（默认 5 条）
        
        Returns:
            检索结果字典
        
        Raises:
            RetryableError: 服务繁忙（503）时可重试
        """
        resp = self._client.post(
            f"{self.base_url}/api/v1/search",
            json={"query": query, "scope": scope, "retrieval_mode": "hybrid"},  # 混合检索模式
            headers=self._headers(),  # 认证头
        )
        if resp.status_code == 503:  # 服务不可用
            raise RetryableError("law RAG 服务繁忙")
        resp.raise_for_status()  # 其他错误抛出异常
        return resp.json()  # 返回 JSON 结果

    @with_retry(max_retries=2)  # 最多重试 2 次
    @traceable(run_type="tool", name="law_rag.ask")  # LangSmith 追踪
    def ask(self, question: str, scope: list[str] | None = None) -> dict:
        """调用 law /api/v1/chat - 问答式检索
        
        基于检索结果生成自然语言答案，适用于需要解释性回答的场景。
        
        Args:
            question: 用户问题
            scope: 检索范围（如：["laws", "cases"]），None 表示全范围
        
        Returns:
            问答结果字典（包含答案和引用来源）
        
        Raises:
            RetryableError: 服务繁忙（503）时可重试
        """
        resp = self._client.post(
            f"{self.base_url}/api/v1/ask",
            json={"question": question, "scope": scope},
            headers=self._headers(),
        )
        if resp.status_code == 503:  # 服务不可用
            raise RetryableError("law RAG 服务繁忙")
        resp.raise_for_status()  # 其他错误抛出异常
        return resp.json()  # 返回 JSON 结果

    def _headers(self) -> dict:
        """生成 HTTP 请求头 - 包含认证信息（如果有）
        
        Returns:
            请求头字典
        """
        headers = {}
        if self.api_key:  # 如果配置了 API Key
            headers["Authorization"] = f"Bearer {self.api_key}"  # Bearer Token 认证
        return headers

    def health(self) -> bool:
        """健康检查 - 检测 law RAG 服务是否可用
        
        Returns:
            True 表示服务可用，False 表示服务不可用
        """
        try:
            resp = self._client.get(f"{self.base_url}/api/v1/health", timeout=5)
            return resp.status_code == 200
        except Exception:  # 任何异常（网络错误、超时等）
            return False


# 全局单例 - 所有节点共享同一个 law RAG 客户端
law_client = LawRAGClient()