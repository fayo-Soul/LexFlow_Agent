"""全局配置 - Pydantic Settings"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


# .env 文件绝对路径
_proj_root = Path(__file__).parent.parent.parent
_env_path = _proj_root / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_env_path) if _env_path.exists() else ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── 应用 ──
    APP_NAME: str = "lexflow-agent"
    APP_VERSION: str = "1.1.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── API ──
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_TOKEN: str = ""
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # ── LLM ──
    LLM_PRIMARY_PROVIDER: str = "deepseek"
    LLM_PRIMARY_MODEL: str = "deepseek-chat"
    LLM_PRIMARY_API_KEY: str = ""
    LLM_PRIMARY_BASE_URL: str = "https://api.deepseek.com/v1"

    LLM_BACKUP_PROVIDER: str = "openai"
    LLM_BACKUP_MODEL: str = "qwen-plus"
    LLM_BACKUP_API_KEY: str = ""
    LLM_BACKUP_BASE_URL: str = ""

    LLM_TIMEOUT: int = 60
    LLM_MAX_TOKENS: int = 4096
    LLM_TEMPERATURE: float = 0.1

    # ── law RAG ──
    LAW_RAG_BASE_URL: str = "http://192.168.88.100:8001"
    LAW_RAG_API_KEY: str = ""
    LAW_RAG_SEARCH_TIMEOUT: int = 30
    LAW_RAG_ASK_TIMEOUT: int = 30

    # ── 数据库 / Checkpoint ──
    CHECKPOINT_DB_URL: str = "sqlite:///./checkpoints.db"
    CHECKPOINT_DB_URL_PSQL: str = ""

    # ── Redis / 缓存 ──
    REDIS_URL: str = ""
    CACHE_TTL_SECONDS: int = 86400

    # ── Runtime ──
    MAX_CONCURRENT_TASKS: int = 5
    WORKFLOW_TIMEOUT_SECONDS: int = 1800
    NODE_TIMEOUT_SECONDS: int = 300
    LLM_CALL_TIMEOUT_SECONDS: int = 60
    GRACEFUL_SHUTDOWN_WAIT: int = 30
    IDEMPOTENCY_KEY_TTL_HOURS: int = 24
    TOKEN_BUDGET_PER_TASK: int = 500_000

    # ── 熔断器 ──
    CB_WINDOW_SECONDS: int = 60
    CB_FAILURE_THRESHOLD: float = 0.5
    CB_COOLDOWN_SECONDS: int = 30

    # ── 重试 ──
    RETRY_MAX_ATTEMPTS: int = 3
    RETRY_BASE_DELAY: float = 1.0
    RETRY_MAX_DELAY: float = 10.0

    # ── 文件路径 ──
    PROMPT_DIR: str = "engine/prompts"
    DATA_DIR: str = "data"

    # ── 文件解析 ──
    MINERU_CONDA_ENV: str = "mineru"       # MinerU 所在的 conda 环境名
    MINERU_BACKEND: str = "pipeline"       # MinerU 后端: pipeline / vlm / hybrid-auto-engine
    MINERU_TIMEOUT: int = 300              # MinerU 单文件解析超时（秒）
    MAX_UPLOAD_FILE_SIZE_MB: int = 50      # 上传文件大小上限
    FILE_PARSER_ENABLED: bool = True       # 是否启用文件解析工具

    # ── LangSmith (可观测性) ──
    LANGSMITH_TRACING: bool = True
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "lexflow-agent"
    LANGSMITH_PROMPT_HUB: bool = True  # 是否启用 LangSmith Prompt Hub

    @property
    def is_checkpoint_psql(self) -> bool:
        return bool(self.CHECKPOINT_DB_URL_PSQL)


settings = Settings()

# API_TOKEN 兜底
if not settings.API_TOKEN and not settings.DEBUG:
    settings.API_TOKEN = "dev-token-do-not-use-in-prod"
    import warnings
    warnings.warn("API_TOKEN 未设置，已使用开发默认值")
