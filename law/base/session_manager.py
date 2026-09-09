#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
会话管理器

管理多轮对话的历史记录。

存储方式：
    - 优先使用 Redis（TTL=1 小时）
    - Redis 不可用时降级为内存存储（进程重启后丢失）

窗口策略：
    - 保留最近 MAX_TURNS 轮对话（1 轮 = 1 问 + 1 答）
    - 超出时丢弃最早轮次
"""

import json
import hashlib
import hmac
import logging
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from base.config import Config

logger = logging.getLogger("session_manager")

# 配置
MAX_TURNS = 10  # 最多保留 10 轮（20 条消息）
SESSION_TTL = 3600  # Redis TTL：1 小时

_cfg = Config()


def _get_redis_client():
    """尝试获取 Redis 连接，失败返回 None。"""
    try:
        import redis
        client = redis.Redis(
            host=_cfg.REDIS_HOST,
            port=int(_cfg.REDIS_PORT),
            password=_cfg.REDIS_PASSWORD or None,
            db=int(_cfg.REDIS_DB),
            socket_connect_timeout=2,
            socket_timeout=2,
            decode_responses=True,
        )
        client.ping()
        return client
    except Exception:
        return None


class SessionManager:
    """会话管理器。"""

    def __init__(self):
        self._redis = _get_redis_client()
        self._memory_store = {}  # 降级方案：内存存储
        self._signing_key = (_cfg.SESSION_SIGNING_KEY or _cfg.API_TOKEN).encode("utf-8")
        if self._redis:
            logger.info("会话管理器：使用 Redis 存储")
        else:
            logger.warning("会话管理器：Redis 不可用，降级为内存存储（重启后丢失）")

    def get_or_create(self, session_id=None):
        """获取或创建会话，返回 (session_id, history)。"""
        if not session_id or not self._valid_session_id(session_id):
            session_id = self._create_session_id()
            history = []
            self._save(session_id, history)
            return session_id, history
        history = self._load(session_id)
        if history is None:
            history = []
            self._save(session_id, history)
        return session_id, history

    def add_turn(self, session_id, user_msg, assistant_msg):
        """添加一轮对话，自动截断。"""
        history = self._load(session_id)
        if history is None:
            history = []
        history.append({"role": "user", "content": user_msg, "timestamp": time.time()})
        history.append({"role": "assistant", "content": assistant_msg, "timestamp": time.time()})
        # 窗口截断：保留最近 MAX_TURNS 轮
        if len(history) > MAX_TURNS * 2:
            history = history[-(MAX_TURNS * 2):]
        self._save(session_id, history)
        return history

    def get_history(self, session_id):
        """获取会话历史。"""
        return self._load(session_id) or []

    def clear(self, session_id):
        """清除会话。"""
        if not self._valid_session_id(session_id):
            return False
        if self._redis:
            self._redis.delete(f"session:{session_id}")
        self._memory_store.pop(session_id, None)
        return True

    def _create_session_id(self):
        value = uuid.uuid4().hex
        signature = hmac.new(
            self._signing_key,
            value.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        return f"{value}.{signature}"

    def _valid_session_id(self, session_id):
        try:
            value, signature = str(session_id).split(".", 1)
        except ValueError:
            return False
        if len(value) != 32 or len(signature) != 64:
            return False
        expected = hmac.new(
            self._signing_key,
            value.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(signature, expected)

    def _load(self, session_id):
        if self._redis:
            data = self._redis.get(f"session:{session_id}")
            if data:
                return json.loads(data)
            return None
        return self._memory_store.get(session_id)

    def _save(self, session_id, history):
        if self._redis:
            self._redis.setex(f"session:{session_id}", SESSION_TTL, json.dumps(history, ensure_ascii=False))
        else:
            self._memory_store[session_id] = history


# 全局单例
_session_manager = None


def get_session_manager():
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
