# 模块文档字符串：语义缓存，用于降级场景的兜底
"""语义缓存 - 降级场景兜底"""

# 启用 Python 3.10+ 的新特性支持
from __future__ import annotations

# 导入哈希计算模块，用于生成缓存键
import hashlib
# 导入 JSON 序列化模块
import json
# 导入时间模块，用于缓存过期判断
import time
# 导入有序字典，用于实现 LRU 淘汰策略
from collections import OrderedDict
# 导入类型提示工具
from typing import Any, Optional


# 定义语义缓存类，用于降级场景的轻量级缓存
class SemanticCache:
    """轻量语义缓存 (降级场景使用)"""

    # 初始化缓存实例
    def __init__(self, ttl_seconds: int = 86400, max_size: int = 1000):
        # 设置缓存项的生存时间（默认 24 小时）
        self._ttl = ttl_seconds
        # 设置缓存的最大容量
        self._max_size = max_size
        # 初始化有序字典，存储键值对和过期时间
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()

    # 根据查询内容和范围生成缓存键
    def _make_key(self, query: str, scope: str = "") -> str:
        # 拼接范围和查询内容，转为小写
        raw = f"{scope}:{query.strip().lower()}"
        # 使用 MD5 哈希生成固定长度的键
        return hashlib.md5(raw.encode()).hexdigest()

    # 从缓存中获取数据
    def get(self, query: str, scope: str = "") -> Optional[Any]:
        # 生成缓存键
        key = self._make_key(query, scope)
        # 如果键不存在，返回 None
        if key not in self._cache:
            return None
        # 获取缓存数据和过期时间
        data, expire_at = self._cache[key]
        # 如果已过期，删除该缓存项并返回 None
        if time.time() > expire_at:
            del self._cache[key]
            return None
        # 将访问的键移到末尾（LRU 策略）
        self._cache.move_to_end(key)
        # 返回缓存的数据
        return data

    # 将数据存入缓存
    def set(self, query: str, data: Any, scope: str = ""):
        # 生成缓存键
        key = self._make_key(query, scope)
        # 存储数据和过期时间戳
        self._cache[key] = (data, time.time() + self._ttl)
        # 如果缓存超出最大容量，淘汰最久未使用的项
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    # 清空所有缓存
    def clear(self):
        self._cache.clear()


# 创建全局语义缓存实例
semantic_cache = SemanticCache()