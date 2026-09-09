"""重试装饰器 - 指数退避 + 抖动

本模块实现了带指数退避和随机抖动的重试机制，用于处理可重试的临时性错误
（如网络超时、503 服务不可用、限流等）。

重试策略：
- 指数退避：每次重试的等待时间按 2^n 增长
- 随机抖动：在退避时间基础上添加 0-10% 的随机偏移，避免多个请求同时重试
- 异常分类：区分可重试异常（RetryableError）和不可重试异常（NonRetryableError）
"""

from __future__ import annotations  # 启用未来类型注解

import random  # 随机数生成（用于抖动）
import time  # 时间处理（用于延迟）
from functools import wraps  # 装饰器工具
from typing import Callable, Tuple, Type  # 类型提示工具


class RetryableError(Exception):
    """可重试的异常 - 表示临时性错误，可以通过重试恢复
    
    典型场景：
    - 网络超时（TimeoutError）
    - 503 服务不可用
    - 限流错误（429 Too Many Requests）
    - 连接重置（ConnectionResetError）
    """
    pass


class NonRetryableError(Exception):
    """不可重试的异常 - 表示永久性错误，重试无法恢复
    
    典型场景：
    - 401 认证失败（API Key 无效）
    - 400 请求错误（参数错误）
    - Schema 校验失败（数据结构不匹配）
    """
    pass


def with_retry(
    max_retries: int = 3,  # 最大重试次数（不含首次调用）
    base_delay: float = 1.0,  # 基础延迟时间（秒）
    max_delay: float = 10.0,  # 最大延迟时间（秒）
    retryable_exceptions: Tuple[Type[Exception], ...] = (  # 可重试的异常类型元组
        RetryableError,  # 自定义可重试错误
        TimeoutError,  # 超时错误
        ConnectionError,  # 连接错误
        ConnectionResetError,  # 连接重置错误
    ),
):
    """带指数退避 + 抖动的重试装饰器
    
    工作原理：
    1. 首次调用函数
    2. 如果失败且异常属于 retryable_exceptions，则等待一段时间后重试
    3. 等待时间 = min(base_delay * 2^attempt, max_delay) + jitter
    4. jitter = random.uniform(0, delay * 0.1) （0-10% 的随机偏移）
    5. 如果异常属于 NonRetryableError，立即抛出，不重试
    6. 达到最大重试次数后，抛出最后一次异常
    
    用法示例:
        @with_retry(max_retries=3)
        def call_llm(...):
            ...
    
    Args:
        max_retries: 最大重试次数（默认 3 次）
        base_delay: 基础延迟时间（秒），用于计算指数退避（默认 1.0 秒）
        max_delay: 最大延迟时间（秒），防止等待过久（默认 10.0 秒）
        retryable_exceptions: 可重试的异常类型元组
    
    Returns:
        装饰器函数
    """
    def decorator(func: Callable):
        @wraps(func)  # 保留原函数的元数据（名称、文档字符串等）
        def wrapper(*args, **kwargs):
            last_exception = None  # 记录最后一次异常
            for attempt in range(max_retries + 1):  # 首次调用 + max_retries 次重试
                try:
                    return func(*args, **kwargs)  # 尝试调用函数
                except retryable_exceptions as e:  # 捕获可重试异常
                    last_exception = e
                    if attempt < max_retries:  # 如果还有重试机会
                        # 计算退避时间：指数增长 + 随机抖动
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        jitter = random.uniform(0, delay * 0.1)
                        time.sleep(delay + jitter)  # 等待
                except NonRetryableError:  # 不可重试异常，立即抛出
                    raise
            raise last_exception  # 达到最大重试次数，抛出最后一次异常
        return wrapper
    return decorator