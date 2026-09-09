"""降级执行器 - 集中管理各节点的降级策略"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Optional

import yaml

from lexflow_agent.engine.resilience.retry import with_retry, RetryableError
from lexflow_agent.engine.resilience.circuit_breaker import circuit_breaker
from lexflow_agent.engine.models.review import DegradationRecord
from lexflow_agent.engine.gateway.factory import LLMFactory


class ExecutionResult:
    def __init__(self, success: bool = False, data: Any = None,
                 degraded: bool = False, fallback_triggered: bool = False,
                 retry_count: int = 0, model_used: str = "",
                 token_usage: int = 0, state_updates: Optional[dict] = None):
        self.success = success
        self.data = data
        self.degraded = degraded
        self.fallback_triggered = fallback_triggered
        self.retry_count = retry_count
        self.model_used = model_used
        self.token_usage = token_usage
        self.state_updates = state_updates or {}
        self.degradations: list[DegradationRecord] = []


def _load_degradation_config() -> dict:
    """加载降级策略配置"""
    import os
    from pathlib import Path
    path = Path(__file__).parent.parent.parent / "config" / "degradation.yaml"
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


DEGRADATION_MAP: dict = {}


def _ensure_config():
    if not DEGRADATION_MAP:
        raw = _load_degradation_config()
        DEGRADATION_MAP.update(raw.get("degradation", {}))


def resilience_execute(
    node_name: str,
    llm_call: Callable,
    state: Any,
    fallback_data: Any = None,
) -> ExecutionResult:
    """统一降级执行器

    五层阶梯:
        1. 熔断检查 → 直接降级
        2. 自动重试 → 指数退避
        3. 模型 Failover → 切备用模型
        4. 节点降级 → 返回 fallback
        5. 兜底标记 → 记录 fallback_triggered
    """
    _ensure_config()
    result = ExecutionResult()
    strategy = DEGRADATION_MAP.get(node_name, {})
    now = datetime.now()

    # 第一层：熔断检查
    if circuit_breaker.is_open():
        result.degraded = True
        result.fallback_triggered = True
        result.degradations.append(DegradationRecord(
            node=node_name, reason="熔断器开启", level=4,
            action=strategy.get("fallback_action", "skip"), timestamp=now,
        ))
        return _apply_node_fallback(result, node_name, strategy, fallback_data)

    # 第二层：自动重试
    if strategy.get("retryable", False):
        try:
            # 使用指数退避装饰器
            @with_retry(max_retries=strategy.get("max_retries", 3))
            def _call():
                # 模型 failover 逻辑
                try:
                    client = LLMFactory.get_client()
                    result.model_used = "primary"
                    return llm_call(client)
                except (RetryableError, TimeoutError, ConnectionError):
                    # 第三层：切备用模型
                    backup = LLMFactory.get_backup_client()
                    result.model_used = "backup"
                    result.degradations.append(DegradationRecord(
                        node=node_name, reason="主模型不可用，切换到备用模型",
                        level=3, action="model_failover", timestamp=now,
                    ))
                    return llm_call(backup)

            data = _call()
            result.success = True
            result.data = data
            circuit_breaker.record_success()
            return result
        except Exception as e:
            circuit_breaker.record_failure()
            if circuit_breaker.should_open():
                circuit_breaker.open()
            # 第四层：节点降级
            return _apply_node_fallback(result, node_name, strategy, fallback_data)
    else:
        # 不可重试节点
        try:
            data = llm_call(LLMFactory.get_client())
            result.success = True
            result.data = data
            return result
        except Exception:
            return _apply_node_fallback(result, node_name, strategy, fallback_data)


def _apply_node_fallback(
    result: ExecutionResult, node_name: str,
    strategy: dict, fallback_data: Any,
) -> ExecutionResult:
    """应用节点降级策略"""
    result.success = True  # 降级也算成功（返回 fallback 值）
    result.degraded = True
    result.data = fallback_data
    degrade_mark = strategy.get("degrade_mark", f"{node_name}_degraded")
    action = strategy.get("fallback_action", "skip")

    result.degradations.append(DegradationRecord(
        node=node_name,
        reason=f"节点降级: {action}",
        level=strategy.get("degrade_level", 2),
        action=action,
        timestamp=datetime.now(),
    ))

    # 第五层：兜底标记
    if strategy.get("degrade_level", 2) >= 4:
        result.fallback_triggered = True

    return result
