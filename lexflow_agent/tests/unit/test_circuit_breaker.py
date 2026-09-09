"""单元测试: 熔断器"""

import time
from lexflow_agent.engine.resilience.circuit_breaker import CircuitBreaker


class TestCircuitBreaker:
    def test_initial_state(self):
        cb = CircuitBreaker(window_seconds=60, failure_threshold=0.5, cooldown_seconds=1)
        assert cb.state == "closed"
        assert not cb.is_open()

    def test_stays_closed_with_few_samples(self):
        cb = CircuitBreaker(window_seconds=60, failure_threshold=0.5, cooldown_seconds=1)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert not cb.should_open()  # 样本不足5个

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(window_seconds=60, failure_threshold=0.5, cooldown_seconds=1)
        for _ in range(6):
            cb.record_failure()
        assert cb.should_open()
        cb.open()
        assert cb.is_open()

    def test_closes_after_cooldown(self):
        cb = CircuitBreaker(window_seconds=60, failure_threshold=0.5, cooldown_seconds=1)
        for _ in range(6):
            cb.record_failure()
        cb.open()
        assert cb.is_open()
        time.sleep(1.1)
        # 冷却后，half_open 状态放行
        assert cb.state == "open"  # 还没触发 half_open
        cb.is_open()  # 触发 half_open 检查
        assert cb.state == "half_open"

    def test_record_success_closes(self):
        cb = CircuitBreaker(window_seconds=60, failure_threshold=0.5, cooldown_seconds=1)
        cb.record_success()
        cb.close()
        assert cb.state == "closed"
