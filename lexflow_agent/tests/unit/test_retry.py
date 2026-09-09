"""单元测试: 重试装饰器"""

import pytest
from lexflow_agent.engine.resilience.retry import with_retry, RetryableError, NonRetryableError


class TestRetryDecorator:
    def test_success_no_retry(self):
        call_count = 0

        @with_retry(max_retries=3)
        def func():
            nonlocal call_count
            call_count += 1
            return "ok"

        assert func() == "ok"
        assert call_count == 1

    def test_retry_then_success(self):
        call_count = 0

        @with_retry(max_retries=3)
        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RetryableError("retry")
            return "ok"

        assert func() == "ok"
        assert call_count == 3

    def test_exhaust_retries(self):
        call_count = 0

        @with_retry(max_retries=2)
        def func():
            nonlocal call_count
            call_count += 1
            raise RetryableError("always fail")

        with pytest.raises(RetryableError):
            func()
        assert call_count == 3  # 初始 + 2次重试

    def test_non_retryable_raises_immediately(self):
        call_count = 0

        @with_retry(max_retries=3)
        def func():
            nonlocal call_count
            call_count += 1
            raise NonRetryableError("bad request")

        with pytest.raises(NonRetryableError):
            func()
        assert call_count == 1
