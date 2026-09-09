"""熔断器 - 滑动窗口计数

本模块实现了熔断器模式（Circuit Breaker Pattern），用于保护系统免受持续故障的影响。

熔断器有三种状态：
1. Closed（关闭）：正常状态，请求正常执行
2. Open（打开）：熔断状态，请求被快速失败，不执行实际调用
3. Half-Open（半开）：试探状态，允许少量请求通过以检测服务是否恢复

工作原理：
- 在滑动窗口内记录成功/失败请求
- 当失败率超过阈值时，熔断器打开
- 经过冷却时间后，熔断器进入半开状态
- 半开状态下如果请求成功，熔断器关闭；如果失败，继续打开
"""

from __future__ import annotations  # 启用未来类型注解

import time  # 时间处理
from collections import deque  # 双端队列（用于滑动窗口）

from lexflow_agent.config.settings import settings  # 全局配置


class CircuitBreaker:
    """滑动窗口熔断器 - 保护系统免受持续故障影响
    
    使用滑动窗口记录最近一段时间内的请求成功/失败情况，
    根据失败率决定是否打开熔断器。
    """

    def __init__(
        self,
        window_seconds: int = 60,  # 滑动窗口大小（秒）
        failure_threshold: float = 0.5,  # 失败率阈值（50%）
        cooldown_seconds: int = 30,  # 冷却时间（秒）
    ):
        """初始化熔断器
        
        Args:
            window_seconds: 滑动窗口大小（秒），默认 60 秒
            failure_threshold: 失败率阈值（0-1），默认 0.5（50%）
            cooldown_seconds: 熔断后冷却时间（秒），默认 30 秒
        """
        self.window = window_seconds or settings.CB_WINDOW_SECONDS  # 窗口大小
        self.threshold = failure_threshold or settings.CB_FAILURE_THRESHOLD  # 失败阈值
        self.cooldown = cooldown_seconds or settings.CB_COOLDOWN_SECONDS  # 冷却时间
        self._records: deque = deque()  # 滑动窗口记录：[(timestamp, success), ...]
        self._state: str = "closed"  # 初始状态：关闭（正常）
        self._last_change: float = time.time()  # 上次状态变化时间
        self._min_samples: int = 5  # 最小样本数（少于 5 个样本不触发熔断）

    def record_success(self):
        """记录成功请求 - 添加成功记录并修剪窗口"""
        self._records.append((time.time(), True))  # 添加当前时间的成功记录
        self._trim_window()  # 修剪过期记录

    def record_failure(self):
        """记录失败请求 - 添加失败记录并修剪窗口"""
        self._records.append((time.time(), False))  # 添加当前时间的失败记录
        self._trim_window()  # 修剪过期记录

    def should_open(self) -> bool:
        """判断是否应该打开熔断器
        
        检查滑动窗口内的失败率是否超过阈值：
        1. 样本数不足 _min_samples → 不打开（避免误判）
        2. 计算失败率 = 失败数 / 总数
        3. 失败率 >= threshold → 应该打开
        
        Returns:
            是否应该打开熔断器
        """
        if len(self._records) < self._min_samples:  # 样本不足
            return False
        failures = sum(1 for _, success in self._records if not success)  # 统计失败数
        return (failures / len(self._records)) >= self.threshold  # 失败率超过阈值

    def is_open(self) -> bool:
        """检查熔断器是否处于打开状态
        
        状态转换逻辑：
        - closed → 返回 False（正常状态）
        - open → 检查是否超过冷却时间，超过则转为 half_open
        - half_open → 返回 False（允许试探请求通过）
        
        Returns:
            熔断器是否打开（阻止请求）
        """
        if self._state == "closed":  # 关闭状态，不阻止
            return False
        if self._state == "open":  # 打开状态
            if time.time() - self._last_change >= self.cooldown:  # 超过冷却时间
                self._state = "half_open"  # 转为半开状态
                self._last_change = time.time()
            return True  # 仍然阻止
        # half_open - 放行试探请求
        return False  # 半开状态，允许请求通过

    def open(self):
        """手动打开熔断器 - 通常在检测到持续失败时调用"""
        self._state = "open"
        self._last_change = time.time()

    def close(self):
        """手动关闭熔断器 - 通常在服务恢复后调用"""
        self._state = "closed"
        self._records.clear()  # 清空历史记录
        self._last_change = time.time()

    @property
    def state(self) -> str:
        """获取当前熔断器状态
        
        Returns:
            状态字符串："closed"、"open" 或 "half_open"
        """
        return self._state

    def _trim_window(self):
        """修剪滑动窗口 - 移除超过窗口大小的过期记录"""
        now = time.time()
        # 从队列头部移除过期记录（时间戳早于窗口起点）
        while self._records and now - self._records[0][0] > self.window:
            self._records.popleft()


# 全局熔断器实例 - 单例模式，所有节点共享同一个熔断器
circuit_breaker = CircuitBreaker()