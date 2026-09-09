# 模块文档字符串：LLM 结构化输出辅助 - 统一所有节点的结构化调用 + Trace 写入 + LangSmith 关联
"""LLM 结构化输出辅助 - 统一所有节点的结构化调用 + Trace 写入 + LangSmith 关联"""

# 启用 Python 3.10+ 的新特性支持
from __future__ import annotations

# 导入时间模块，用于计算耗时
import time
# 导入类型提示工具
from typing import Any, Optional, Type

# 导入 Pydantic 基础模型
from pydantic import BaseModel

# 导入 LLM 工厂
from lexflow_agent.engine.gateway.factory import LLMFactory
# 导入弹性执行器（降级和重试）
from lexflow_agent.engine.resilience.fallback import resilience_execute
# 导入可重试错误
from lexflow_agent.engine.resilience.retry import RetryableError
# 导入节点追踪模型
from lexflow_agent.engine.models.response import NodeTrace
# 导入降级记录模型
from lexflow_agent.engine.models.review import DegradationRecord
# 导入 Prompt 管理器
from lexflow_agent.engine.prompts.manager import prompt_manager
# 导入 LangSmith traceable 装饰器
from lexflow_agent.engine.tracing import trace_node


# LLM 结构化调用函数
@trace_node("llm_structured_call", run_type="llm")
def llm_structured_call(
    node_name: str,                              # 节点名称
    messages: list[dict],                        # 消息列表
    schema: Type[BaseModel],                     # 输出模式
    state: Any,                                  # 案件状态
    fallback_data: Any = None,                   # 降级数据
    temperature: float = 0.1,                    # 温度参数
) -> tuple[Optional[BaseModel], bool, list[DegradationRecord], NodeTrace]:
    # 记录开始时间
    start_time = time.time()
    # 初始化追踪对象
    trace = NodeTrace(node=node_name, status="success")
    # 初始化降级记录列表
    degs: list[DegradationRecord] = []

    # 关联 Prompt 版本到 trace
    pv = prompt_manager.versions
    prompt_version = pv.get(node_name, "unknown")
    trace.prompt_version = prompt_version

    # 定义 LLM 调用函数
    def _call(client):
        trace.model = client.model_name          # 记录使用的模型
        structured = client.with_structured_output(schema)  # 获取结构化输出包装器
        result = structured(messages, temperature=temperature)  # 调用 LLM
        if result is None:
            raise RetryableError("LLM 返回空结构化结果")  # 如果结果为空，抛出可重试错误
        return result

    # 执行弹性调用（包含重试和降级）
    exec_result = resilience_execute(
        node_name=node_name,
        llm_call=_call,
        state=state,
        fallback_data=fallback_data,
    )

    # 收集降级记录
    if hasattr(exec_result, 'degradations'):
        degs = exec_result.degradations
    # 如果发生降级，更新追踪状态
    if exec_result.degraded:
        trace.status = "degraded"
    # 记录耗时
    trace.duration_ms = (time.time() - start_time) * 1000
    # 记录重试次数
    trace.retry_count = getattr(exec_result, 'retry_count', 0)

    # 返回结果元组
    return (
        exec_result.data if exec_result.success else fallback_data,  # 数据
        exec_result.degraded,                    # 是否降级
        degs,                                    # 降级记录
        trace,                                   # 追踪信息
    )


# LLM 对话调用函数（非结构化输出）
@trace_node("llm_chat_call", run_type="llm")
def llm_chat_call(
    node_name: str,                              # 节点名称
    messages: list[dict],                        # 消息列表
    state: Any,                                  # 案件状态
    fallback_text: str = "",                     # 降级文本
    temperature: float = 0.1,                    # 温度参数
) -> tuple[str, bool, list[DegradationRecord], NodeTrace]:
    # 记录开始时间
    start_time = time.time()
    # 初始化追踪对象
    trace = NodeTrace(node=node_name, status="success")
    # 初始化降级记录列表
    degs: list[DegradationRecord] = []

    # 关联 Prompt 版本到 trace
    pv = prompt_manager.versions
    prompt_version = pv.get(node_name, "unknown")
    trace.prompt_version = prompt_version

    # 定义 LLM 调用函数
    def _call(client):
        trace.model = client.model_name          # 记录使用的模型
        return client.chat(messages, temperature=temperature)  # 调用对话 API

    # 执行弹性调用（包含重试和降级）
    exec_result = resilience_execute(
        node_name=node_name,
        llm_call=_call,
        state=state,
        fallback_data=fallback_text,
    )

    # 收集降级记录
    if hasattr(exec_result, 'degradations'):
        degs = exec_result.degradations
    # 如果发生降级，更新追踪状态
    if exec_result.degraded:
        trace.status = "degraded"
    # 记录耗时
    trace.duration_ms = (time.time() - start_time) * 1000
    # 记录重试次数
    trace.retry_count = getattr(exec_result, 'retry_count', 0)

    # 返回结果元组
    return (
        str(exec_result.data) if exec_result.data else fallback_text,  # 文本
        exec_result.degraded,                    # 是否降级
        degs,                                    # 降级记录
        trace,                                   # 追踪信息
    )