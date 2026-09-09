# 模块文档字符串：LLM 工厂，统一管理模型创建、缓存与路由
"""LLM 工厂 - 统一管理模型创建、缓存与路由"""

# 启用 Python 3.10+ 的新特性支持
from __future__ import annotations

# 导入操作系统模块，用于环境变量读取
import os
# 导入类型提示工具
from typing import Any, Optional, Type

# 导入 OpenAI 客户端
from openai import OpenAI
# 导入 Pydantic 基础模型类
from pydantic import BaseModel
# 导入 LangSmith 包装器，用于追踪 LLM 调用
from langsmith.wrappers import wrap_openai

# 导入项目配置设置
from lexflow_agent.config.settings import settings
# 导入模型配置加载器
from lexflow_agent.config.config_loader import load_models_config


# 定义 LLM 客户端封装类
class LLMClient:
    """LLM 客户端封装"""

    # 初始化 LLM 客户端
    def __init__(
        self,
        model_name: str,           # 模型名称
        base_url: str,             # API 基础 URL
        api_key: str,              # API 密钥
        temperature: float = 0.1,  # 温度参数，控制输出随机性
        max_tokens: int = 4096,    # 最大生成 token 数
        timeout: int = 60,         # 请求超时时间（秒）
        supports_function_calling: bool = True,  # 是否支持函数调用
        supports_json_mode: bool = False,        # 是否支持 JSON 模式
    ):
        # 保存模型名称
        self.model_name = model_name
        # 保存温度参数
        self.temperature = temperature
        # 保存最大 token 数
        self.max_tokens = max_tokens
        # 保存超时时间
        self.timeout = timeout
        # 保存是否支持函数调用
        self.supports_function_calling = supports_function_calling
        # 保存是否支持 JSON 模式
        self.supports_json_mode = supports_json_mode

        # 创建 OpenAI 客户端，绕过系统代理
        self.client = wrap_openai(OpenAI(
            base_url=base_url,              # API 基础 URL
            api_key=api_key,                # API 密钥
            http_client=None,               # 使用默认 httpx 客户端
            timeout=timeout,                # 超时时间
        ))

    # 基础对话方法
    def chat(self, messages: list[dict], **kwargs) -> str:
        """基础对话"""
        # 调用 OpenAI API 进行对话
        resp = self.client.chat.completions.create(
            model=self.model_name,          # 使用的模型
            messages=messages,              # 消息列表
            temperature=kwargs.get("temperature", self.temperature),  # 温度参数
            max_tokens=kwargs.get("max_tokens", self.max_tokens),     # 最大 token 数
        )
        # 返回模型响应内容
        return resp.choices[0].message.content or ""

    # 返回结构化输出的包装器
    def with_structured_output(self, schema: Type[BaseModel]) -> Optional[BaseModel]:
        """返回一个 callable，传入 messages 即可获得结构化输出"""
        # 如果支持函数调用，使用函数调用方式
        if self.supports_function_calling:
            return _FunctionCallingWrapper(self, schema)
        # 如果支持 JSON 模式，使用 JSON 模式
        elif self.supports_json_mode:
            return _JSONModeWrapper(self, schema)
        # 否则使用纯 Prompt 约束方式
        else:
            return _PromptBasedWrapper(self, schema)


# 函数调用方式的结构化输出包装器
class _FunctionCallingWrapper:
    """Function Calling 结构化输出 - 使用 tools 参数而非 beta.chat.parse"""

    # 初始化包装器
    def __init__(self, client: LLMClient, schema: Type[BaseModel]):
        # 保存 LLM 客户端
        self.client = client
        # 保存输出模式
        self.schema = schema

    # 调用方法，传入消息返回结构化输出
    def __call__(self, messages: list[dict], **kwargs) -> Optional[BaseModel]:
        try:
            # 导入 JSON 模块
            import json
            # 获取模式的 JSON Schema 定义
            schema_def = self.schema.model_json_schema()
            # 生成工具名称
            tool_name = f"output_{self.schema.__name__}"

            # 调用 OpenAI API，使用 tools 参数
            resp = self.client.client.chat.completions.create(
                model=self.client.model_name,   # 模型名称
                messages=messages,              # 消息列表
                tools=[{                        # 工具定义
                    "type": "function",         # 工具类型
                    "function": {               # 函数定义
                        "name": tool_name,      # 函数名称
                        "description": f"Output structured {self.schema.__name__}",  # 函数描述
                        "parameters": schema_def,  # 函数参数 Schema
                    },
                }],
                max_tokens=kwargs.get("max_tokens", self.client.max_tokens),  # 最大 token 数
            )
            # 获取响应消息
            msg = resp.choices[0].message
            # 如果模型触发了工具调用
            if msg.tool_calls:
                # 解析工具调用的参数
                args = json.loads(msg.tool_calls[0].function.arguments)
                # 验证并返回结构化输出
                return self.schema.model_validate(args)
            # 如果 LLM 没有触发 tool call，尝试从 content 提取 JSON
            if msg.content:
                # 导入正则表达式模块
                import re
                # 使用正则表达式匹配 JSON 对象
                json_match = re.search(r"\{.*\}", msg.content, re.DOTALL)
                # 如果匹配到 JSON，解析并返回
                if json_match:
                    return self.schema.model_validate_json(json_match.group())
            # 无法提取结构化输出，返回 None
            return None
        except Exception:
            # 发生异常时返回 None
            return None


# JSON 模式的结构化输出包装器
class _JSONModeWrapper:
    """JSON mode 结构化输出"""

    # 初始化包装器
    def __init__(self, client: LLMClient, schema: Type[BaseModel]):
        # 保存 LLM 客户端
        self.client = client
        # 保存输出模式
        self.schema = schema

    # 调用方法，传入消息返回结构化输出
    def __call__(self, messages: list[dict], **kwargs) -> Optional[BaseModel]:
        try:
            # 添加 JSON 格式指导到消息列表
            messages_with_guide = messages + [{
                "role": "system",  # 系统角色
                "content": f"你必须输出合法的 JSON，Schema: {self.schema.model_json_schema()}"  # 指导内容
            }]
            # 调用 OpenAI API，使用 JSON 模式
            resp = self.client.client.chat.completions.create(
                model=self.client.model_name,           # 模型名称
                messages=messages_with_guide,           # 消息列表
                response_format={"type": "json_object"},  # JSON 对象格式
                temperature=kwargs.get("temperature", self.client.temperature),  # 温度参数
                max_tokens=kwargs.get("max_tokens", self.client.max_tokens),     # 最大 token 数
            )
            # 获取响应内容
            content = resp.choices[0].message.content or "{}"
            # 验证并返回结构化输出
            return self.schema.model_validate_json(content)
        except Exception:
            # 发生异常时返回 None
            return None


# 纯 Prompt 约束的结构化输出包装器
class _PromptBasedWrapper:
    """纯 Prompt 约束 + 正则提取"""

    # 初始化包装器
    def __init__(self, client: LLMClient, schema: Type[BaseModel]):
        # 保存 LLM 客户端
        self.client = client
        # 保存输出模式
        self.schema = schema

    # 调用方法，传入消息返回结构化输出
    def __call__(self, messages: list[dict], **kwargs) -> Optional[BaseModel]:
        try:
            # 导入 JSON 模块
            import json
            # 将 Schema 转为 JSON 字符串
            schema_json = json.dumps(self.schema.model_json_schema(), ensure_ascii=False)
            # 构建增强的消息列表，添加 JSON 约束到 system prompt
            enhanced = [
                {"role": "system", "content": f"你必须输出严格的 JSON，严格符合以下 Schema，不要包含其他文字：\n{schema_json}"},
            ] + messages
            # 调用 OpenAI API
            resp = self.client.client.chat.completions.create(
                model=self.client.model_name,           # 模型名称
                messages=enhanced,                      # 增强的消息列表
                temperature=kwargs.get("temperature", self.client.temperature),  # 温度参数
                max_tokens=kwargs.get("max_tokens", self.client.max_tokens),     # 最大 token 数
            )
            # 获取响应内容
            content = resp.choices[0].message.content or "{}"
            # 导入正则表达式模块
            import re
            # 使用正则表达式匹配 JSON 对象
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            # 如果匹配到 JSON，解析并返回
            if json_match:
                return self.schema.model_validate_json(json_match.group())
            # 无法提取 JSON，返回 None
            return None
        except Exception:
            # 发生异常时返回 None
            return None


# 定义 LLM 工厂类，使用单例模式管理客户端实例
class LLMFactory:
    """LLM 工厂 - 单例模式"""

    # 存储已创建的客户端实例
    _instances: dict[str, LLMClient] = {}
    # 存储配置信息
    _config: dict = {}

    # 初始化方法，加载模型配置
    @classmethod
    def initialize(cls):
        cls._config = load_models_config()

    # 获取主模型客户端
    @classmethod
    def get_client(cls, task_type: str = "primary") -> LLMClient:
        """获取主模型客户端"""
        # 生成实例键
        key = f"primary_{task_type}"
        # 如果实例不存在，创建新实例
        if key not in cls._instances:
            # 意图分类任务不使用 LLM
            if task_type == "intent_classification":
                raise ValueError("intent_classification 不应使用 LLM")
            # 创建主模型客户端
            cls._instances[key] = cls._create_client("primary")
        # 返回缓存的实例
        return cls._instances[key]

    # 获取备用模型客户端
    @classmethod
    def get_backup_client(cls) -> LLMClient:
        """获取备用模型客户端"""
        # 生成实例键
        key = "backup_client"
        # 如果实例不存在，创建新实例
        if key not in cls._instances:
            cls._instances[key] = cls._create_client("backup")
        # 返回缓存的实例
        return cls._instances[key]

    # 创建客户端的内部方法
    @classmethod
    def _create_client(cls, role: str) -> LLMClient:
        # 如果是主模型角色
        if role == "primary":
            return LLMClient(
                model_name=settings.LLM_PRIMARY_MODEL,                    # 主模型名称
                base_url=settings.LLM_PRIMARY_BASE_URL,                   # 主模型 API URL
                api_key=settings.LLM_PRIMARY_API_KEY,                     # 主模型 API 密钥
                temperature=settings.LLM_TEMPERATURE,                     # 温度参数
                max_tokens=settings.LLM_MAX_TOKENS,                       # 最大 token 数
                timeout=settings.LLM_TIMEOUT,                             # 超时时间
                supports_function_calling=True,                           # 支持函数调用
                supports_json_mode=False,                                 # 不支持 JSON 模式
            )
        # 如果是备用模型角色
        else:
            return LLMClient(
                model_name=settings.LLM_BACKUP_MODEL,                     # 备用模型名称
                base_url=settings.LLM_BACKUP_BASE_URL or settings.LLM_PRIMARY_BASE_URL,  # 备用 API URL
                api_key=settings.LLM_BACKUP_API_KEY or settings.LLM_PRIMARY_API_KEY,     # 备用 API 密钥
                temperature=settings.LLM_TEMPERATURE,                     # 温度参数
                max_tokens=settings.LLM_MAX_TOKENS,                       # 最大 token 数
                timeout=settings.LLM_TIMEOUT,                             # 超时时间
                supports_function_calling=True,                           # 支持函数调用
                supports_json_mode=True,                                  # 支持 JSON 模式
            )

    # 获取结构化输出 callable
    @classmethod
    def with_structured_output(cls, schema: Type[BaseModel], task_type: str = "primary"):
        """获取结构化输出 callable"""
        # 获取客户端
        client = cls.get_client(task_type)
        # 返回结构化输出包装器
        return client.with_structured_output(schema)

    # 重置所有实例（用于测试）
    @classmethod
    def reset(cls):
        """测试用：重置所有实例"""
        cls._instances.clear()