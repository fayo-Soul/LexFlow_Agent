"""Prompt 版本管理 - 加载 + 注入幻觉基线 + 版本追踪 + LangSmith Prompt Hub 同步"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from lexflow_agent.config.settings import settings


class PromptManager:
    """Prompt 版本管理 - 支持本地文件 + LangSmith Prompt Hub"""

    def __init__(self, prompt_dir: Optional[str] = None):
        self._prompt_dir = Path(prompt_dir or (
            Path(__file__).parent.parent / "prompts"
        ))
        self._versions: dict[str, str] = {}
        self._hub_connected = False
        self._hub_client = None

        # 尝试连接 LangSmith Prompt Hub
        self._connect_hub()

    def _connect_hub(self):
        """连接到 LangSmith Prompt Hub"""
        if settings.LANGSMITH_PROMPT_HUB and settings.LANGSMITH_API_KEY:
            try:
                from langsmith import Client as LangSmithClient
                api_key = settings.LANGSMITH_API_KEY or os.environ.get("LANGSMITH_API_KEY")
                if api_key:
                    self._hub_client = LangSmithClient(
                        api_key=api_key,
                        api_url=settings.LANGSMITH_ENDPOINT,
                    )
                    self._hub_connected = True
            except Exception:
                self._hub_connected = False

    def load(self, agent_id: str, version: Optional[str] = None) -> str:
        """加载指定 agent 的 Prompt，优先从 LangSmith Hub 拉取

        优先级: LangSmith Hub > 本地文件 > 默认兜底
        """
        if version is None:
            version = self._get_latest_version(agent_id)

        # 尝试从 LangSmith Hub 拉取
        if self._hub_connected:
            hub_prompt = self._load_from_hub(agent_id, version)
            if hub_prompt:
                baseline = self._load_baseline()
                self._versions[agent_id] = f"hub:{version}"
                return f"{baseline}\n\n{hub_prompt}"

        # 从本地文件加载
        local_prompt = self._load_from_file(agent_id, version)
        if local_prompt:
            baseline = self._load_baseline()
            self._versions[agent_id] = version
            return f"{baseline}\n\n{local_prompt}"

        # 兜底
        self._versions[agent_id] = "unknown"
        return f"你是一个 {agent_id}。请根据输入完成任务。"

    def _load_from_hub(self, agent_id: str, version: str) -> Optional[str]:
        """从 LangSmith Hub 拉取 Prompt"""
        if not self._hub_client:
            return None
        try:
            hub_name = f"lexflow-{agent_id.replace('_', '-')}"
            prompt_obj = self._hub_client.pull_prompt(hub_name)
            value = prompt_obj.invoke({"input": ""})
            messages = value.to_messages()
            if messages:
                return str(messages[0].content)
        except Exception:
            pass

        # 降级尝试: 不加版本号
        try:
            prompt_obj = self._hub_client.pull_prompt(f"lexflow-{agent_id.replace('_', '-')}")
            value = prompt_obj.invoke({"input": ""})
            messages = value.to_messages()
            if messages:
                return str(messages[0].content)
        except Exception:
            pass

        return None

    def push_to_hub(self, agent_id: str, version: str, prompt_text: str) -> bool:
        """将 Prompt 推送到 LangSmith Hub

        用法:
            prompt_manager.push_to_hub("fact_agent", "v2.0.0", prompt_text)
        """
        if not self._hub_client:
            return False
        try:
            from langchain_core.prompts import ChatPromptTemplate
            hub_name = f"lexflow-{agent_id.replace('_', '-')}"
            prompt = ChatPromptTemplate.from_messages([
                ("system", prompt_text),
                ("human", "{input}"),
            ])
            self._hub_client.push_prompt(
                hub_name,
                object=prompt,
                description=f"LexFlow Agent - {agent_id} {version}",
                tags=["lexflow", "four-agent"],
                commit_description=version,
            )
            return True
        except Exception:
            return False

    def _load_from_file(self, agent_id: str, version: str) -> Optional[str]:
        """从本地文件加载 Prompt"""
        prompt_path = self._prompt_dir / f"{agent_id}_{version}.txt"
        if not prompt_path.exists():
            prompt_path = self._prompt_dir / f"{agent_id}.txt"
        if not prompt_path.exists():
            return None
        with open(prompt_path, encoding="utf-8") as f:
            return f.read()

    @property
    def versions(self) -> dict[str, str]:
        return dict(self._versions)

    @property
    def hub_connected(self) -> bool:
        return self._hub_connected

    def _load_baseline(self) -> str:
        path = self._prompt_dir / "base_hallucination_prompt.txt"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return f.read().strip()
        return ""

    def _get_latest_version(self, agent_id: str) -> str:
        prefix = f"{agent_id}_"
        candidates = []
        for f in self._prompt_dir.glob(f"{prefix}*.txt"):
            if f.name != "base_hallucination_prompt.txt":
                candidates.append(f)
        if candidates:
            candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            stem = candidates[0].stem
            return stem[len(prefix):] if prefix in stem else "v1.0.0"
        return "v1.0.0"


# 全局单例
prompt_manager = PromptManager()
