"""YAML 模型路由与降级策略配置加载"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).parent


def _load_yaml(name: str) -> dict[str, Any]:
    path = _CONFIG_DIR / name
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_models_config() -> dict[str, Any]:
    return _load_yaml("models.yaml")


def load_degradation_config() -> dict[str, Any]:
    return _load_yaml("degradation.yaml")


def load_runtime_config() -> dict[str, Any]:
    return _load_yaml("runtime.yaml")


def load_pipelines_config() -> dict[str, Any]:
    return _load_yaml("pipelines.yaml")
