# -*- coding: utf-8 -*-

# runners/_base/configio.py
from __future__ import annotations
import json
import os
from typing import Any, Dict

def _simple_yaml_load(path: str) -> Dict[str, Any]:
    """
    降级用的极简 YAML 解析器：
    - 支持 key: value
    - 支持两层缩进（用于 library/optimizer/differentiation）
    - 支持 bool/int/float/string
    不支持复杂 YAML（足够跑通最小框架）
    """
    def parse_scalar(v: str) -> Any:
        v = v.strip()
        if v.lower() in ("true", "false"):
            return v.lower() == "true"
        try:
            if "." in v:
                return float(v)
            return int(v)
        except Exception:
            return v.strip('"').strip("'")

    root: Dict[str, Any] = {}
    stack = [(0, root)]  # (indent, dict)
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            indent = len(line) - len(line.lstrip(" "))
            if ":" not in s:
                continue
            k, v = s.split(":", 1)
            k = k.strip()
            v = v.strip()

            while stack and indent < stack[-1][0]:
                stack.pop()
            cur = stack[-1][1]

            if v == "":
                cur[k] = {}
                stack.append((indent + 2, cur[k]))
            else:
                cur[k] = parse_scalar(v)
    return root

def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f)
        return obj or {}
    except Exception:
        return _simple_yaml_load(path)

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归合并：dict 合并，其他类型直接覆盖。
    """
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_json_override(s: str) -> Dict[str, Any]:
    if not s:
        return {}
    return json.loads(s)
