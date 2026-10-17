"""配置体系：YAML + _base_ 继承 + 点号路径覆盖，不引入重型框架。

- load_config(path)：解析 _base_（相对该文件所在 configs/ 目录），子键覆盖父键（dict 深合并）。
- apply_overrides(cfg, ["optim.lr=1e-3", "data.root=/x"])：点号路径写值，int/float 自动转换。
- snapshot(cfg, path)：把解析后的完整配置写盘，供 run 目录存档。
"""
import copy
import os

import yaml


def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path):
    path = os.path.abspath(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base is not None:
        base_path = os.path.join(os.path.dirname(path), base)
        cfg = _deep_merge(load_config(base_path), cfg)
    cfg["_config_path"] = path
    return cfg


def _coerce(v):
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def apply_overrides(cfg, overrides):
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        assert "=" in item, f"override must be key=value, got: {item}"
        key, value = item.split("=", 1)
        node = cfg
        keys = key.split(".")
        for k in keys[:-1]:
            assert k in node and isinstance(node[k], dict), f"unknown config path: {key}"
            node = node[k]
        node[keys[-1]] = _coerce(value)
    return cfg


def snapshot(cfg, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path
