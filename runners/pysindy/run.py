# -*- coding: utf-8 -*-
# runners/pysindy/run.py
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict

from runners._base.configio import deep_merge, load_json_override, load_yaml
from runners._base.dataio import load_system_spec, resolve_data_path, load_X, validate_X

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)     # pysindy
    p.add_argument("--system", required=True)
    p.add_argument("--case", required=True)       # "01"
    p.add_argument("--dataset", required=True)    # "01"
    p.add_argument("--data_root", default="data/systems")

    p.add_argument("--out", required=True)
    p.add_argument("--coef_out", required=True)

    # benchmark controls (长任务友好默认)
    p.add_argument("--rep", type=int, default=1)
    p.add_argument("--warmup", type=int, default=0)

    # dt：你已决定从参数给；这里仍允许为空时回退 system.yaml 的 dt
    p.add_argument("--dt", type=float, default=None)

    # 可选：临时覆盖任意配置（JSON）
    p.add_argument("--config_override_json", default="")

    return p.parse_args()

def ensure_parent(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def write_jsonl(path: str, obj: Dict[str, Any]):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def get_deps() -> Dict[str, str]:
    deps: Dict[str, str] = {}
    try:
        import numpy as np
        deps["numpy"] = np.__version__
    except Exception:
        pass
    try:
        import pysindy as ps
        deps["pysindy"] = ps.__version__
    except Exception:
        pass
    return deps

def build_model(cfg: Dict[str, Any]):
    import pysindy as ps

    # differentiation
    diff_cfg = cfg.get("differentiation", {}) or {}
    diff_type = str(diff_cfg.get("type", "finite_difference"))
    if diff_type == "finite_difference":
        diff = ps.FiniteDifference()
    elif diff_type == "smoothed_finite_difference":
        alpha = float(diff_cfg.get("alpha", 0.0))
        diff = ps.SmoothedFiniteDifference(alpha=alpha)
    else:
        raise ValueError(f"Unknown differentiation.type: {diff_type}")

    # library
    lib_cfg = cfg.get("library", {}) or {}
    lib_type = str(lib_cfg.get("type", "polynomial"))
    if lib_type != "polynomial":
        raise ValueError(f"Minimal runner only implements polynomial library, got: {lib_type}")
    degree = int(lib_cfg.get("degree", 3))
    include_bias = bool(lib_cfg.get("include_bias", False))
    try:
        library = ps.PolynomialLibrary(degree=degree, include_bias=include_bias)
    except TypeError:
        library = ps.PolynomialLibrary(degree=degree)

    # optimizer
    opt_cfg = cfg.get("optimizer", {}) or {}
    opt_type = str(opt_cfg.get("type", "stlsq"))
    if opt_type != "stlsq":
        raise ValueError(f"Minimal runner only implements STLSQ, got: {opt_type}")
    threshold = float(opt_cfg.get("threshold", 0.1))
    optimizer = ps.STLSQ(threshold=threshold)

    model = ps.SINDy(
        feature_library=library,
        optimizer=optimizer,
        differentiation_method=diff,
    )
    return model

def extract_coef(model):
    if hasattr(model, "coefficients"):
        return model.coefficients()
    if hasattr(model, "model") and hasattr(model.model, "coef_"):
        return model.model.coef_
    raise AttributeError("Cannot find coefficients on this PySINDy model/version")

def main():
    args = parse_args()

    run_id = os.environ.get("RUN_ID", "")
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")

    system_dir = os.path.join(args.data_root, args.system)
    spec = load_system_spec(system_dir)
    data_path = resolve_data_path(args.data_root, args.system, args.case, args.dataset)

    # 读取 per-system per-method 配置：data/systems/<system>/configs/pysindy.yaml
    method_cfg_path = os.path.join(system_dir, "configs", f"{args.method}.yaml")
    method_cfg = load_yaml(method_cfg_path)

    # 组装 cfg：code defaults -> method_cfg -> CLI override(json)
    code_defaults = {
        "library": {"type": "polynomial", "degree": 3, "include_bias": False},
        "optimizer": {"type": "stlsq", "threshold": 0.1},
        "differentiation": {"type": "finite_difference"},
    }
    cfg = deep_merge(code_defaults, method_cfg)
    cfg = deep_merge(cfg, load_json_override(args.config_override_json))

    # dt：CLI 优先，其次 system.yaml dt
    dt = args.dt if args.dt is not None else spec.dt_default
    if dt is None:
        raise ValueError("dt is required: provide --dt or set dt in system.yaml")
    dt = float(dt)
    cfg["dt"] = dt

    record: Dict[str, Any] = {
        "run_id": run_id,
        "method": args.method,
        "env": conda_env,
        "system": args.system,
        "case_id": args.case,
        "dataset_id": args.dataset,
        "data_path": data_path,
        "method_cfg_path": method_cfg_path,
        "cfg": cfg,  # 最小版本直接写进去，便于复现（后续可改为hash）

        "rep": args.rep,
        "warmup": args.warmup,
        "dt_used": dt,

        "t_init_ns": [],
        "t_fit_ns": [],
        "ok": False,
        "error_type": None,
        "error_msg": None,

        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "python_version": sys.version.split()[0],
        "deps": get_deps(),
    }

    try:
        # load 不计时
        X = load_X(data_path, spec)
        validate_X(X, spec)
        record["X_shape"] = list(X.shape)

        # warmup（不计入 t_fit_ns）
        for _ in range(args.warmup):
            m = build_model(cfg)
            m.fit(X, t=dt)
            _ = extract_coef(m)

        # rep（计时）
        last_coef = None
        for _ in range(args.rep):
            t0 = time.perf_counter_ns()
            m = build_model(cfg)
            t1 = time.perf_counter_ns()
            m.fit(X, t=dt)
            coef = extract_coef(m)
            t2 = time.perf_counter_ns()

            record["t_init_ns"].append(t1 - t0)
            record["t_fit_ns"].append(t2 - t1)
            last_coef = coef

        import numpy as np
        ensure_parent(args.coef_out)
        np.save(args.coef_out, last_coef)

        c = np.asarray(last_coef)
        record["coef_shape"] = list(c.shape)
        record["coef_l1"] = float(np.sum(np.abs(c)))
        record["coef_l2"] = float(np.sqrt(np.sum(c * c)))

        record["ok"] = True
        write_jsonl(args.out, record)
        return 0

    except Exception as e:
        record["ok"] = False
        record["error_type"] = e.__class__.__name__
        record["error_msg"] = str(e)
        record["traceback"] = traceback.format_exc()
        write_jsonl(args.out, record)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
