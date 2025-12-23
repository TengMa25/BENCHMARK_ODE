# -*- coding: utf-8 -*-
# runners/_base/dataio.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from runners._base.configio import load_yaml

@dataclass
class SystemSpec:
    name: str
    data_format: str = "csv"
    has_header: bool = False
    delimiter: str = ","
    x_columns: Optional[List[int]] = None
    dims: Optional[int] = None
    dt_default: Optional[float] = None

def load_system_spec(system_dir: str) -> SystemSpec:
    cfg = load_yaml(os.path.join(system_dir, "system.yaml"))
    return SystemSpec(
        name=str(cfg.get("name", os.path.basename(system_dir))),
        data_format=str(cfg.get("data_format", "csv")),
        has_header=bool(cfg.get("has_header", False)),
        delimiter=str(cfg.get("delimiter", ",")),
        x_columns=cfg.get("x_columns", None),
        dims=cfg.get("dims", None),
        dt_default=cfg.get("dt", None),
    )

def resolve_data_path(data_root: str, system: str, case_id: str, dataset_id: str) -> str:
    base = os.path.join(data_root, system, f"case_{case_id}", f"ds_{dataset_id}")
    for ext in (".npz", ".npy", ".csv"):
        p = base + ext
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Dataset not found: {base}(.npz/.npy/.csv)")

def load_X(path: str, spec: SystemSpec):
    import numpy as np

    if path.endswith(".npy"):
        X = np.load(path)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=False)
        X = z["X"] if "X" in z else z[list(z.keys())[0]]
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    if path.endswith(".csv"):
        skip = 1 if spec.has_header else 0
        X = np.loadtxt(path, delimiter=spec.delimiter, skiprows=skip)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if spec.x_columns is not None:
            X = X[:, spec.x_columns]
        return X

    raise ValueError(f"Unsupported data format: {path}")

def validate_X(X, spec: SystemSpec):
    if spec.dims is not None and X.shape[1] != spec.dims:
        raise ValueError(f"dims mismatch: spec.dims={spec.dims} but X.shape={X.shape}")
