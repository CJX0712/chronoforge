"""Pipeline：把数据生成 → 基准 → 确定性校验串成可复现一键流程。"""

from __future__ import annotations

import json
import os

from ..core.seed import set_all
from ..data.synthetic import REGIMES, generate_dataset
from ..eval.benchmark import run_benchmark


def run(
    seed: int = 12345,
    horizon: int = 12,
    regimes=None,
    seeds=(7, 42, 123, 2024, 99),
    length: int = 120,
    period: int = 12,
    noise_sigma: float = 1.0,
    n_origins: int | None = None,
    ridge_alpha: float | None = None,
    safeguard: bool | None = None,
):
    set_all(seed)
    dataset = generate_dataset(
        regimes=regimes or list(REGIMES),
        seeds=seeds,
        length=length,
        period=period,
        noise_sigma=noise_sigma,
    )
    result = run_benchmark(
        dataset,
        horizon=horizon,
        n_origins=n_origins,
        ridge_alpha=ridge_alpha,
        safeguard=safeguard,
    )
    return result, dataset


def determinism_check(**kwargs) -> dict:
    """同 seed 两次运行，核对核心指标逐位一致（elapsed_sec 除外）。"""
    r1, _ = run(**kwargs)
    r2, _ = run(**kwargs)
    keys = ["chrono_smape", "strongest_single_smape", "chrono_vs_strongest_smape_reduction_pct"]
    diffs = {}
    for k in keys:
        a, b = r1.get(k), r2.get(k)
        if a is None or b is None:
            continue
        diffs[k] = abs(float(a) - float(b))
    # 各方法 sMAPE 一致性
    m_a = r1["methods"]
    m_b = r2["methods"]
    max_method_diff = 0.0
    for m in m_a:
        if m in m_b:
            max_method_diff = max(max_method_diff, abs(m_a[m]["smape_mean"] - m_b[m]["smape_mean"]))
    diffs["max_method_smape_diff"] = max_method_diff
    return {
        "diffs": diffs,
        "bit_identical": all(v == 0.0 for v in diffs.values()),
        "max_abs_diff": max(diffs.values()) if diffs else 0.0,
    }


def save_report(result: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
