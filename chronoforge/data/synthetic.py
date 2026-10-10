"""合成时间序列生成器。

覆盖 8 类难度递增的 DGP regime，每类可调噪声强度，构造「甜点」：
基线明显低于天花板，使 benchmark 有区分度（不会全满分也不会全崩）。
全部使用 RandomState(seed) 自行设种子，独立于全局 RNG，复现稳定。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import DataError
from ..core.types import TimeSeries

REGIMES = [
    "trend_linear",
    "trend_exp",
    "seasonal_add",
    "seasonal_mul",
    "trend_seasonal",
    "level_shift",
    "multiple_seasonal",
    "noisy_random_walk",
]


def _noise(rs: np.random.RandomState, n: int, sigma: float) -> np.ndarray:
    return rs.normal(0.0, sigma, size=n)


def generate_series(
    regime: str,
    seed: int,
    length: int = 120,
    period: int = 12,
    noise_sigma: float = 1.0,
) -> TimeSeries:
    """按 regime 生成一条合成序列。period 仅对季节性 regime 生效。"""
    if length < 2 * period:
        raise DataError("length 须 >= 2*period")
    rs = np.random.RandomState(seed)
    t = np.arange(length, dtype=float)
    m = period

    if regime == "trend_linear":
        a, b = 10.0, 0.25
        y = a + b * t + _noise(rs, length, noise_sigma)

    elif regime == "trend_exp":
        y = 5.0 * np.exp(0.012 * t) + np.abs(_noise(rs, length, noise_sigma)) + 1.0

    elif regime == "seasonal_add":
        level = 50.0
        y = level + 12.0 * np.sin(2 * np.pi * t / m) + _noise(rs, length, noise_sigma)

    elif regime == "seasonal_mul":
        level = 50.0
        y = (
            level * (1.0 + 0.3 * np.sin(2 * np.pi * t / m))
            + np.abs(_noise(rs, length, noise_sigma))
            + 1.0
        )

    elif regime == "trend_seasonal":
        level = 20.0 + 0.15 * t
        y = level + 10.0 * np.sin(2 * np.pi * t / m) + _noise(rs, length, noise_sigma)

    elif regime == "level_shift":
        y = np.empty(length)
        k = length // 2
        y[:k] = 30.0 + _noise(rs, k, noise_sigma * 0.6)
        y[k:] = 55.0 + _noise(rs, k, noise_sigma * 0.6)

    elif regime == "multiple_seasonal":
        level = 40.0
        s1 = 10.0 * np.sin(2 * np.pi * t / m)
        s2 = 4.0 * np.sin(2 * np.pi * t / 4)
        y = level + s1 + s2 + _noise(rs, length, noise_sigma)

    elif regime == "noisy_random_walk":
        shocks = _noise(rs, length, 1.0)
        y = 50.0 + np.cumsum(shocks)
        y = y + _noise(rs, length, noise_sigma * 0.3)

    else:
        raise DataError(f"未知 regime: {regime}")

    return TimeSeries(values=y, name=f"{regime}_s{seed}", period=m)


def generate_dataset(
    regimes=None,
    seeds=(7, 42, 123, 2024, 99),
    length: int = 120,
    period: int = 12,
    noise_sigma: float = 1.0,
) -> list[TimeSeries]:
    """生成完整基准数据集。返回 TimeSeries 列表（每条带 regime 元数据于 name）。"""
    if regimes is None:
        regimes = list(REGIMES)
    out: list[TimeSeries] = []
    for r in regimes:
        for s in seeds:
            out.append(generate_series(r, s, length, period, noise_sigma))
    return out
