"""接口契约：Forecaster 与 Combiner 的 Protocol。

模块只依赖这些抽象，保证可插拔与单测可独立验证。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .types import Forecast, TimeSeries


@runtime_checkable
class Forecaster(Protocol):
    """单条序列的 h 步预测器。"""

    name: str

    def fit(self, series: TimeSeries, horizon: int) -> Forecaster:
        """在 train 部分拟合并记住 horizon。返回 self。"""
        ...

    def forecast(self) -> Forecast:
        """输出长度=horizon 的预测。"""
        ...


@runtime_checkable
class Combiner(Protocol):
    """把多个基预测器输出融合为单一预测。"""

    name: str

    def fit(self, base_forecasts: np.ndarray, actuals: np.ndarray) -> Combiner:
        """X: (n_samples, n_bases, horizon); y: (n_samples, horizon)。"""
        ...

    def combine(self, base_forecasts: np.ndarray) -> np.ndarray:
        """输入 (n_bases, horizon) -> (horizon,)。"""
        ...
