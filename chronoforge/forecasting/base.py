"""基预测器公共工具与基类。"""

from __future__ import annotations

import numpy as np

from ..core.errors import FitError
from ..core.types import Forecast, TimeSeries


def last_value(series: TimeSeries) -> float:
    return float(series.values[-1])


def train_test_split(values: np.ndarray, horizon: int):
    """切分：train = 前 n-h，test = 末 h。仅在 pipeline 层调用，保证无泄漏。"""
    if horizon < 1 or horizon >= len(values):
        raise FitError("horizon 须 1<=h<len(values)")
    return values[: len(values) - horizon], values[len(values) - horizon :]


class BaseForecaster:
    """简单基类：统一 name / fit / forecast 约定。"""

    name = "base"

    def __init__(self) -> None:
        self._fitted = False
        self._horizon = 0

    def fit(self, series: TimeSeries, horizon: int) -> BaseForecaster:
        raise NotImplementedError

    def _ensure_fitted(self) -> None:
        if not self._fitted:
            raise FitError(f"{self.name} 未 fit 即 forecast")

    def forecast(self) -> Forecast:
        self._ensure_fitted()
        raise NotImplementedError
