"""Theta 方法（Assimakopoulos 2000 经典分解），纯 numpy 实现。

θ=0 线用 SES，θ=2 线去线性趋势后 SES 再外推趋势，两线等权融合。
作为强可解释基线，参与 ChronoFuse 组合。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import FitError
from ..core.types import Forecast, TimeSeries
from .base import BaseForecaster
from .exp_smoothing import _ses_best


class ThetaForecaster(BaseForecaster):
    name = "theta"

    def fit(self, series: TimeSeries, horizon: int) -> ThetaForecaster:
        if horizon < 1:
            raise FitError("horizon 须>=1")
        y = np.asarray(series.values, dtype=float)
        n = len(y)
        t = np.arange(n, dtype=float)
        b = float(np.polyfit(t, y, 1)[0])  # 线性趋势斜率
        z2 = y - b * t
        _, self._level0 = _ses_best(y)
        _, self._level2 = _ses_best(z2)
        self._b = b
        self._n = n
        self._horizon = horizon
        self._fitted = True
        return self

    def forecast(self) -> Forecast:
        self._ensure_fitted()
        h = self._horizon
        k = np.arange(1, h + 1, dtype=float)
        fc2 = self._level2 + self._b * (self._n - 1 + k)
        out = 0.5 * self._level0 + 0.5 * fc2
        return Forecast(mean=out, method=self.name)
