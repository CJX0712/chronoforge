"""朴素法：Naive（末值外推）与 SeasonalNaive（同季节末值外推）。"""

from __future__ import annotations

import numpy as np

from ..core.errors import FitError
from ..core.types import Forecast, TimeSeries
from .base import BaseForecaster


class NaiveForecaster(BaseForecaster):
    name = "naive"

    def fit(self, series: TimeSeries, horizon: int) -> NaiveForecaster:
        if horizon < 1:
            raise FitError("horizon 须>=1")
        self._last = float(series.values[-1])
        self._horizon = horizon
        self._fitted = True
        return self

    def forecast(self) -> Forecast:
        self._ensure_fitted()
        return Forecast(mean=np.full(self._horizon, self._last, dtype=float), method=self.name)


class SeasonalNaiveForecaster(BaseForecaster):
    name = "seasonal_naive"

    def fit(self, series: TimeSeries, horizon: int) -> SeasonalNaiveForecaster:
        if horizon < 1:
            raise FitError("horizon 须>=1")
        m = max(2, int(series.period))
        y = series.values
        if len(y) < m:
            raise FitError("seasonal_naive 需要长度>=period")
        self._last_season = y[-m:]
        self._horizon = horizon
        self._fitted = True
        return self

    def forecast(self) -> Forecast:
        self._ensure_fitted()
        m = len(self._last_season)
        out = np.array([self._last_season[i % m] for i in range(self._horizon)], dtype=float)
        return Forecast(mean=out, method=self.name)
