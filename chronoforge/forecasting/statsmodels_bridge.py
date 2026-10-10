"""Tier-0 后端：statsmodels 的 ARIMA 与 ETS（Holt-Winters）。

缺失时 available_*()=False，benchmark 自动跳过并在报告中标注 skipped，不伪造数字。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import FitError
from ..core.types import Forecast, TimeSeries
from .base import BaseForecaster

try:  # pragma: no cover - 依赖可选
    from statsmodels.tsa.arima.model import ARIMA as _SM_ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as _SM_ETS

    _HAVE_SM = True
except Exception:  # noqa: BLE001
    _HAVE_SM = False


def available() -> bool:
    return _HAVE_SM


class ARIMAForecaster(BaseForecaster):
    name = "arima"

    def fit(self, series: TimeSeries, horizon: int) -> ARIMAForecaster:
        if not _HAVE_SM:
            raise FitError("statsmodels 不可用，ARIMA 后端缺失")
        if horizon < 1:
            raise FitError("horizon 须>=1")
        y = np.asarray(series.values, dtype=float)
        best = None
        for p in (1, 2, 3):
            try:
                res = _SM_ARIMA(y, order=(p, 1, 0)).fit()
            except Exception:  # noqa: BLE001
                continue
            aic = float(res.aic)
            if best is None or aic < best[2]:
                best = (p, res, aic)
        if best is None:
            raise FitError("ARIMA 拟合失败")
        self._res = best[1]
        self._horizon = horizon
        self._fitted = True
        return self

    def forecast(self) -> Forecast:
        self._ensure_fitted()
        fc = np.asarray(self._res.forecast(steps=self._horizon), dtype=float).ravel()
        return Forecast(mean=fc, method=self.name)


class ETSForecaster(BaseForecaster):
    name = "ets"

    def fit(self, series: TimeSeries, horizon: int) -> ETSForecaster:
        if not _HAVE_SM:
            raise FitError("statsmodels 不可用，ETS 后端缺失")
        if horizon < 1:
            raise FitError("horizon 须>=1")
        y = np.asarray(series.values, dtype=float)
        m = int(series.period)
        seasonal = "add" if m > 1 else None
        seas_p = m if m > 1 else None
        model = _SM_ETS(
            y,
            trend="add",
            seasonal=seasonal,
            seasonal_periods=seas_p,
            initialization_method="estimated",
        )
        self._res = model.fit()
        self._horizon = horizon
        self._fitted = True
        return self

    def forecast(self) -> Forecast:
        self._ensure_fitted()
        fc = np.asarray(self._res.forecast(steps=self._horizon), dtype=float).ravel()
        return Forecast(mean=fc, method=self.name)
