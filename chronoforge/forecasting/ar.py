"""自回归 AR(p)：OLS 拟合并以 AIC 在 train 内定阶，递归多步预测。"""

from __future__ import annotations

import numpy as np

from ..core.config import CONFIG
from ..core.errors import FitError
from ..core.types import Forecast, TimeSeries
from .base import BaseForecaster


def _design(y: np.ndarray, p: int):
    n = len(y)
    X = np.empty((n - p, p + 1), dtype=float)
    X[:, 0] = 1.0
    for i in range(1, p + 1):
        X[:, i] = y[p - i : n - i]
    target = y[p:]
    return X, target


def _fit_p(y: np.ndarray, p: int):
    X, target = _design(y, p)
    beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    resid = target - X @ beta
    n = len(target)
    k = p + 1
    sigma2 = float(resid @ resid) / max(1, n - k)
    if sigma2 <= 0:
        sigma2 = 1e-12
    aic = n * (np.log(2 * np.pi * sigma2) + 1.0) + 2.0 * k
    return beta, aic


class ARForecaster(BaseForecaster):
    name = "ar"

    def fit(self, series: TimeSeries, horizon: int) -> ARForecaster:
        if horizon < 1:
            raise FitError("horizon 须>=1")
        y = np.asarray(series.values, dtype=float)
        maxp = min(int(CONFIG.max_ar_order), len(y) - 3)
        if maxp < 1:
            maxp = 1
        best = None
        for p in range(1, maxp + 1):
            beta, aic = _fit_p(y, p)
            if best is None or aic < best[2]:
                best = (p, beta, aic)
        self._p, self._beta, _ = best
        self._seed_vals = y[-self._p :].copy()
        self._horizon = horizon
        self._fitted = True
        return self

    def forecast(self) -> Forecast:
        self._ensure_fitted()
        h = self._horizon
        p = self._p
        beta = self._beta
        hist = list(self._seed_vals)
        out = np.empty(h, dtype=float)
        for k in range(h):
            x = np.array([1.0, *hist[-p:]], dtype=float)
            yhat = float(x @ beta)
            out[k] = yhat
            hist.append(yhat)
        return Forecast(mean=out, method=self.name)
