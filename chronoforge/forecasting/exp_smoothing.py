"""指数平滑族：SES / Holt / Holt-Winters（加/乘）。

参数以训练内 1 步向前 SSE 网格搜索选定（仅在 train 内，无泄漏）。
纯 numpy 实现，确定性可复现，作为离线兜底基线。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import FitError
from ..core.types import Forecast, TimeSeries
from .base import BaseForecaster

_INF = float("inf")
_ALPHAS = (0.2, 0.5, 0.8)
_BETAS = (0.0, 0.1, 0.4)
_GAMMAS = (0.0, 0.1, 0.4)


def _ses(y: np.ndarray, alpha: float):
    n = len(y)
    level = float(y[0])
    sse = 0.0
    for t in range(1, n):
        err = y[t] - level
        sse += err * err
        level = alpha * y[t] + (1.0 - alpha) * level
    return level, sse


def _ses_best(y: np.ndarray):
    best = (None, _INF)
    for a in _ALPHAS:
        level, sse = _ses(y, a)
        if sse < best[1]:
            best = ((a, level), sse)
    (a, level), _ = best
    return a, level


def _holt(y: np.ndarray, alpha: float, beta: float):
    n = len(y)
    level = float(y[0])
    trend = float(y[1] - y[0]) if n > 1 else 0.0
    sse = 0.0
    for t in range(1, n):
        fit = level + trend
        err = y[t] - fit
        sse += err * err
        prev = level
        level = alpha * y[t] + (1.0 - alpha) * (prev + trend)
        trend = beta * (level - prev) + (1.0 - beta) * trend
    return level, trend, sse


def _holt_best(y: np.ndarray):
    best = (None, _INF)
    for a in _ALPHAS:
        for b in _BETAS:
            level, trend, sse = _holt(y, a, b)
            if sse < best[1]:
                best = ((a, b, level, trend), sse)
    (a, b, level, trend), _ = best
    return a, b, level, trend


def _hw(y: np.ndarray, m: int, alpha: float, beta: float, gamma: float, mul: bool):
    n = len(y)
    if n < 2 * m:
        return None, _INF
    yy = y.astype(float)
    if mul:
        yy = np.clip(yy, 1e-6, None)
        lv = float(np.mean(yy[:m]))
        seas = yy[:m] / lv
    else:
        lv = float(np.mean(yy[:m]))
        seas = yy[:m] - lv
    tr = (float(np.mean(yy[m : 2 * m])) - lv) / m
    sse = 0.0
    for t in range(m, n):
        idx = (t - m) % m
        fit = (lv + tr) * seas[idx] if mul else lv + tr + seas[idx]
        err = yy[t] - fit
        sse += err * err
        prev = lv
        if mul:
            lv = alpha * (yy[t] / seas[idx]) + (1.0 - alpha) * (prev + tr)
        else:
            lv = alpha * (yy[t] - seas[idx]) + (1.0 - alpha) * (prev + tr)
        tr = beta * (lv - prev) + (1.0 - beta) * tr
        if mul:
            seas[idx] = gamma * (yy[t] / lv) + (1.0 - gamma) * seas[idx]
        else:
            seas[idx] = gamma * (yy[t] - lv) + (1.0 - gamma) * seas[idx]
    return (lv, tr, seas, mul), sse


def _hw_best(y: np.ndarray, m: int, mul: bool):
    best = (None, _INF)
    for a in _ALPHAS:
        for b in _BETAS:
            for g in _GAMMAS:
                state, sse = _hw(y, m, a, b, g, mul)
                if state is not None and sse < best[1]:
                    best = ((a, b, g, state), sse)
    if best[0] is None:
        # 数据不足两周期：退化为 Holt
        a, b, level, trend = _holt_best(y)
        return ("holt", a, b, level, trend), best[1]
    (a, b, g, state), _ = best
    return (a, b, g, state), _


class SESForecaster(BaseForecaster):
    name = "ses"

    def fit(self, series: TimeSeries, horizon: int) -> SESForecaster:
        if horizon < 1:
            raise FitError("horizon 须>=1")
        self._a, self._level = _ses_best(np.asarray(series.values, dtype=float))
        self._horizon = horizon
        self._fitted = True
        return self

    def forecast(self) -> Forecast:
        self._ensure_fitted()
        return Forecast(mean=np.full(self._horizon, self._level, dtype=float), method=self.name)


class HoltForecaster(BaseForecaster):
    name = "holt"

    def fit(self, series: TimeSeries, horizon: int) -> HoltForecaster:
        if horizon < 1:
            raise FitError("horizon 须>=1")
        self._a, self._b, self._level, self._trend = _holt_best(
            np.asarray(series.values, dtype=float)
        )
        self._horizon = horizon
        self._fitted = True
        return self

    def forecast(self) -> Forecast:
        self._ensure_fitted()
        h = self._horizon
        out = self._level + self._trend * np.arange(1, h + 1)
        return Forecast(mean=out, method=self.name)


class HoltWintersForecaster(BaseForecaster):
    name = "holt_winters"

    def __init__(self, season_type: str = "add") -> None:
        super().__init__()
        if season_type not in ("add", "mul"):
            raise FitError("season_type 须 add/mul")
        self._stype = season_type

    def fit(self, series: TimeSeries, horizon: int) -> HoltWintersForecaster:
        if horizon < 1:
            raise FitError("horizon 须>=1")
        y = np.asarray(series.values, dtype=float)
        m = max(2, int(series.period)) if int(series.period) > 1 else 2
        res, _ = _hw_best(y, m, mul=(self._stype == "mul"))
        self._state = res
        self._m = m
        self._len = len(y)
        self._horizon = horizon
        self._fitted = True
        return self

    def forecast(self) -> Forecast:
        self._ensure_fitted()
        h = self._horizon
        st = self._state
        if st[0] == "holt":
            _, _, _, level, trend = st
            out = level + trend * np.arange(1, h + 1)
            return Forecast(mean=out, method=self.name)
        (lv, tr, seas, mul) = st[3]
        out = np.empty(h, dtype=float)
        m = self._m
        n = self._len
        for k in range(1, h + 1):
            idx = (n - 1 + k - m) % m
            if mul:
                out[k - 1] = (lv + k * tr) * seas[idx]
            else:
                out[k - 1] = lv + k * tr + seas[idx]
        return Forecast(mean=out, method=self.name)
