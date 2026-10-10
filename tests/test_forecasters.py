"""基预测器单测：形状 / 有限性 / 确定性。"""

from __future__ import annotations

import numpy as np
import pytest
from chronoforge.core import seed
from chronoforge.core.types import TimeSeries
from chronoforge.forecasting import (
    ARForecaster,
    HoltForecaster,
    HoltWintersForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    SESForecaster,
    ThetaForecaster,
)
from chronoforge.forecasting.statsmodels_bridge import ARIMAForecaster, ETSForecaster

_FACTORIES = [
    NaiveForecaster,
    SeasonalNaiveForecaster,
    SESForecaster,
    HoltForecaster,
    lambda: HoltWintersForecaster("add"),
    lambda: HoltWintersForecaster("mul"),
    ARForecaster,
    ThetaForecaster,
]

_H = 12


@pytest.mark.parametrize("factory", _FACTORIES)
def test_forecast_shape_and_finite(factory):
    seed.set_all(11)
    y = np.sin(np.arange(120) * 0.5) + np.random.randn(120) * 0.1
    m = factory()
    ts = TimeSeries(values=y, period=12)
    m.fit(ts, _H)
    fc = m.forecast().mean
    assert fc.shape == (_H,)
    assert np.all(np.isfinite(fc))


@pytest.mark.parametrize("factory", _FACTORIES)
def test_forecast_deterministic(factory):
    y = np.sin(np.arange(120) * 0.5) + 0.1 * np.arange(120) * 0.01
    m1 = factory()
    m1.fit(TimeSeries(values=y.copy(), period=12), _H)
    m2 = factory()
    m2.fit(TimeSeries(values=y.copy(), period=12), _H)
    assert np.allclose(m1.forecast().mean, m2.forecast().mean, atol=1e-12)


def test_arima_ets_if_available():
    if not (ARIMAForecaster and ETSForecaster):
        pytest.skip("statsmodels 不可用")
    from chronoforge.forecasting.statsmodels_bridge import available

    if not available():
        pytest.skip("statsmodels 不可用")
    seed.set_all(3)
    y = np.sin(np.arange(120) * 0.5) + np.random.randn(120) * 0.2
    for Fac in (ARIMAForecaster, ETSForecaster):
        m = Fac()
        m.fit(TimeSeries(values=y, period=12), _H)
        fc = m.forecast().mean
        assert fc.shape == (_H,) and np.all(np.isfinite(fc))
