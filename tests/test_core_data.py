"""core 与 data 层单测。"""

from __future__ import annotations

import numpy as np
import pytest
from chronoforge.core import config, errors, seed, types
from chronoforge.data import synthetic


def test_set_all_reproducible():
    seed.set_all(7)
    a = np.random.randn(5)
    seed.set_all(7)
    b = np.random.randn(5)
    assert np.array_equal(a, b)


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("CHRONOFORGE_DEFAULT_HORIZON", "24")
    c = config.Config()
    assert c.default_horizon == 24


def test_config_validation():
    with pytest.raises(errors.ConfigError):
        config.Config(default_horizon=0)


def test_errors_have_codes():
    assert errors.ConfigError().code == "E100"
    assert errors.FitError().code == "E300"


def test_timeseries_guard():
    with pytest.raises(errors.DataError):
        types.TimeSeries(values=np.array([1.0]), period=1)


def test_synthetic_shapes_reproducible():
    s1 = synthetic.generate_series("seasonal_add", 7, 120, 12, 1.0)
    s2 = synthetic.generate_series("seasonal_add", 7, 120, 12, 1.0)
    assert s1.values.shape == (120,)
    assert np.array_equal(s1.values, s2.values)


def test_synthetic_all_regimes():
    for r in synthetic.REGIMES:
        s = synthetic.generate_series(r, 42, 120, 12, 1.0)
        assert len(s.values) == 120 and np.all(np.isfinite(s.values))


def test_generate_dataset_count():
    ds = synthetic.generate_dataset(
        regimes=["trend_linear"], seeds=(1, 2, 3), length=100, period=12
    )
    assert len(ds) == 3
