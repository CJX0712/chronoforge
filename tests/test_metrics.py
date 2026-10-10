"""指标单测：已知解析式交叉验证。"""

from __future__ import annotations

import numpy as np
from chronoforge.eval.metrics import mae, mase, naive_scale, rmse, smape


def test_smape_known():
    # f=100,a=120 -> 200*20/220 = 18.1818...
    assert abs(smape([100.0], [120.0]) - 18.1818181818) < 1e-6


def test_smape_perfect_zero():
    assert smape([1.0, 2.0], [1.0, 2.0]) == 0.0


def test_mae_rmse_known():
    f = np.array([1.0, 2.0, 3.0])
    a = np.array([2.0, 2.0, 2.0])
    assert abs(mae(f, a) - 2.0 / 3.0) < 1e-12
    assert abs(rmse(f, a) - (np.sqrt((1 + 0 + 1) / 3.0))) < 1e-12


def test_mase_scale():
    train = np.array([1.0, 2.0, 4.0, 7.0])
    scale = naive_scale(train)
    assert abs(scale - (1 + 2 + 3) / 3.0) < 1e-12
    assert mase([1.0], [2.0], scale) == abs(1.0) / scale


def test_mase_zero_scale_nan():
    assert mase([1.0], [2.0], 0.0) != 0.0  # nan
