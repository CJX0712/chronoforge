"""Pipeline 与基准单测：结构 / ≥3 seeds / 确定性逐位一致 / 离线兜底。"""

from __future__ import annotations

from chronoforge.core import seed
from chronoforge.eval import benchmark
from chronoforge.eval.benchmark import run_benchmark
from chronoforge.pipeline import determinism_check, run


def _small_kwargs():
    return dict(regimes=["trend_linear", "seasonal_add"], seeds=(7, 42, 123), length=100, period=12)


def test_run_structure():
    res, ds = run(**_small_kwargs())
    assert res["n_series"] == 6
    assert "chronofuse" in res["methods"]
    assert "strongest_single_base" in res


def test_benchmark_three_seeds_mean_std():
    seed.set_all(5)
    from chronoforge.data.synthetic import generate_dataset

    ds = generate_dataset(**_small_kwargs())
    res = run_benchmark(ds, horizon=12)
    for m, d in res["methods"].items():
        assert d["n"] >= 3
        if d["n"] > 1:
            assert d["smape_std"] >= 0.0


def test_determinism_bit_identical():
    dc = determinism_check(**_small_kwargs())
    assert dc["bit_identical"] is True, dc


def test_offline_fallback_no_statsmodels(monkeypatch):
    # 模拟 statsmodels 缺失：benchmark 仍应产出 chronofuse 且不崩溃
    import chronoforge.forecasting.statsmodels_bridge as bridge

    monkeypatch.setattr(bridge, "_HAVE_SM", False)
    monkeypatch.setattr(benchmark, "_sm_available", lambda: False)
    seed.set_all(9)
    from chronoforge.data.synthetic import generate_dataset

    ds = generate_dataset(**_small_kwargs())
    res = run_benchmark(ds, horizon=12)
    assert "chronofuse" in res["methods"]
    assert res["reference_available"] is False
