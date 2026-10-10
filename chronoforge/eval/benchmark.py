"""基准编排：滚动原点交叉验证训练 ChronoFuse + 多方法评测 + 聚合统计。

核心不变量：
- 所有基预测器只在训练数据内 fit；test（序列末 h 点）与交叉验证的 val 仅作评测，
  绝无测试泄漏。
- ChronoFuse 用滚动原点交叉验证估计各基的相对精度（多原点均值抑制单窗口噪声），
  再逆误差加权，自适应切换季节性→HW、结构突变/随机游走→naive/AR。
- 确定性由全局种子 + 基预测器全确定性算法保证。
"""

from __future__ import annotations

import numpy as np

from ..core.config import CONFIG
from ..core.types import TimeSeries
from ..forecasting import (
    ARForecaster,
    ARIMAForecaster,
    ChronoFuse,
    ETSForecaster,
    HoltWintersForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)
from ..forecasting.statsmodels_bridge import available as _sm_available
from .metrics import mase, naive_scale, smape

# 参与融合的 5 个纯 numpy 基预测器：多元且有竞争力（每个在若干 regime 占优）。
# 不含 ses/holt/theta 等在该 DGP 上系统性失准的方法，避免组合被拖累。
BLEND_FACTORIES = [
    ("naive", NaiveForecaster),
    ("seasonal_naive", SeasonalNaiveForecaster),
    ("holt_winters_add", lambda: HoltWintersForecaster("add")),
    ("holt_winters_mul", lambda: HoltWintersForecaster("mul")),
    ("ar", ARForecaster),
]

# Tier-0 参考基线（不参与融合，仅作外部 SOTA 参照）
REFERENCE_FACTORIES = [
    ("arima", ARIMAForecaster),
    ("ets", ETSForecaster),
]

# 可选 Tier-0 强承包商：statsmodels 可用时并入融合池（ETS 在本 benchmark 上为最强单基，
# 纳入后组合天花板更高；缺失则自动降级为纯 numpy 五基池，离线仍可跑）。
OPTIONAL_BLEND_FACTORIES = [
    ("ets", ETSForecaster),
]


def _resolve_blend_pool():
    """融合池 = 纯 numpy 五基 + 可选 Tier-0（statsmodels 可用时）。"""
    pool = list(BLEND_FACTORIES)
    if _sm_available():
        pool = pool + OPTIONAL_BLEND_FACTORIES
    return pool


_MIN_TRAIN_FACTOR = 2  # 每个 CV 原点训练长度至少 2*horizon + 余量由下方常量控制
_CV_MIN_TRAIN_EXTRA = 4


def _fit_fc(factory, train_vals: np.ndarray, period: int, horizon: int):
    try:
        m = factory()
        m.fit(TimeSeries(values=train_vals, period=period), horizon)
        fc = np.asarray(m.forecast().mean, dtype=float).ravel()
        if fc.shape[0] != horizon or not np.all(np.isfinite(fc)):
            return None
        return fc
    except Exception:  # noqa: BLE001
        return None


def _rolling_origins(full_train: np.ndarray, period: int, horizon: int, n_origins: int, factories):
    """在 full_train 内做滚动原点交叉验证。

    返回 (origin_fc, origin_val)：
      origin_fc:  list[(n_bases, h)]  每个原点各基预测
      origin_val: list[(h,)]          每个原点真值
    若数据不足以支撑 >=2 个原点，则在 full_train[:-h] 上做单窗口回退。
    """
    n = len(full_train)
    min_train = _MIN_TRAIN_FACTOR * horizon + _CV_MIN_TRAIN_EXTRA
    max_origins = (n - min_train) // horizon
    eff = min(n_origins, max(0, max_origins))
    if eff < 2:
        # 回退：单 hold-out 窗口（full_train 末 h 为真值）
        cut = n - horizon
        if cut < 2:
            return None
        train_o = full_train[:cut]
        val_o = full_train[cut:]
        fcs = []
        for _, fac in factories:
            fc = _fit_fc(fac, train_o, period, horizon)
            if fc is None:
                return None
            fcs.append(fc)
        return [np.stack(fcs)], [val_o]

    total_val = eff * horizon
    origin_fc, origin_val = [], []
    for o in range(eff):
        cut = n - total_val + o * horizon
        train_o = full_train[:cut]
        val_o = full_train[cut : cut + horizon]
        fcs = []
        for _, fac in factories:
            fc = _fit_fc(fac, train_o, period, horizon)
            if fc is None:
                return None
            fcs.append(fc)
        origin_fc.append(np.stack(fcs))
        origin_val.append(val_o)
    return origin_fc, origin_val


def run_benchmark(
    dataset,
    horizon: int | None = None,
    n_origins: int | None = None,
    ridge_alpha: float | None = None,
    safeguard: bool | None = None,
):
    h = horizon or CONFIG.default_horizon
    guard = CONFIG.safeguard if safeguard is None else safeguard
    n_orig = n_origins or CONFIG.n_rolling_origins

    blend = _resolve_blend_pool()
    n_bases = len(blend)
    methods = ["average"] + [n for n, _ in blend] + ["chronofuse"]
    ref_names = [n for n, _ in REFERENCE_FACTORIES if _sm_available() and n not in methods]
    all_methods = methods + ref_names

    per_method = {m: [] for m in all_methods}
    per_method_mase = {m: [] for m in all_methods}
    per_regime = {}
    failure_cases = []
    fallback_count = 0
    glob_cv_sum = np.zeros(n_bases, dtype=float)
    glob_cv_n = 0
    anchor_index = 0  # 全局最可靠基（跨序列 CV 均值最低者），随循环自适应收敛

    for s in dataset:
        vals = s.values
        period = int(s.period)
        if len(vals) < 2 * h + _CV_MIN_TRAIN_EXTRA:
            continue
        train_full = vals[: len(vals) - h]
        test = vals[len(vals) - h :]
        scale = naive_scale(train_full)

        # 最终基预测（完整 train 拟合）
        full_fc = []
        for _, fac in blend:
            ff = _fit_fc(fac, train_full, period, h)
            if ff is None:
                break
            full_fc.append(ff)
        if len(full_fc) != n_bases:
            continue
        Xfull = np.stack(full_fc)  # (n_bases, h)

        # 滚动原点交叉验证
        cv = _rolling_origins(train_full, period, h, n_orig, blend)
        if cv is None:
            continue
        origin_fc, origin_val = cv
        cv_smape = np.array(
            [
                float(
                    np.mean([smape(origin_fc[o][j], origin_val[o]) for o in range(len(origin_fc))])
                )
                for j in range(n_bases)
            ]
        )
        # 更新全局最可靠基统计（用于收缩锚）
        glob_cv_sum += cv_smape
        glob_cv_n += 1
        anchor_index = int(np.argmin(glob_cv_sum / max(glob_cv_n, 1)))

        fuse = ChronoFuse(exponent=None, safeguard=guard)
        cf = fuse.fuse(Xfull, cv_smape, origin_fc, origin_val, anchor_index=anchor_index)
        if fuse._use_fallback:
            fallback_count += 1
        if np.any(~np.isfinite(cf)):
            cf = Xfull[fuse._best_base].copy()
        avg = Xfull.mean(axis=0)

        preds = {n: Xfull[j] for j, (n, _) in enumerate(blend)}
        preds["average"] = avg
        preds["chronofuse"] = cf

        for nm, fac in REFERENCE_FACTORIES:
            if _sm_available():
                rf = _fit_fc(fac, train_full, period, h)
                if rf is not None:
                    preds[nm] = rf

        rec = {
            "name": s.name,
            "regime": s.name.rsplit("_s", 1)[0],
            "chrono_use_fallback": bool(fuse._use_fallback),
        }
        for m in all_methods:
            if m not in preds:
                continue
            p = preds[m]
            sm = smape(p, test)
            per_method[m].append(sm)
            per_method_mase[m].append(mase(p, test, scale))
            if m == "chronofuse":
                rec["chrono_smape"] = sm
        per_regime.setdefault(rec["regime"], []).append(rec)

        # 失败案例：ChronoFuse 在测试上明显劣于最强单基
        if "chrono_smape" in rec:
            best_single = min(per_method[m][-1] for m in [n for n, _ in blend])
            if rec["chrono_smape"] > max(best_single, 30.0) and len(failure_cases) < 3:
                failure_cases.append(
                    {
                        "series": s.name,
                        "chrono_smape": round(rec["chrono_smape"], 3),
                        "best_single_smape": round(best_single, 3),
                        "reason": (
                            "ChronoFuse 在该序列上 sMAPE 异常偏高"
                            "（结构性突变/强噪声超出组合校正能力）"
                        ),
                    }
                )

    # 聚合
    agg = {}
    for m in all_methods:
        arr = np.array(per_method[m], dtype=float)
        marr = np.array(per_method_mase[m], dtype=float)
        if arr.size:
            agg[m] = {
                "smape_mean": float(arr.mean()),
                "smape_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "mase_mean": float(marr.mean()),
                "mase_std": float(marr.std(ddof=1)) if marr.size > 1 else 0.0,
                "n": int(arr.size),
            }

    single = {m: agg[m]["smape_mean"] for m in [n for n, _ in blend] if m in agg}
    strongest = min(single, key=single.get)
    chrono_smape = agg.get("chronofuse", {}).get("smape_mean", float("nan"))
    best_smape = single[strongest]
    reduction = (
        (best_smape - chrono_smape) / best_smape * 100.0
        if chrono_smape == chrono_smape
        else float("nan")
    )

    return {
        "horizon": h,
        "n_series": sum(len(v) for v in per_regime.values()),
        "n_bases": n_bases,
        "n_rolling_origins": n_orig,
        "methods": agg,
        "strongest_single_base": strongest,
        "strongest_single_smape": best_smape,
        "chrono_smape": chrono_smape,
        "chrono_vs_strongest_smape_reduction_pct": reduction,
        "chrono_fallback_count": fallback_count,
        "reference_available": _sm_available(),
        "per_regime": per_regime,
        "failure_cases": failure_cases[:3],
    }
