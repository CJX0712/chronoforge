"""预测层：基预测器 + ChronoFuse 融合旗舰。

模块只依赖 core 契约与 numpy；statsmodels 为可选 Tier-0 后端，缺失自动降级。
"""

from .ar import ARForecaster
from .chronofuse import ChronoFuse
from .exp_smoothing import (
    HoltForecaster,
    HoltWintersForecaster,
    SESForecaster,
)
from .naive import NaiveForecaster, SeasonalNaiveForecaster
from .statsmodels_bridge import ARIMAForecaster, ETSForecaster
from .theta import ThetaForecaster

__all__ = [
    "ARForecaster",
    "ARIMAForecaster",
    "ChronoFuse",
    "ETSForecaster",
    "HoltForecaster",
    "HoltWintersForecaster",
    "NaiveForecaster",
    "SESForecaster",
    "SeasonalNaiveForecaster",
    "ThetaForecaster",
]
