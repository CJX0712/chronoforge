"""核心数据类型。全 dataclass，零依赖。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TimeSeries:
    """一条单变量时间序列。"""

    values: np.ndarray  # 1-D float
    name: str = "series"
    period: int = 1  # 已知季节周期（无季节则为 1）

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float).ravel()
        if self.values.ndim != 1 or self.values.size < 2:
            from .errors import DataError

            raise DataError("values 须为长度>=2 的一维数组")
        if self.period < 1:
            raise DataError("period 须>=1")


@dataclass
class Forecast:
    """一次 h 步预测结果。"""

    mean: np.ndarray  # 1-D, 长度 h
    method: str = "unknown"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=float).ravel()
