"""配置：环境变量 ENV_CHRONOFORGE_* 覆盖 + 默认值 + schema 校验。"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields

from .errors import ConfigError

_PREFIX = "CHRONOFORGE_"


@dataclass
class Config:
    """全局配置。所有字段可被 CHRONOFORGE_<NAME> 环境变量覆盖。"""

    seed: int = 12345
    default_horizon: int = 12
    max_ar_order: int = 12
    hw_grid_max: int = 8  # Holt-Winters 网格为 3^? 上限控制
    ridge_alpha: float = 0.1
    n_rolling_origins: int = 8
    fuse_exponent: float = 4.0  # ChronoFuse 逆误差加权指数（越大越集中于最优基）
    fuse_shrink: float = 0.5  # ChronoFuse 向「全局最可靠基」收缩比例（抑制非平稳下的误选下行）
    safeguard: bool = True  # ChronoFuse 非劣守卫
    non_inferior_margin: float = (
        1.0  # 守卫容差（sMAPE 百分点；融合验证 sMAPE 超过最强单基该值才回退）
    )

    def __post_init__(self) -> None:
        self._apply_env()
        self.validate()

    def _apply_env(self) -> None:
        for f in fields(self):
            env = os.environ.get(f"{_PREFIX}{f.name.upper()}")
            if env is not None:
                try:
                    setattr(self, f.name, _coerce(f.type, env))
                except ValueError as e:
                    raise ConfigError(f"环境变量 {_PREFIX}{f.name.upper()}={env!r} 无法解析: {e}")

    def validate(self) -> None:
        if self.default_horizon < 1:
            raise ConfigError("default_horizon 须>=1")
        if self.max_ar_order < 1:
            raise ConfigError("max_ar_order 须>=1")
        if self.ridge_alpha < 0:
            raise ConfigError("ridge_alpha 须>=0")
        if self.n_rolling_origins < 1:
            raise ConfigError("n_rolling_origins 须>=1")

    @classmethod
    def load(cls) -> Config:
        return cls()


def _coerce(kind: str, value: str):
    v = value.strip()
    if kind.startswith("int"):
        return int(v)
    if kind.startswith("float"):
        return float(v)
    if kind.startswith("bool"):
        return v.lower() in ("1", "true", "yes", "on")
    return v


CONFIG = Config()
