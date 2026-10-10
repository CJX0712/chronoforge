"""错误码体系 E100~E500。

每个异常带稳定 code，便于 CI 门禁与调用方按码处理。
"""

from __future__ import annotations


class ChronoError(Exception):
    """基类。"""

    code = "E000"

    def __init__(self, message: str = "") -> None:
        self.message = message
        super().__init__(f"[{self.code}] {message}" if message else f"[{self.code}]")


class ConfigError(ChronoError):
    code = "E100"


class DataError(ChronoError):
    code = "E200"


class FitError(ChronoError):
    code = "E300"


class ForecastError(ChronoError):
    code = "E400"


class UnsupportedError(ChronoError):
    code = "E500"
