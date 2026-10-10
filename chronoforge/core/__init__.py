"""ChronoForge · 时间序列预测系统（作者：晨星）。

core 包：类型 / 错误码 / 配置 / 接口契约 / 全局确定性种子。
调用单向无环：cli -> pipeline -> {data, forecasting, eval} -> core。
"""

from . import config, errors, interfaces, seed, types

__all__ = ["config", "errors", "interfaces", "seed", "types"]
