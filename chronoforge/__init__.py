"""ChronoForge · 世界顶级时间序列预测系统。

ChronoFuse 融合旗舰：滚动原点交叉验证 → 逆误差加权 → 向全局最可靠基收缩 → 非劣守卫。
无泄漏、确定性可复现、离线可降级（statsmodels 缺失时退化为纯 numpy 基池）。

作者：晨星
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "晨星"

__all__ = ["__author__", "__version__"]
