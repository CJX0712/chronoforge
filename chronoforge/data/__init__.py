"""数据层：合成时间序列生成 + 文件载入。

生成器固定 seed（RandomState），与全局 RNG 解耦，保证可复现且互不串扰。
"""

from . import synthetic

__all__ = ["synthetic"]
