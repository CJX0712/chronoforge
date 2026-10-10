"""全局确定性种子。

唯一入口 set_all(seed) 一次性设齐 random / numpy 全局 RNG，保证同 seed 两次
运行 benchmark.json 核心指标逐位一致（elapsed_sec 除外）。
"""

from __future__ import annotations

import random

import numpy as np

_DEFAULT_SEED = 12345
_CURRENT_SEED = {"seed": _DEFAULT_SEED}


def set_all(seed: int = _DEFAULT_SEED) -> int:
    """设齐所有随机源，返回实际生效的 seed。"""
    s = int(seed)
    _CURRENT_SEED["seed"] = s
    random.seed(s)
    np.random.seed(s)
    return s


def get_seed() -> int:
    return _CURRENT_SEED["seed"]
