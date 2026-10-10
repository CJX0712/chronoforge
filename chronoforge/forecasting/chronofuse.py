"""ChronoFuse · 自适应预测融合旗舰（滚动原点交叉验证加权）。

方法：对每个待预测序列，在训练数据上做滚动原点（rolling-origin）交叉验证，
估计各基预测器的多步 sMAPE（取多原点均值，抑制单窗口「幸运/厄运」噪声）；
再按「逆误差加权」（performance-based combination, Jain & Malek 2012；M4 竞赛
主流组合法之一）对各基在完整训练上的最终预测做凸组合。目标指标 sMAPE 与评测
直接对齐。

内置「非劣守卫」：若融合在交叉验证上的 sMAPE 不优于最强单基（超 margin），
则回退该最强单基在完整训练上的预测，确保融合绝不会系统性劣于最强单基。

纯 numpy，确定性可复现。
"""

from __future__ import annotations

import numpy as np

from ..core.config import CONFIG
from ..core.errors import FitError


class ChronoFuse:
    name = "chronofuse"

    def __init__(
        self,
        exponent: float | None = None,
        safeguard: bool | None = None,
        margin: float | None = None,
        shrink: float | None = None,
    ) -> None:
        self._exp = CONFIG.fuse_exponent if exponent is None else exponent
        self._safeguard = CONFIG.safeguard if safeguard is None else safeguard
        self._margin = CONFIG.non_inferior_margin if margin is None else margin
        self._shrink = CONFIG.fuse_shrink if shrink is None else shrink
        self._use_fallback = False
        self._best_base = 0
        self._weights = None
        self._fitted = False

    @staticmethod
    def _smape(fc: np.ndarray, ac: np.ndarray) -> float:
        # 惰性导入：避免在包初始化时与 eval 层形成循环依赖。
        from ..eval.metrics import smape

        return float(smape(fc, ac))

    def fuse(
        self,
        base_full_fc: np.ndarray,
        cv_smape: np.ndarray,
        origin_fc=None,
        origin_val=None,
        anchor_index: int | None = None,
    ):
        """对单条序列融合。

        base_full_fc: (n_bases, h)  在完整 train 上拟合后的最终基预测
        cv_smape:     (n_bases,)    各基在滚动原点交叉验证上的均值 sMAPE
        origin_fc:    list[(n_bases, h)]  各原点 CV 预测（用于守卫诊断）
        origin_val:   list[(h,)]          各原点 CV 真值
        返回 (h,) 融合预测。
        """
        Xfull = np.asarray(base_full_fc, dtype=float)
        if Xfull.ndim != 2:
            raise FitError("ChronoFuse.fuse: base_full_fc 须为 (n_bases, h)")
        n_bases = Xfull.shape[0]
        cv = np.asarray(cv_smape, dtype=float).ravel()
        if cv.shape[0] != n_bases:
            raise FitError("ChronoFuse.fuse: cv_smape 长度须=n_bases")

        eps = 1e-6
        w = (1.0 / (cv + eps)) ** self._exp
        w = w / w.sum()

        # 锚定基 = 全局最可靠基（由调用方传入 anchor_index，通常为跨序列 CV 均值最低者，
        # 本 DGP 即 Holt-Winters）。向该稳健基收缩可抑制非平稳数据下「CV 最优基被误选」
        # 导致的下行（Claeskens et al. 2016：组合的主要价值是稳健性，而非稳定击败单一最优模型）。
        best = int(np.argmin(cv))
        anchor_idx = best if anchor_index is None else int(anchor_index)

        if 0.0 < self._shrink < 1.0:
            anchor = np.zeros(n_bases, dtype=float)
            anchor[anchor_idx] = 1.0
            w = self._shrink * anchor + (1.0 - self._shrink) * w
            w = w / w.sum()

        self._weights = w
        fused = w @ Xfull  # (h,)
        self._use_fallback = False
        self._fitted = True

        if self._safeguard and origin_fc and origin_val:
            fused_cv = []
            for ofc, oval in zip(origin_fc, origin_val):
                fused_cv.append(
                    self._smape(w @ np.asarray(ofc, dtype=float), np.asarray(oval, dtype=float))
                )
            fused_cv_smape = float(np.mean(fused_cv))
            best_cv = cv[best]
            if fused_cv_smape >= best_cv + self._margin:
                self._use_fallback = True
                self._best_base = best
                return Xfull[best].copy()
        return fused
