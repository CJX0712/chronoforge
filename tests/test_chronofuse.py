"""ChronoFuse 融合器单测：形状 / 逆误差加权 / 向可靠基收缩 / 非劣守卫回退。"""

from __future__ import annotations

import numpy as np
from chronoforge.forecasting.chronofuse import ChronoFuse


def test_fuse_shape_and_finite():
    rng = np.random.RandomState(0)
    n_bases, h = 5, 12
    base_full = rng.randn(n_bases, h) + 10.0
    cv = np.array([2.0, 3.0, 1.5, 4.0, 2.5])
    origin_fc = [rng.randn(n_bases, h) + 10.0 for _ in range(4)]
    origin_val = [rng.randn(h) + 10.0 for _ in range(4)]
    fuse = ChronoFuse(safeguard=False)
    out = fuse.fuse(base_full, cv, origin_fc, origin_val, anchor_index=2)
    assert out.shape == (h,)
    assert np.all(np.isfinite(out))


def test_inverse_error_weighting_concentrates():
    # cv_smape 越低权重越大；base2 远优于其余 → 融合应贴近 base2
    n_bases, h = 3, 8
    base_full = np.zeros((n_bases, h))
    base_full[0] = 100.0  # 差
    base_full[1] = 100.0  # 差
    base_full[2] = 5.0  # 优
    cv = np.array([20.0, 20.0, 1.0])
    origin_fc = [np.zeros((n_bases, h)) for _ in range(3)]
    origin_val = [np.full(h, 5.0) for _ in range(3)]
    fuse = ChronoFuse(exponent=4.0, shrink=0.0, safeguard=False)
    out = fuse.fuse(base_full, cv, origin_fc, origin_val, anchor_index=2)
    # 融合应高度贴近 base2（≈5），远小于差基（100）
    assert np.allclose(out, 5.0, atol=0.01)
    assert np.all(out < 6.0)


def test_shrink_stays_between_anchor_and_adaptive():
    # 收缩把融合拉向 anchor（全局最可靠基），不应偏离锚定基过远。
    # 仅传 anchor_index（不传 origin_fc），隔离测试收缩行为。
    h = 8
    base_full = np.array([[100.0] * h, [100.0] * h, [5.0] * h])
    cv = np.array([20.0, 20.0, 1.0])
    # anchor=2（最可靠，cv 最低），shrink=0.5 → 0.5*5.0 + 0.5*自适应(≈5.0) ≈ 5.0
    fuse = ChronoFuse(exponent=4.0, shrink=0.5, safeguard=False)
    out = fuse.fuse(base_full, cv, None, None, anchor_index=2)
    assert np.allclose(out, 5.0, atol=0.01)

    # anchor=0（差基），shrink=0.5 → 0.5*100 + 0.5*5 ≈ 52.5
    fuse2 = ChronoFuse(exponent=4.0, shrink=0.5, safeguard=False)
    out2 = fuse2.fuse(base_full, cv, None, None, anchor_index=0)
    assert np.allclose(out2, 52.5, atol=0.01)


def test_shrink_zero_recovers_pure_inverse_weighting():
    # shrink=0 时应完全退化为纯逆误差加权（锚不参与）
    h = 8
    base_full = np.array([[100.0] * h, [100.0] * h, [5.0] * h])
    cv = np.array([20.0, 20.0, 1.0])
    f_zero = ChronoFuse(exponent=4.0, shrink=0.0, safeguard=False)
    f_anchor_good = ChronoFuse(exponent=4.0, shrink=0.0, safeguard=False)
    o1 = f_zero.fuse(base_full, cv, None, None, anchor_index=0)
    o2 = f_anchor_good.fuse(base_full, cv, None, None, anchor_index=2)
    # shrink=0 时锚无效 → 两者结果应一致
    assert np.allclose(o1, o2)


def test_safeguard_fallback_returns_best_base():
    # 构造：融合 CV 远差于最强单基 CV → 触发回退
    h = 6
    base_full = np.array([[1.0] * h, [2.0] * h, [3.0] * h])
    cv = np.array([1.0, 2.0, 3.0])  # base0 最优（CV 最低）
    # origin 预测：融合（按逆误差）应≈base0（cv 最低）→ 但人为让融合 CV 崩坏不可行，
    # 改为直接验证守卫分支：把 margin 设很大使 fused_cv 必超阈值。
    origin_fc = [np.array([[1.0] * h, [2.0] * h, [3.0] * h]) for _ in range(2)]
    origin_val = [np.full(h, 99.0) for _ in range(2)]  # 与所有预测均不符 → 融合/单基 CV 都高
    fuse = ChronoFuse(exponent=4.0, shrink=0.0, safeguard=True, margin=-100.0)
    out = fuse.fuse(base_full, cv, origin_fc, origin_val, anchor_index=0)
    assert fuse._use_fallback is True
    assert np.allclose(out, base_full[0])  # 回退到 CV 最强单基 base0


def test_fuse_deterministic():
    rng = np.random.RandomState(42)
    n_bases, h = 4, 10
    base_full = rng.randn(n_bases, h) + 5.0
    cv = np.array([3.0, 1.5, 2.0, 4.0])
    origin_fc = [rng.randn(n_bases, h) + 5.0 for _ in range(3)]
    origin_val = [rng.randn(h) + 5.0 for _ in range(3)]
    f1 = ChronoFuse()
    o1 = f1.fuse(
        base_full.copy(),
        cv.copy(),
        [o.copy() for o in origin_fc],
        [o.copy() for o in origin_val],
        anchor_index=1,
    )
    f2 = ChronoFuse()
    o2 = f2.fuse(
        base_full.copy(),
        cv.copy(),
        [o.copy() for o in origin_fc],
        [o.copy() for o in origin_val],
        anchor_index=1,
    )
    assert np.allclose(o1, o2)
