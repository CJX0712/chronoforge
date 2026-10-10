# ChronoForge 架构说明

作者：晨星。本文解释**为什么这样分层**，而不只是列出目录。

## 1. 单向依赖（无环）

```
core  →  data  →  forecasting  →  eval  →  pipeline  →  cli
```

- `core` 只定义契约（`Forecaster`/`Combiner` Protocol）、错误码、`Config`、全局种子，**不依赖任何上层**。
- `forecasting` 只依赖 `core` 与 numpy；`statsmodels` 通过 `statsmodels_bridge.available()` 探测，
  缺失时 Tier-0 不可用，上层自动降级，绝不硬崩。
- `eval/benchmark.py` 依赖 `forecasting`，反过来不成立。
- `chronofuse.py` 需要 `smape`，但若在模块顶层 `from ..eval.metrics import smape`，
  会经过 `eval/__init__` → `benchmark` → `forecasting` → `chronofuse` 形成**循环导入**。
  解法：在方法内部惰性导入（运行时模块已全加载），保持公式单一真值、不复制实现。

## 2. 确定性设计

- `core/seed.set_all(seed)` 一次性设齐 `random` 与 `numpy` 的 RNG。
- 数据生成器 (`data/synthetic.py`) 使用独立的 `np.random.RandomState(seed)`，
  与全局 RNG **解耦**：换全局种子不会改变数据。
- 所有基预测器为确定性算法（网格搜索 SSE / OLS 闭式 / AIC 定阶），无随机初始化。
- 结果：`python -m chronoforge.cli detcheck` 两次运行 `max_abs_diff = 0.0`。

## 3. 无泄漏设计

```
序列: [ 0 .................... n-h-1 | n-h ...... n-1 ]
                              train_full └── test ──┘
      ├─ rolling-origin CV windows ──┘   （全在 train_full 内部）
```

- 最终预测：各基在 `train_full` 拟合 → `Xfull (n_bases, h)`。
- 交叉验证：在 `train_full` 内做滚动原点切片，每个原点用其之前的数据拟合、
  预测该原点的后 `h` 点，得到 `origin_fc` / `origin_val`。
- **test（序列末 h 点）从不参与任何拟合或权重学习**，仅用于最终评测。
- 滚动原点数按序列长度自动收敛：`eff = min(n_origins, (len - min_train) // horizon)`，
  不足 2 个原点时回退为单 hold-out 窗口。

## 4. ChronoFuse 融合管线

```
for 每条序列:
    1. fit 各基于 train_full            → Xfull (B, h)
    2. rolling-origin CV              → cv_smape (B,)  ← 各基多原点均值 sMAPE
    3. w = (1/cv_smape)^exp, 归一化     ← 逆误差加权（目标 = 评测指标 sMAPE）
    4. w = λ·e_anchor + (1-λ)·w         ← 向全局最可靠基（跨序列 CV 均值最低者）收缩
    5. fused = w @ Xfull               → (h,)
    6. 守卫: fused_cv ≥ best_cv + margin → 回退最强单基 Xfull[argmin cv]
```

**为什么必须有 4（收缩）？** 纯逆误差加权在第 4 步之前等价于「几乎满仓 CV 最优基」。
非平稳数据下 CV 最优 ≠ 测试最优，一旦选错就是灾难（实测 -12%）。收缩把融
合拉回稳健基（本 DGP 为 ETS/HW），把下行从 -12% 锁到 -2%。

**为什么 anchor 取「全局」而非「本序列 CV 最优」？** 若锚定本序列 CV 最优，
而该基恰是被误选的那个，锚就失效了（实测该锚法最差 -14.6%）。跨序列统计的
全局最可靠基才是真正的稳健支点。

## 5. 为什么不用 ridge stacking

早期版本在 CV 残差上做 ridge 融合（Athanasopoulos 2010 思路）。失败原因：
**ridge 优化 MSE，而评测指标是 sMAPE**，目标错位导致有-bias 的保守预测被偏好。
改为直接对 sMAPE 做逆误差加权后，「优化目标」与「评测目标」一致。

## 6. 组合池选择

- 纯 numpy 五基：naive、seasonal_naive、holt_winters_add、holt_winters_mul、ar。
- 可选 Tier-0：ets（statsmodels 可用时并入）。
- **不纳入** ses / holt / theta：其在该 DGP 上系统性失准（sMAPE 12~17），
  进入凸组合只会稀释权重。`median` 与 `mean-trim` 组合方案实测分别为 -6.46% /
  -42.31%，印证了「弱成员拉垮组合」。
