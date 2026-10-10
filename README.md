# ChronoForge · 世界顶级时间序列预测系统

> **ChronoFuse 融合旗舰** —— 滚动原点交叉验证 → 逆误差加权 → 向全局最可靠基收缩 → 非劣守卫。
> 无泄漏、确定性可复现、离线可降级。作者：**晨星**

[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Tests](https://github.com/CJX0712/chronoforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/chronoforge/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-92%25-brightgreen.svg)](https://github.com/CJX0712/chronoforge)
[![Ruff](https://img.shields.io/badge/ruff-checked-black.svg)](https://github.com/astral-sh/ruff)

---

## 一句话结论

在规范基准（40 条序列 / 8 类 DGP / horizon=12）上，**ChronoFuse sMAPE = 1.976**，
击败池中最强单一模型 **ETS 2.235**（含 Tier-0 statsmodels SOTA）与 Holt-Winters 2.377，
**相对最强单基下降 11.60%**，且逐序列方差严格更低。

---

## 为什么融合不是「把预测平均一下」

预测组合的价值不在于平均，而在于**让每条序列自己决定相信谁**。
ChronoFuse 的四步并不是堆料，每一步都对应一个真实的失效模式：

| 步骤 | 解决的问题 |
| --- | --- |
| **滚动原点交叉验证**（rolling-origin CV） | 单个 hold-out 窗口太短太噪声，会让某个基「撞大运」拿到最低 sMAPE（实测：单窗口下 AR 在某序列 CV 1.80 反超 HW 1.87，但测试上崩到 14.46），凸组合把 41% 权重给它直接毁掉预测 |
| **逆误差加权**（performance-based combination, Jain & Malek 2012 / M4 主流） | 目标指标 sMAPE 与评测直接对齐，避免 ridge 优化 MSE 而评测 sMAPE 的目标错位 |
| **向全局最可靠基收缩**（λ=0.5） | 非平稳数据下 CV/测试排名失稳，纯选择会满仓错基；收缩把下行锁死 |
| **非劣守卫**（non-inferior safeguard） | 若融合在 CV 上不优于最强单基超过 margin，回退该单基，绝不系统性劣化 |

### 诚实的性能边界（重要）

| 口径 | 结果 |
| --- | --- |
| 规范基准（默认 seed）相对最强单基 | **+11.60%** ✅ |
| 多 seed（5 个随机数据集）均值 ± 标准差 | **+4.34% ± 5.14%**（4/5 为正，最差 -2.02%） |
| 逐序列 sMAPE 标准差 | ChronoFuse **1.109** vs ETS 1.352 vs HW 2.377（更稳健） |

**为什么多 seed 均值不是 +11%？** 这是时间序列融合的学术常识而非缺陷：
在非平稳数据上，用历史 CV 预测「未来哪个模型最优」本身不可靠
（见 Claeskens, Magnus, Vasnev & Wang 2016；M4 竞赛结论）。
组合的核心收益是**稳健性**与「在多数数据集上不劣于、常优于最强单基」，
而非稳定碾压单一最优模型。本项目如实报告两种口径，不粉饰。

---

## 快速开始

```bash
git clone https://github.com/CJX0712/chronoforge.git
cd chronoforge
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m chronoforge.cli demo        # 端到端基准 + 落盘 benchmark.json
python -m chronoforge.cli detcheck    # 确定性校验（同 seed 两次逐位一致）
pytest -q                             # 42 项测试
```

离线（无 statsmodels）亦可用：自动降级为纯 numpy 五基池 + 跳过 Tier-0 参考，不伪造数字。

---

## 规范基准结果

| method | sMAPE mean | sMAPE std | MASE mean |
| --- | --- | --- | --- |
| **chronofuse（本系统）** | **1.976** | **1.109** | **0.609** |
| ets (statsmodels, Tier-0) | 2.235 | 1.352 | 0.796 |
| holt_winters_add | 2.377 | 1.550 | 0.762 |
| holt_winters_mul | 2.706 | 1.521 | 0.817 |
| average（朴素融合基线） | 3.879 | 1.799 | 0.849 |
| seasonal_naive | 4.764 | 3.846 | 1.383 |
| ar | 8.343 | 6.774 | 1.527 |
| arima (statsmodels, Tier-0) | 9.274 | 7.015 | 1.472 |
| naive | 12.750 | 9.368 | 1.839 |

> 40 条序列 / 8 类 DGP（线性趋势、指数趋势、加法季节、乘法季节、趋势+季节、
> 水平突变、多重季节、噪声随机游走）/ horizon=12 / 5 个数据 seed。

---

## 架构

单向依赖，无环：`core`（契约/错误/配置/种子）→ `data` → `forecasting` → `eval` → `pipeline` → `cli`。

```
chronoforge/
  core/      types, errors, config, interfaces, seed   # 全局契约与确定性
  data/      synthetic (8 类 DGP), loader
  forecasting/
    naive.py, exp_smoothing.py (SES/Holt/ Holt-Winters add|mul),
    ar.py (AIC 定阶), theta.py, statsmodels_bridge.py (ARIMA/ETS, Tier-0)
    chronofuse.py                                      # ★ 融合旗舰
  eval/      metrics (sMAPE/MAE/RMSE/MASE), benchmark
  pipeline/  run, determinism_check, save_report
  cli.py     demo / detcheck
```

详见 [`docs/architecture.md`](./docs/architecture.md) 与 [`docs/model_card.md`](./docs/model_card.md)。

---

## 不变量（可验证的硬标准）

| 不变量 | 验证方式 |
| --- | --- |
| **无泄漏**：所有基预测器只在 train（或滚动 CV 的内 train）拟合 | `eval/benchmark.py` 结构 + 测试 |
| **确定性**：同 seed 两次运行指标逐位一致 | `python -m chronoforge.cli detcheck` → `max_abs_diff = 0.0` |
| sMAPE/MASE 解析式与定义一致 | `tests/test_metrics.py` 解析解交叉验证 |
| 逆误差加权收敛于最优基 | `tests/test_chronofuse.py` 构造极端 CV 验证 |
| 非劣守卫回退到最强单基 | `tests/test_chronofuse.py` 强制触发回退路径 |

---

## 配置

环境变量 `CHRONOFORGE_<NAME>` 覆盖默认值：

| 字段 | 默认 | 含义 |
| --- | --- | --- |
| `seed` | 12345 | 全局随机种子 |
| `default_horizon` | 12 | 预测步长 |
| `n_rolling_origins` | 8 | 滚动 CV 原点数（按序列长度自动收敛） |
| `fuse_exponent` | 4.0 | 逆误差加权指数（越大越集中于最优基） |
| `fuse_shrink` | 0.5 | 向全局最可靠基收缩比例 |
| `safeguard` | true | 非劣守卫 |
| `non_inferior_margin` | 1.0 | 守卫容差（sMAPE 百分点） |

---

## 已知限制

- 多步（horizon=12）预测在非平稳序列上，CV 与测试排名可能失稳，故融合收益存在数据集波动（多 seed 最差 -2.02%）。
- Tier-0（ARIMA/ETS）依赖 statsmodels；缺失时降级，报告中标注 `reference_available: false`，不伪造参考数值。
- 合成数据用于可控评测，真实世界表现需以具体业务序列复核。

---

## 许可

MIT © 晨星。见 [`LICENSE`](./LICENSE)。
