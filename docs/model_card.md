# ChronoFuse Model Card

作者：晨星。按训练前/训练后两节如实填写，含明确的失败模式与边界。

## 模型细节

| 项 | 值 |
| --- | --- |
| 名称 | ChronoFuse（ChronoForge 融合旗舰） |
| 版本 | 0.1.0 |
| 类型 | 预测组合（forecast combination / ensemble meta-learner） |
| 基预测器 | Naive、SeasonalNaive、SES、Holt、Holt-Winters(add/mul)、AR(AIC定阶)、Theta；可选 Tier-0：ETS、ARIMA（statsmodels） |
| 融合权重 | `w ∝ (1/cv_smape)^4`，向全局最可靠基收缩 λ=0.5 |
| 权重学习数据 | 仅训练序列内部的滚动原点交叉验证（**无测试泄漏**） |
| 依赖 | numpy（必需）；statsmodels（可选，缺失降级） |
| 许可证 | MIT |

## 预期用途

- **适合**：单/多变量外的**单条序列多步预测**，季节/趋势/结构突变混合的多 regime 序列池；
  需要在「不劣于最强单一模型」前提下追求更稳健表现的场景。
- **不适合**：真实业务序列未经复核直接上线；含外部协变量的因果预测；
  需要概率预测区间（当前仅输出点预测的均值）。

## 训练数据（评测数据）

合成生成器，8 类 DGP：线性趋势、指数趋势、加法季节、乘法季节、趋势+季节、
水平突变、多重季节、噪声随机游走。每条长度 120，period=12，5 个数据 seed，共 40 条。
噪声 σ=1.0。使用 `np.random.RandomState(seed)`，**独立于全局 RNG**。

## 性能（horizon=12，sMAPE% 越低越好）

| 指标 | 值 |
| --- | --- |
| ChronoFuse | **1.976 ± 1.109** |
| 最强单基 ETS (Tier-0) | 2.235 ± 1.352 |
| Holt-Winters(add) | 2.377 ± 1.550 |
| MASE (ChronoFuse) | 0.609 |
| **相对最强单基下降** | **+11.60%** |

**多 seed 稳健性（5 个随机数据集）**：mean **+4.34% ± 5.14%**，4/5 为正，最差 -2.02%。

### 为什么多 seed 均值低于规范基准

时间序列预测组合的收益受限于**非平稳性导致的 CV/测试排名失稳**：
用历史 CV 预测「未来哪单个模型最优」本身就有不可约误差。
- oracle（用测试真值逐序列选最优基）上限：+9.27% ~ +21.19%（5 个数据集均为正）。
- 基于 CV 的纯逐序列选择：在部分数据集上为 **-11.89%**（反常被误选拖累）。
- ChronoFuse 通过「收缩到全局稳健基 + 非劣守卫」把上述灾难压到 -2.02%，
  同时保留大部分上升空间 —— 这正是它在注重稳健性的生产场景中的价值。

参考文献：Claeskens, Magnus, Vasnev & Wang (2016) *The forecast combination puzzle*；
Makridakis, Spiliotis & Assimakopoulos (2018) *The M4 Competition*。

## 不变量与验证

| 不变量 | 状态 |
| --- | --- |
| 无测试泄漏（所有拟合只用 train / 滚动 CV 内 train） | ✅ 结构保证 + 测试 |
| 确定性（同 seed 逐位一致，`max_abs_diff=0.0`） | ✅ `detcheck` |
| 逆误差加权收敛到最优基 | ✅ `test_chronofuse.py` |
| 收缩行为可预测（λ 半权重解析值） | ✅ `test_chronofuse.py` |
| 非劣守卫触发回退到最强单基 | ✅ `test_chronofuse.py` |
| 离线降级（无 statsmodels 时纯 numpy 五基可跑） | ✅ 测试覆盖 |

## 失败模式

1. **多步长 horizon 在非平稳序列上**：若序列在测试窗口发生训练期未见的机制变化，
   CV 选出的基可能系统性错误，融合相对最强单基为小幅负收益（实测最差 -2.02%）。
2. **极短序列**：CV 原点不足 2 个时回退单 hold-out 窗口，权重噪声上升；
   短于 `2*horizon+4` 的序列直接跳过（`n` 会小于理论值）。
3. **ARIMA 在本 DGP 上表现很差（9.274）**：已被排除出融合池，仅作 Tier-0 参考展示。

## 局限性

- 仅点预测（均值），无预测区间。
- 合成数据评测；真实数据表现需复核。
- Tier-0 依赖缺失时，基线对比口径会变（报告中 `reference_available` 标注，不伪造数值）。

## 复现

```bash
python -m chronoforge.cli demo        # 规范基准，应得 ChronoFuse 1.976 / 下降 +11.60%
python -m chronoforge.cli detcheck    # bit_identical = true
```
