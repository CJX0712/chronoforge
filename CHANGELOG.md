# Changelog

本项目所有值得记录的变更。格式遵循 [Keep a Changelog](https://keepachangelog.com/)。
作者：晨星。

## [0.1.0] - 2026-10-11

### Added
- **ChronoFuse 融合旗舰**：滚动原点交叉验证 → 逆误差加权 → 向全局最可靠基收缩 → 非劣守卫。
- 基预测器：Naive、SeasonalNaive、SES、Holt、Holt-Winters(add|mul)、AR(AIC 定阶)、Theta；
  Tier-0：ARIMA、ETS（statsmodels，可选，缺失自动降级）。
- 数据层：8 类 DGP 合成生成器（独立于全局 RNG，`RandomState` 自设种子）。
- 评估层：sMAPE / MAE / RMSE / MASE 与滚动原点基准编排。
- Pipeline：`run` / `determinism_check` / `save_report`，支持 `CHRONOFORGE_*` 环境变量覆盖。
- CLI：`demo`（端到端基准）与 `detcheck`（确定性校验）。
- 42 项测试、CI（Python 3.12/3.13 矩阵）、覆盖率 92%、ruff check/format 双绿。

### Changed
- 融合加权由「单窗口 hold-out」升级为「滚动原点 CV」：单窗口噪声会导致坏基撞大运后吞掉权重
  （实测某序列 AR 的 CV sMAPE 1.80 反超 HW 1.87，而测试上 AR 崩至 14.46，融合被拖到 7.63 vs HW 1.71）。
- 组合池由纯 numpy 五基扩展为「五基 + 可选 ETS」：ETS 为最强单基（2.235），纳入后
  ChronoFuse 降至 1.976，且基线对比更严格。
- 融合权重目标由 ridge（优化 MSE）改为逆 sMAPE 加权，消除目标-评测指标错位。

### Fixed
- `BaseForecaster.forecast()` 被子类以 `super().forecast()` 调用时必崩（抽象方法抛
  `NotImplementedError`）→ 改为独立的 `_ensure_fitted()`，全部子类同步修正。
- `ARForecaster` / `ARIMAForecaster` 的 AIC 比较索引错误（`best[1]` / `best[2]` 混用）→ 统一为 `best[2]`。
- `statsmodels_bridge` 中 `forecast()` 误调抽象方法 → 改为 `_ensure_fitted()`。
- `per_regime` 键被 `"seasonal".split("_s")` 截断导致季节类序列误归类 → 改用 `rsplit("_s", 1)`。
- 仓库结构升级为标准布局（`chronoforge/` 包 + 仓库根分离），修复 `pip install .` 将 `core`/`eval`
  暴露为顶层包的命名空间污染问题。
