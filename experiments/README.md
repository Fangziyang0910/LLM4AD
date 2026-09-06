# 实验导航

科研背景见 [文档导航](../docs/README.md)。各方法的运行脚本位于 `experiments/<method>/`，近期版本的结果位于其 `results/<task>/<run>/`；历史布局以对应实验记录为准。共享后端、任务构建与客户端工具位于 `runners/`。原始工件只留本地。

## 近期开发

- **V10.5**：[机制设计](../docs/methods/TraceAAD-V10.5完整机制设计.md)、[运行与恢复](traceaad_v10_5/README.md)、[启动记录](traceaad_v10_5/launch_20260905.md)。从批次 manifest 查看实际进度。
- **V10.4**：[机制设计](../docs/methods/TraceAAD-V10.4完整机制设计.md)，运行入口 `traceaad_v10_4/run.py`、`launch.py`。
- **V10.3**：[机制设计](../docs/methods/TraceAAD-V10.3完整机制设计.md)，运行入口 `traceaad_v10_3/run.py`、`launch.py`。
- **V10.2**：[机制设计](../docs/methods/TraceAAD-V10.2完整机制设计.md)、[实验结果](../docs/experiments/机制实验/2026-09-02-TraceAAD-V10.2正式实验/结果.md)。
- **V10.1**：[机制设计](../docs/methods/TraceAAD-V10.1完整机制设计.md)，运行入口 `traceaad_v10_1/run.py`、`launch.py`。

## 实验默认设置

新正式比较采用 **Qwen3.8-27B、每路 1000 次真实 evaluator 调用**。采样参数为 temperature=1.0、top_p=0.95、top_k=20，客户端显式下发。thinking 开关与采样参数分别记录；V10.5 使用 `enable_thinking=False`。具体批次以对应协议及 `run_config.json` 为准。

初始化、评价失败和超时计入真实评价预算；仅生成或解析失败不算 evaluator 调用。冒烟与诊断实验采用适合问题的小预算，单独记录。重复数、数据划分与运行容量查对应版本；已启动批次沿用原配置与恢复协议。

这里的 Qwen 是被研究的算法生成模型，科研协作者使用 Astra 不改变实验配置。

## 已有比较与历史探索

[V9 系列主实验结果](../docs/experiments/主实验/结果.md)及[配置](../docs/experiments/主实验/配置.md)保留 Qwen3.6 时期的比较口径。外部对照包括 EoH、ReEvo、MCTS-AHD、PathWise、CALM；其他实验包还包括 ShinkaEvo 等方法。

其他版本与批次见 [历史版本](../docs/experiments/其他实验/历史版本.md)、[机制实验](../docs/experiments/机制实验/)及 [工作日志](../docs/worklog/)。引用旧结果时沿用其原始模型、预算与任务设置。
