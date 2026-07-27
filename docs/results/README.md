# 结果记录

结果按 task 组织，不按 method 或 model 拆目录：

```text
docs/results/<task>/
  结果汇总.md
  搜索曲线.png
```

训练 / 测试与搜索配置见 [`docs/实验配置.md`](../实验配置.md)。

`结果汇总.md` 统一结构为：实验参数 → 三次运行平均（保留全部已完成版本）→ 搜索曲线（只绘制 MCTS-AHD / PathWise / TraceAAD v4 / v5.1；OBP 无 v4 时不画 v4）→ 各次运行。只有重复实验和测试评估全部完成后，才更新结果页。
