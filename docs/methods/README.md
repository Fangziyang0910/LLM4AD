# 方法设计导航

近期开发：[V10.10：轨迹搜索与一次有界错误修复](TraceAAD-V10.10-机制设计.md)，从 [V10.9](TraceAAD-V10.9-机制设计.md)复制后移除新结构试用与直接子代试错上下文，并增加最小错误处理；尚未启动正式搜索，见[运行说明](../../experiments/traceaad_v10_10/README.md)。

按版本保存设计，版本号不表示效果排序。表中的实现入口只表示当前仓库存在对应文件，不表示实验已完成或机制已验证；结果与运行状态以实验记录为准。

近期设计：[V10.8](TraceAAD-V10.8-机制设计.md) 已实现近期连续代码转移与同代码组机会计量，机制收益待实验验证；[V10.7R](TraceAAD-V10.7R-机制设计.md) 保留独立证据基线与冻结身份。此前研究来源见[上下文采样构想](../ideas/2026-09-07-按设计任务采样与组织历史上下文/构想.md)。

| 版本 | 设计 | 实现与分析入口 |
| --- | --- | --- |
| V10.10 | [机制设计](TraceAAD-V10.10-机制设计.md) | [实现目录](../../llm4ad/method/traceaad_v10_10)；[运行说明](../../experiments/traceaad_v10_10/README.md)，已实现，正式实验待启动 |
| V10.9 | [机制设计](TraceAAD-V10.9-机制设计.md) | [实现目录](../../llm4ad/method/traceaad_v10_9)；[运行说明](../../experiments/traceaad_v10_9/README.md) |
| V9 | [机制设计](TraceAAD-V9-机制设计.md) | 历史设计；当前无同名实现目录；[分析](../analysis/版本综合分析/TraceAAD-V9机制分析.md) |
| V9.7 | [机制设计](TraceAAD-V9.7-机制设计.md) | [实现目录](../../llm4ad/method/traceaad_v9_7)；[分析](../analysis/版本综合分析/V9.7机制诊断/结论.md) |
| V9.14 | [机制设计](TraceAAD-V9.14-机制设计.md) | [实现目录](../../llm4ad/method/traceaad_v9_14) |
| V9.15 | [机制设计](TraceAAD-V9.15-机制设计.md) | 历史设计；当前无同名实现目录；[分析](../analysis/版本综合分析/TraceAAD-V9.15机制诊断.md) |
| V9.16 | [机制设计](TraceAAD-V9.16-机制设计.md) | [实现目录](../../llm4ad/method/traceaad_v9_16)；[分析](../analysis/版本综合分析/TraceAAD-V9.16机制诊断.md) |
| V9.17 | [机制设计](TraceAAD-V9.17-机制设计.md) | 历史设计；当前无同名实现目录；[分析](../analysis/版本综合分析/V9.16-V9.17对照分析/结论.md) |
| V9.18 | [机制设计](TraceAAD-V9.18-机制设计.md) | 历史设计；当前无同名实现目录；[分析](../analysis/版本综合分析/TraceAAD-V9.18-R0机制与证据分析.md) |
| V9.19 | [机制设计](TraceAAD-V9.19-机制设计.md) | 历史设计；当前无同名实现目录；[分析](../analysis/版本综合分析/TraceAAD-V9.19新版机制设计审视.md) |
| V9.20 | [机制设计](TraceAAD-V9.20-机制设计.md) | 历史设计；当前无同名实现目录 |
| V9.21 | [机制设计](TraceAAD-V9.21-机制设计.md) | 历史设计；当前无同名实现目录；[分析](../analysis/版本综合分析/TraceAAD-V9.21机制首跑分析.md) |
| V9.22 | [机制设计](TraceAAD-V9.22-机制设计.md) | 历史设计；当前无同名实现目录；[分析](../analysis/版本综合分析/TraceAAD-V9.22机制设计与实现分析.md) |
| V10 | [机制设计](TraceAAD-V10-机制设计.md) | 历史设计；当前无同名实现目录；[分析](../analysis/版本综合分析/2026-09-06-V10系列实验核查/TraceAAD-V10系列实验核查-20260906.md) |
| V10.1 | [机制设计](TraceAAD-V10.1-机制设计.md) | [实现目录](../../llm4ad/method/traceaad_v10_1)；[实验结果](../experiments/整体比较/2026-09-02-TraceAAD-V10.1完整搜索/结果.md) |
| V10.2 | [机制设计](TraceAAD-V10.2-机制设计.md) | [实现目录](../../llm4ad/method/traceaad_v10_2)；[实验结果](../experiments/整体比较/2026-09-02-TraceAAD-V10.2完整搜索/结果.md) |
| V10.3 | [机制设计](TraceAAD-V10.3-机制设计.md) | [实现目录](../../llm4ad/method/traceaad_v10_3)；[实验结果](../experiments/整体比较/2026-09-04-TraceAAD-V10.3完整搜索/结果.md) |
| V10.4 | [机制设计](TraceAAD-V10.4-机制设计.md) | [实现目录](../../llm4ad/method/traceaad_v10_4)；[实验结果](../experiments/整体比较/2026-09-04-TraceAAD-V10.4完整搜索/结果.md) |
| V10.5 | [机制设计](TraceAAD-V10.5-机制设计.md) | [运行说明](../../experiments/traceaad_v10_5/README.md)；[实验结果](../experiments/整体比较/2026-09-05-TraceAAD-V10.5完整搜索/结果.md) |
| V10.6 | [机制设计](TraceAAD-V10.6/机制设计.md) | [运行说明](../../experiments/traceaad_v10_6/README.md) |
| V10.7 | [机制设计](TraceAAD-V10.7-机制设计.md) | [运行说明](../../experiments/traceaad_v10_7/README.md) |
| V10.7R | [机制设计](TraceAAD-V10.7R-机制设计.md) | [实现目录](../../llm4ad/method/traceaad_v10_7)；[运行说明](../../experiments/traceaad_v10_7/README.md) |
| V10.8 | [方法报告：完整流程、模块机制与示例](TraceAAD-V10.8-机制设计.md) | [实现目录](../../llm4ad/method/traceaad_v10_8)；[运行说明](../../experiments/traceaad_v10_8/README.md) |

## 历史与讨论

- [历史机制探索](TraceAAD-历史机制探索.md)：记录早期设计如何演进；跨版本经验见[机制尝试](../knowledge/TraceAAD机制尝试.md)。
- [V10.6 讨论稿](TraceAAD-V10.6/讨论稿.md)：保留已被后续设计替代的推理过程，实施时读取同目录机制设计。

单篇设计使用 `TraceAAD-V版本-机制设计.md`；同版本有需要一起保存的讨论材料时再建版本目录。
