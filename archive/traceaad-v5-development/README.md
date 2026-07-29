# TraceAAD V5 开发归档

本目录保存正式 V5 定稿前的实验工件，只用于追溯开发过程，不属于当前结果记录。
正式 V5 的代码位于 `llm4ad/method/traceaad_v5/`，实验位于各 task 的
`traceaad_v5/version5/`。

旧代码不复制为多个可运行包，统一由 Git 精确保存：

| 开发快照 | Git commit |
| --- | --- |
| 早期 V5 实现 | `d2ed669` |
| Action 与执行链路修正版 | `634f474` |
| 轨迹上下文与评分修正版 | `8fd1d0e` |
| 最终正式 V5 的原始快照 | `bb40ff0` |
| 后续联合修改快照 | `3035b01` |

`experiments/` 保存未被选为正式 V5 的四任务运行；`docs/evidence/` 保存当时的
冻结审计。开发结论的凝练版本见
`docs/studies/TraceAAD-V5开发复盘.md`。
