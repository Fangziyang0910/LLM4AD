# TraceAAD V9.18-R0 实验协议

本文记录实现前固定的实验顺序。结果写入各运行工件和正式结果页后，本文
只补充实际完成的数字，不根据中途曲线改写预注册规则。

## 1. 实验问题

| 阶段 | 比较 | 可回答的问题 |
| --- | --- | --- |
| P0 | 固定锚点 History-off/on | 父代来时路与 Global-Facts-Lite 是否改变下一步提议 |
| A | `q-atomic` / `q+O-atomic` | 边界机会项是否改变锚点路由和有限预算结果 |
| B | 原 Explore / Global-Facts-Lite | 极短全局事实是否改善 Explore 提议 |
| C | A 与 B 的联合臂 | 两个已经识别的改动共同作用如何 |

P0 是 transition 证据，A/B/C 是完整搜索证据。任何阶段的过程激活都不
等于质量改善。

## 2. 固定协议

- 模型逻辑名：Qwen3.6-27B；不同服务源按同一模型记录。
- 每路八个有效根，1000 个 primary evaluator slots。
- `Refine=0.7`、`Explore=0.3`，temperature、max output tokens、repair
  规则和 evaluator 与 V9.16 正式口径一致。
- A 阶段两臂从同一个精确八根 checkpoint 分叉，称 matched initialization。
  评分不同后搜索轨迹必然分叉，不称逐 slot paired trajectory。
- held-out 只在搜索结束后对每个 repeat 的 train best 独立评估，不反馈在线。
- 修复调用不占 primary slot，但单独报告 evaluator calls、LLM calls、repair
  次数、timeout 和墙钟成本。

## 3. 运行级安排

第一批使用 30 个运行槽位：

| 臂 | 任务 | 重复 | 路数 |
| --- | --- | ---: | ---: |
| A0 `q-atomic` | 5 任务 | 3 | 15 |
| A1 `q+O-atomic` | 5 任务 | 3 | 15 |

每个任务×重复是一个独立运行单位。服务容量按 server3、server3b、
server1 的既有并发上限均衡分配；若本地 3 路被纳入，总槽位不得超过实际
服务容量，且运行配置记录 backend 和 request metadata。

B 阶段不与 A 阶段混跑，等 A 的过程审计和初步结果确认后再占用下一批
槽位。这样 30 路用于识别最核心的评分问题，不把所有机制同时铺开。

## 4. 过程审计

每个 slot 必须记录：

- 选择前的 `q`、`O`、`n_after`、`sigma_q`、`S` 快照；
- selected anchor、operator、operator draw、decision index 和 request seed；
- prompt hash、prompt chars、事实板 hash、事实板 omitted 标记；
- primary candidate、repair calls、status、fitness、failure kind、重复和 no-op；
- 选择后节点事实、`n_after`、global best 和 best-at-budget。

每个 run summary 必须包含：`status=finished`、`budget_slots=1000`、根质量池、
`sigma_q`、`lambda_O`、机会项激活率、选择熵、top-k 份额、不同锚点数、
valid/invalid/timeout/repair/duplicate 率、LLM/evaluator calls 和成本字段。

## 5. 预先判定

### 5.1 A 阶段评分

先检查机制是否运行：`O` 非零、边界重访概率随 `n_after` 衰减、评分实际
改变选择。再看质量：best-at-budget、search best 和 held-out。

以下情况不称评分成功：机会项长期为零；只改变日志不改变选择；低质量入口
占用比例明显上升；入口覆盖系统性下降；过程改变但最终质量没有改善。

### 5.2 B 阶段算子

Global-Facts-Lite 只有在固定锚点上改变提议且不造成有效率、timeout、repair
或重复的系统性恶化后，才进入完整搜索。完整参考程序、Diagnosis 强制契约
和代表锚点板不在本阶段。

### 5.3 C 阶段联合

联合结果只能回答整体搜索是否改善。五任务整体不劣要求：三重复 held-out
完成，平均名次不低于 V9.16，且没有任务族的系统性退化。单任务三重复同向
改善只能写成条件性有效。

## 6. 结果状态词

- `mechanism_not_run`：配置、日志或 checkpoint 显示机制未激活；
- `ran_no_improvement`：过程改变，但 search/held-out 没有改善；
- `ran_harmful`：过程改变并伴随有效率、覆盖、成本或质量恶化；
- `conditional_work`：预先指定任务族达到三重复质量门槛；
- `general_upgrade`：五任务整体不劣且没有系统性副作用。

## 7. 证据链接

机制规范见[TraceAAD V9.18-R0 完整机制设计](../methods/TraceAAD-V9.18完整机制设计.md)。
历史判断见[TraceAAD V9.16 与 V9.17 对照分析](../analysis/TraceAAD-V9.16-V9.17对照分析.md)。

## 8. 自动执行

统一调度入口为 `experiments/runners/traceaad/launch_v918.py`：

````text
uv run python -m experiments.runners.traceaad.launch_v918 --watch
````

调度器识别已有 tmux、完成 summary 和 V9.18 checkpoint，因而可以在中断后
重新启动。搜索全部完成后，它按任务×臂运行 held-out，并调用过程审计脚本
写入 `docs/analysis/traceaad_v918_process/summary.json` 与
`docs/analysis/TraceAAD-V9.18-A阶段过程审计.md`。审计只把完整的
`status=finished`、`budget_slots=1000` 运行纳入聚合；partial 工件保留但不
进入正式质量结论。
