# TraceAAD V9.5 简洁历史生成识别实验

> 状态：已完成。72 个 anchor、216 个 A/B pair、432 次生成均已收口。这是低成本的 generation-interface probe，不是完整搜索消融。

## 问题

对于已经积累至少两条局部历史证据的同一个真实 executable anchor，加入简洁的
`Source + Idea + Result + Fitness transition` 是否改善 Qwen3.6-27B 的下一步算法修改？

本实验不外推到 fresh anchor，不回答完整搜索或 held-out 效果，也不比较 actual diff
和 generation operator。

## 设计

- 独立单位：一次 V9.5 `anchor_selected → evidence_built` 真实 snapshot。
- 来源：V9.5 正式批次 `20260811_171029` 中状态为 `finished` 的 runs。
- 防泄漏：只使用当时 `evidence_built` 记录的 attempt IDs；最终 checkpoint 只用于读取不可变的代码与 attempt 事实。
- 资格：当时至少有两条已选 Evidence；按 evaluator-input code hash 去重。
- 分层：在每个 task 的 eligible deduplicated pool 内，按 directed \(q\) 排序后等数分为 low / middle / high 三层。
- 抽样：每层 6 个 anchor，优先在已完成 repeats 间均衡；某 repeat 在该层数量不足时，从同层其他 repeat 的剩余 pool 补齐。
- 规模：4 tasks × 3 strata × 6 anchors = 72 个 anchor。
- 重复：每个 anchor 在 A/B 下各 3 次，共 432 responses。三次采样是 anchor 内重复，不当成独立 anchor。

## A/B 条件

`No History` 与 `Concise History` 共享：Task、Current Fitness、Current Code、逐字相同的
Improvement Instruction、Output Contract、temperature=1.0、sampling seed、8192-token 最大输出和
32768-token 总上下文。唯一系统差异是 B 在 Current Code 与 Instruction 之间加入：

```text
[Concise Search History]
Event i
Source: Formation / Direct
Idea: ...
Result: improve / plateau / regress / invalid
Fitness: parent -> child
```

最多使用当时 EvidenceBuilder 已选的 8 条事件；不含 actual diff、LOC、subtree result、
后代结果或 LLM 历史摘要。

每个 `anchor × replicate` 构成一个 pair，A/B 共享 sampling seed。每个 task × stratum 内的
AB / BA 执行顺序精确各半，每对相邻执行。

## 完整性与度量

每个 anchor 保存 canonical prompt components，并强制验证：

```text
hash(prompt_A) == hash(prompt_B removing the exact history block)
```

只使用 directed quality：

\[
\Delta q=q_{child}-q_{parent}.
\]

主要报告 valid rate、条件于 valid 的 \(\Delta q\)、parent improvement rate、line change ratio 与
absolute LOC change。某 anchor × condition 三次全部 invalid 时，其 conditional \(\Delta q\) 为缺失，不填人工 penalty；
valid rate 照常进入配对分析。

分析先在 anchor 内聚合三次采样，再计算 B−A 配对差值。按 task 和 quality stratum 分层报告，
附 anchor-level bootstrap interval 和 B 更好 / 持平 / A 更好的 anchor 计数。不要求四个 task 和所有质量层同时变好；
重点解释总体方向、task interaction、有效性与修改幅度的 trade-off。

## 复现入口

- 抽样与运行：`experiments/runners/traceaad/generation_probe.py`
- 分析：`experiments/analysis/analyze_traceaad_generation_probe.py`
- 测试：`tests/test_traceaad_generation_probe.py`
- 本地正式工件：`experiments/generation_probe/20260812_v95_concise_history_probe/`

原始 prompt、response、生成代码、evaluator 结果和分片状态只留本地，不进入 Git。

## 结果

完整性审计通过：432 个 trial ID 全部唯一，无缺失、无重复结果行；每个
task 在 A/B 下各 54 次；24 个 shard 全部以 `finished` 结束。Prompt 在构造时已强制验证：
从 B 移除精确 History block 后与 A 逐字相同。

| Task | Valid: No History | Valid: History | Conditional Δq: No History | Conditional Δq: History | 配对差 B−A [95% bootstrap CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| CVRP | 48/54 (88.9%) | 52/54 (96.3%) | -1.657 | -0.472 | +1.185 [0.452, 1.941] |
| OBP | 53/54 (98.1%) | 54/54 (100.0%) | -428.336 | -572.755 | -144.419 [-520.744, 254.567] |
| OP | 51/54 (94.4%) | 49/54 (90.7%) | -1.745 | -1.147 | +0.598 [-0.179, 1.402] |
| TSP | 41/54 (75.9%) | 32/54 (59.3%) | -1.210 | -1.039 | +0.172 [-0.365, 0.663] |

Conditional Δq 先在 `anchor × condition` 内对 valid repeats 取均值，再做 anchor-level B−A
配对。TSP 有 3 个 anchor 在某一条件下三次全 invalid，因而 conditional 指标的配对
anchor 数为 15；没有对缺失的 Δq 填惩罚值。四任务不合并原始 Δq 幅度，因为尺度不同。

修改行为显示了更稳定的跨任务信号。History 使 conditional line-change ratio 在 CVRP
从 0.845 降至 0.530，OP 从 0.745 降至 0.615，TSP 从 0.807 降至 0.732；
三者的 anchor-level 95% interval 均不跨 0。OBP 从 0.749 到 0.740，没有可分辨差异。
在 69 个同时存在 conditional Δq 的 anchor 中，History 方向更好 44 个、No History
更好 23 个、2 个持平；这只是跨任务方向计数，不是汇总效应量。

History 使平均 prompt 增加 535--842 tokens；最长任务 CVRP 的平均输入从 2737
增至 3580 tokens，距 32768 总上下文上限很远。平均 response tokens 在 CVRP
增加 780，OBP 增加 137，OP 减少 40，TSP 减少 5；因此“修改幅度变小”
不等于完整代码输出更短。Exact-parent no-op 只出现 5 次，不足以解释主要差异。

TSP invalid 差异主要由 20 秒 evaluator timeout 驱动：History 20 次，No History
8 次。为区分候选算法本身过慢与并发 CPU 竞争，在所有正式 shard 结束后，对这
28 份原 timeout 代码用同一 evaluator 进行空闲、顺序复评。History 有 8/20 恢复有效，
No History 有 3/8 恢复有效；恢复率相近，仍有 12 对 5 份稳定 timeout。这说明并发
放大了绝对 invalid 数，但不足以解释 History 条件下更多慢候选的条件差异。
复评仅作诊断，不替换冻结的 432 条主结果。

## 当前判断

Concise History 确实改变了 Qwen3.6-27B 的单步生成分布，最清楚的行为是让修改更贴近
current code。质量上，CVRP 有明确正信号；OP 和 TSP 的平均方向为正但不确定；
OBP 方向为负且不确定。TSP 同时出现 16.7 个百分点的 valid-rate 下降，是不能由
conditional Δq 掩盖的代价。

因此，这一轮支持“简洁试错历史可以提供单步生成价值”，但不支持“对所有任务都改善”。
它也不证明完整搜索或 held-out 表现会提高。下一轮若增加 compact actual change，
应保留同一批 anchor 与相同生成合同，并继续把 TSP 运行超时作为不可忽略的价值维度。
