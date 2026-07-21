# TraceAAD v3 算子选择观察

## 1. 审计范围与数据完整性

本观察是 2026-07-21 16:18 +08 的过程快照。只读取 `experiments/*/traceaad/version3/*` 下已落盘的 `run_config.json`、`logs/method_events.jsonl`、`logs/method_state.jsonl`、`logs/samples/`、checkpoint 和可用的 `run_summary.json`；没有修改实验、配置、代码、结果页或既有文档。纳入 5 个 task 的 14 个已有有效过程记录 run；尚无目录和过程记录的 knapsack rep3、tsp_gls 三个 rep 没有纳入。

所有纳入 run 的共同搜索设置是预算 1000、`n_init=4`、每 attempt 两个 action、活跃池上限 160、最大轨迹长度 8、novelty threshold 0.92、算子 UCB 系数 0.5。LLM 均为 temperature 1.0，但各 rep 的服务端/量化模型不同，因而 rep 间也不能当作严格固定随机种子对照。各 task 的训练评估参数和实际模型写在对应 run 的 `run_config.json` 中。

重建单位为一次 `attempt_id`：以 `operator_selection` 为决策起点，与同一 id 的 `iteration_start`、`operator_batch` 连接；再以 iteration/node id 连接 `child_accepted` 与 `best_updated`。其中 `selected_value=(Q,P,D)` 分别是 quality、potential、diversity，`selected_scalar_value` 是其当时的标量 value。`scores` 是选择前的 UCB 分数，`operator_batch.portfolio` 是本批次记账后的 attempts 与 mean reward。

可可靠恢复：候选算子集、选择算子和其 UCB 分数、原始选中轨迹与算子重定向后的 target trajectory、base node/reason、Q/P/D/value、非 fresh-start 的 action、有效评估结果、batch reward、novelty 状态，以及 run 内 best 更新。不能可靠恢复的字段统一标记为 missing：轨迹的当时 `visit count` 没有落盘；`novelty_jump` 的 fresh start 没有 parent-edge delta 或 action 字段；样本 JSON 虽有程序和 sample order，但没有稳定 node-id 主键。因而没有从文本提示词猜补这些字段。

统计口径：`valid eval` 为 `program_evaluated.status=ok/(ok+eval_failed)`；improve/plateau/regress 为已进入 `operator_batch.candidates` 的有效候选（fresh-start 的 outcome 依据 batch anchor）；active/rejected 为有效 child 的 novelty gate 最终状态；delta 为 `score-reference_score`，所有本批次 task 均为 maximize 定义。delta 的 Q1/median/Q3 只在 task 内解释，不能横向比较。初/中/后按 `profiler_sample_order/1000` 的 0--33%/33--66%/66--100% 划分，未完成 run 的未消耗预算不被伪装为后期数据。

## 2. 各 task 与 run 清单

下表的 `eval` 是审计快照时已记录评估数；`sel/batch/state` 是算子选择、批次收尾和轨迹状态事件数。`完整`表示三类事件数量相等且有 `run_summary.json`；`进行中`的最后一个 selection 可能尚未写入 batch，不能把它的未完成动作当作失败。

| task | run 路径 | 状态 | eval/预算 | sel/batch/state | 过程文件与可分析范围 |
| --- | --- | --- | --- | --- | --- |
| cvrp_aco | `/home/fang/code/LLM4AD/LLM4AD/experiments/cvrp_aco/traceaad/version3/20260720_233339_cvrp_v3_rep1` | finished | 1000/1000 | 500/500/500 | 完整：events、state、samples、summary、best |
| cvrp_aco | `/home/fang/code/LLM4AD/LLM4AD/experiments/cvrp_aco/traceaad/version3/20260720_233339_cvrp_v3_rep2` | finished | 1000/1000 | 506/506/506 | 完整：events、state、samples、summary、best |
| cvrp_aco | `/home/fang/code/LLM4AD/LLM4AD/experiments/cvrp_aco/traceaad/version3/20260720_233339_cvrp_v3_rep3` | running | 988/1000 | 494/493/494 | 缺 summary；最后 selection 未闭合，仅分析已完成 493 batch |
| knapsack_construct | `/home/fang/code/LLM4AD/LLM4AD/experiments/knapsack_construct/traceaad/version3/20260720_233339_kp_v3_rep1` | running | 78/1000 | 38/37/38 | 缺 summary；仅初期 7.8% 预算，不能作阶段结论 |
| knapsack_construct | `/home/fang/code/LLM4AD/LLM4AD/experiments/knapsack_construct/traceaad/version3/20260720_233339_kp_v3_rep2` | running | 51/1000 | 25/24/25 | 同上，仅初期 5.1% 预算 |
| online_bin_packing | `/home/fang/code/LLM4AD/LLM4AD/experiments/online_bin_packing/traceaad/version3/20260720_233339_obp_v3_rep1` | running | 507/1000 | 254/253/254 | 缺 summary；已覆盖初/中期的一部分 |
| online_bin_packing | `/home/fang/code/LLM4AD/LLM4AD/experiments/online_bin_packing/traceaad/version3/20260720_233339_obp_v3_rep2` | running | 825/1000 | 413/412/413 | 缺 summary；有部分后期样本 |
| online_bin_packing | `/home/fang/code/LLM4AD/LLM4AD/experiments/online_bin_packing/traceaad/version3/20260720_233339_obp_v3_rep3` | running | 287/1000 | 143/142/143 | 缺 summary；仅初期 |
| orienteering_construct | `/home/fang/code/LLM4AD/LLM4AD/experiments/orienteering_construct/traceaad/version3/20260720_233339_op_v3_rep1` | running | 928/1000 | 466/465/466 | 缺 summary；有后期样本 |
| orienteering_construct | `/home/fang/code/LLM4AD/LLM4AD/experiments/orienteering_construct/traceaad/version3/20260720_233339_op_v3_rep2` | running | 601/1000 | 300/299/300 | 缺 summary；初/中期为主 |
| orienteering_construct | `/home/fang/code/LLM4AD/LLM4AD/experiments/orienteering_construct/traceaad/version3/20260720_233339_op_v3_rep3` | running | 653/1000 | 326/325/326 | 缺 summary；初/中期为主 |
| tsp_construct | `/home/fang/code/LLM4AD/LLM4AD/experiments/tsp_construct/traceaad/version3/20260720_233339_tspc_v3_rep1` | finished/evaluated | 1000/1000 | 501/501/501 | 完整：events、state、samples、summary、best、测试评估 |
| tsp_construct | `/home/fang/code/LLM4AD/LLM4AD/experiments/tsp_construct/traceaad/version3/20260720_233339_tspc_v3_rep2` | finished/evaluated | 1000/1000 | 503/503/503 | 同上 |
| tsp_construct | `/home/fang/code/LLM4AD/LLM4AD/experiments/tsp_construct/traceaad/version3/20260720_233339_tspc_v3_rep3` | finished/evaluated | 1000/1000 | 506/506/506 | 同上 |

所有 14 个 run 都有 `method_events.jsonl`、`method_state.jsonl`、分段 samples、`samples_best.json` 和 checkpoint；5 个闭环 run 有 `run_summary.json`。因此算子行为可作快照审计，但只有 TSP 三个 rep 与 CVRP 两个 rep 已具备完整 1000-budget 过程；其余结论应随运行结束复核。

## 3. 算子选择总体分布

总计 4,975 次已记录选择。`eligible` 是该算子在当时满足 trigger 的次数；选择率是 selected/eligible，不是四算子间无条件比例。

| 算子 | eligible | selected | 在可用条件下选择率 |
| --- | ---: | ---: | ---: |
| endpoint_refine | 4,975 | 2,830 | 56.9% |
| backtrack_branch | 4,830 | 889 | 18.4% |
| mechanism_crossover | 4,975 | 1,003 | 20.2% |
| novelty_jump | 4,975 | 253 | 5.1% |

这不是均匀轮换。选择从初期的 endpoint/backtrack/crossover/novelty = 43.2%/24.6%/22.5%/9.6%（n=2,021），收缩为中期 60.9%/16.4%/20.2%/2.4%（n=1,715），后期 73.6%/8.8%/16.3%/1.3%（n=1,239）。UCB 确实改变了分布，且其主要表征是压低 fresh-start novelty、提高 endpoint refine；它不是只做了一次四算子覆盖后完全固定，因为 crossover/backtrack 仍在后期被选中，但集中趋势很强。

## 4. 按 task 的统计

以下每格依次为：可用、选择、选择率、valid eval、候选 outcome (I/P/R)、平均 batch reward、delta Q1/median/Q3、active/rejected、best 更新。reward 是 task 内标准化后再经 `tanh` 的 batch 平均；delta 保留 task 原尺度。

### cvrp_aco（3 run，1,500 次选择）

| 算子 | 可用/选择/率 | valid eval | I/P/R | reward | delta Q1/Med/Q3 | active/rejected | best |
| --- | --- | --- | --- | ---: | --- | --- | ---: |
| endpoint | 1500/918/61.2% | 1681/1821 (92.3%) | 437/292/952 | -0.1076 | -0.1319/-0.01629/0.003923 | 1047/634 | 38 |
| backtrack | 1481/202/13.6% | 392/402 (97.5%) | 72/38/282 | -0.2014 | -2.935/-0.2004/0 | 288/104 | 6 |
| crossover | 1500/332/22.1% | 627/657 (95.4%) | 157/46/424 | -0.1506 | -0.6119/-0.07566/0.0002158 | 486/141 | 9 |
| novelty | 1500/48/3.2% | 85/96 (88.5%) | 1/0/84 | -0.3645 | -7.882/-5.113/-2.359 | 85/0 | 1 |

### knapsack_construct（2 个短 run，63 次选择）

| 算子 | 可用/选择/率 | valid eval | I/P/R | reward | delta Q1/Med/Q3 | active/rejected | best |
| --- | --- | --- | --- | ---: | --- | --- | ---: |
| endpoint | 63/22/34.9% | 41/42 (97.6%) | 12/16/13 | -0.0238 | -3.406/0/0.1562 | 33/8 | 1 |
| backtrack | 30/15/50.0% | 30/30 (100%) | 9/7/14 | -0.0741 | -12.51/0/0.5078 | 25/5 | 1 |
| crossover | 63/17/27.0% | 31/31 (100%) | 9/2/20 | -0.0740 | -24.03/-3.406/1.188 | 30/1 | 1 |
| novelty | 63/9/14.3% | 18/18 (100%) | 2/0/16 | -0.1825 | -165.1/-14.42/-4.273 | 17/1 | 2 |

仅 6.3% 预算，且全部是初期，不能据此声称 backtrack 的 50.0% 选择率或任何 reward 排序稳定。

### online_bin_packing（3 个进行中 run，810 次选择）

| 算子 | 可用/选择/率 | valid eval | I/P/R | reward | delta Q1/Med/Q3 | active/rejected | best |
| --- | --- | --- | --- | ---: | --- | --- | ---: |
| endpoint | 810/420/51.9% | 818/833 (98.2%) | 148/259/410 | -0.1059 | -26.8/-0.2/0 | 449/369 | 14 |
| backtrack | 762/171/22.4% | 331/338 (97.9%) | 92/66/173 | -0.0941 | -42.6/-0.2/0.5 | 138/193 | 2 |
| crossover | 810/118/14.6% | 222/235 (94.5%) | 56/27/139 | -0.1842 | -52.4/-10.6/0.15 | 187/35 | 7 |
| novelty | 810/101/12.5% | 171/201 (85.1%) | 2/11/158 | -0.1786 | -78.6/-50.2/-24.8 | 165/6 | 2 |

### orienteering_construct（3 个进行中 run，1,092 次选择）

| 算子 | 可用/选择/率 | valid eval | I/P/R | reward | delta Q1/Med/Q3 | active/rejected | best |
| --- | --- | --- | --- | ---: | --- | --- | ---: |
| endpoint | 1092/629/57.6% | 1199/1251 (95.8%) | 244/278/675 | -0.1204 | -0.6069/-0.02625/0 | 703/496 | 21 |
| backtrack | 1064/233/21.9% | 448/461 (97.2%) | 98/61/289 | -0.1900 | -2.712/-0.2522/0 | 289/159 | 13 |
| crossover | 1092/194/17.8% | 369/387 (95.3%) | 74/51/244 | -0.1924 | -2.105/-0.1819/0 | 290/79 | 4 |
| novelty | 1092/36/3.3% | 67/71 (94.4%) | 2/0/65 | -0.4111 | -5.012/-2.944/-2.107 | 67/0 | 2 |

### tsp_construct（3 个完整 run，1,510 次选择）

| 算子 | 可用/选择/率 | valid eval | I/P/R | reward | delta Q1/Med/Q3 | active/rejected | best |
| --- | --- | --- | --- | ---: | --- | --- | ---: |
| endpoint | 1510/841/55.7% | 1573/1668 (94.3%) | 369/339/865 | -0.1388 | -0.1631/-0.005808/0 | 980/593 | 36 |
| backtrack | 1493/268/18.0% | 498/531 (93.8%) | 72/133/293 | -0.1921 | -0.5328/-0.04675/0 | 360/138 | 9 |
| crossover | 1510/342/22.6% | 628/672 (93.5%) | 134/76/418 | -0.1836 | -0.6196/-0.08928/0 | 535/93 | 11 |
| novelty | 1510/59/3.9% | 106/117 (90.6%) | 1/2/103 | -0.3570 | -2.875/-1.477/-0.7005 | 106/0 | 1 |

## 5. 按搜索阶段的统计

| task | 初期 n: endpoint/backtrack/crossover/novelty | 中期 n: endpoint/backtrack/crossover/novelty | 后期 n: endpoint/backtrack/crossover/novelty |
| --- | --- | --- | --- |
| cvrp_aco | 496: 46.6/23.0/22.8/7.7% | 496: 66.5/11.5/20.6/1.4% | 508: 70.3/6.1/23.0/0.6% |
| knapsack_construct | 63: 34.9/23.8/27.0/14.3% | 0: missing | 0: missing |
| online_bin_packing | 472: 42.4/23.3/16.9/17.4% | 255: 62.0/22.4/10.2/5.5% | 83: 74.7/4.8/14.5/6.0% |
| orienteering_construct | 493: 45.0/28.8/20.9/5.3% | 464: 60.3/19.2/18.5/1.9% | 135: 94.1/1.5/3.7/0.7% |
| tsp_construct | 497: 40.0/23.5/28.4/8.0% | 500: 55.2/15.8/26.6/2.4% | 513: 71.3/14.0/13.3/1.4% |

从完整 TSP、接近完成 CVRP 和 OP 的一致方向看，调度后期持续沿用初期形成的 endpoint 偏好，OP 后期达到 94.1%。这支持“operator score 改变行为”，但也构成强归纳偏置的过程证据：novelty 的可用性没有消失，却几乎被排除。OBP 的完成度不齐、Knapsack 没有中后期，不能作为这个趋势的独立确认。

## 6. 算子评分与搜索结果的关系

为避免跨 run 的 score 尺度混合，以下按每个 run 内有限的被选 UCB 分数分为低/中/高三分位，再合并 task。它检验“高分选择后，这一批是否关联更高 reward/更多 improve”，不检验因果。

| task | 低分：batch / reward / improve | 中分：batch / reward / improve | 高分：batch / reward / improve |
| --- | --- | --- | --- |
| cvrp_aco | 498 / -0.1331 / 226/920 | 492 / -0.1463 / 233/918 | 497 / -0.1302 / 205/924 |
| knapsack_construct | 19 / -0.0624 / 16/38 | 16 / -0.0787 / 7/32 | 18 / -0.0957 / 8/35 |
| online_bin_packing | 268 / -0.1713 / 120/521 | 262 / -0.1283 / 81/498 | 265 / -0.0706 / 94/499 |
| orienteering_construct | 361 / -0.1842 / 103/679 | 356 / -0.1807 / 126/679 | 360 / -0.1061 / 184/701 |
| tsp_construct | 501 / -0.1826 / 183/919 | 496 / -0.2189 / 194/914 | 501 / -0.0984 / 192/951 |

观察：OBP、OP、TSP 的高分三分位随后平均 reward 较高；CVRP 没有单调关系，短 Knapsack 反向。这是初步的过程关联，不能证明 score 学到了算子固有能力：score 同时含历史 reward、尝试次数的探索项，且所选 operator 改变 target/base/context。没有固定轨迹、固定 action 分布、固定模型采样下的调度对照，不能说它优于均匀或固定概率。

新轨迹接受率也不支持把 novelty 解释成“有效多样性”：CVRP/TSP/OP 中 novelty 的 fresh start 均有 85--106 个有效候选、几乎全通过 novelty gate，却分别只有 1/1/2 次 best 更新，且候选几乎全 regress。它被低频调用是有观测结果支持的，但无法从这批记录判断低频本身是否最优。

## 7. 轨迹状态与算子选择的关系

算子不是在相同上下文中调用。按 task 汇总后，endpoint 的平均 scalar value 为 1.23--1.37，crossover 为 1.26--1.39；backtrack 为 0.97--1.04，且 base reason 明确指向困难位置：CVRP 为 `last_regressed/endpoint_not_best/recent_plateau = 60/136/6`，TSP 为 85/163/20，OP 为 7/205/20，OBP 为 16/128/26。endpoint 全部以 `endpoint` 为 base；crossover 全部以 `crossover_base`；novelty 全部是 `fresh_start`。

这给出较强的混杂证据：backtrack 的低 reward/高 regress 率，部分来自它被定向用于非最佳 endpoint、近期 regress 或 plateau 的路线；novelty 不继承 parent；crossover 选择互补 base。因而“endpoint 的平均 reward 较高”不能读作 endpoint 的独立能力更强。轨迹选择已经先把不同 quality/potential/diversity 和不同困难度的状态分给了算子，符合当前科学认识中“轨迹选择承担主要探索与利用”的结构，但尚不能量化两层各自的因果贡献。

该结构也解释了为何算子均值跨 task 不稳定：CVRP endpoint/crossover 的选择率为 61.2%/22.1%，TSP 为 55.7%/22.6%，OP 为 57.6%/17.8%，OBP 则为 51.9%/14.6% 且 backtrack 22.4%。同一算子面临的 parent、base reason、任务尺度和阶段都不同，单个全局 mean reward 混合了这些条件。

## 8. plateau 与搜索停滞分析

固定 `positive_threshold=1e-6` 只是在父/anchor 参照下标记单步 outcome，不等价于轨迹或全局停滞。task 内 raw delta 的尺度已经相差很大：TSP endpoint median -0.005808、CVRP -0.01629、OP -0.02625、OBP -0.2、Knapsack 0；novelty 的 median 还分别为 -1.477、-5.113、-2.944、-50.2、-14.42。故绝对 delta 不能跨 task 比较，也不应把同一 zero/epsilon plateau 含义外推为同样的搜索难度。

直接过程反例也存在：CVRP rep1 i217 的 endpoint plateau 后，i218 同一路线继续 endpoint，得到新的全局 best -9.04834 -> -9.03879；TSP rep1 i13 plateau 后，i14 改由 backtrack，得到 -6.82397 -> -6.82021；OP rep2 i68 backtrack plateau 后，i69 backtrack 得到 12.63438 -> 13.57188；OBP rep1 i5 plateau 后，i6 crossover 得到 -2046.4 -> -2044.0。它们只说明一次 plateau 不能证明路线不可发展，也不能证明后续算子导致提升。

## 9. 典型过程案例

每个 task 给出四类案例。路径均指向该 run 的 `logs/method_events.jsonl` 与 `logs/method_state.jsonl`；`Q/P/D; V` 为当时轨迹状态，`分数`列是 UCB，`候选`为 `score/reference: outcome: active(A)或过滤(R)`。fresh-start 的 action 为 missing 是记录缺口，不作推断。

### cvrp_aco

| 类型 | run / iteration | 轨迹与算子状态 | 关键 action 与候选结果 | best 前 -> 后 | 观察 |
| --- | --- | --- | --- | --- | --- |
| 评分似乎帮助 | `.../20260720_233339_cvrp_v3_rep2`, i5 | endpoint, base=1；Q/P/D=.866/0/.243, V=1.083；分数 E=.322, C=.076, N=.039 | farthest-first cluster center；-16.351/-16.328 regress A，-13.447/-16.328 improve A | -15.4164 -> -13.4469 | 高分 endpoint 后出现 best 更新；是过程关联。 |
| 可能偏置 | `.../cvrp_v3_rep2`, i367 | crossover, base=656；Q/P/D=.976/.143/.118, V=1.511；C=.009 最高 | marginal-cost-per-demand transplant；-38.295/-8.731 regress R，reward=-1 | -8.6991 -> -8.6991 | 后期仍因相对分数最高被选，但该批失败并被过滤。 |
| 难以归因 | `.../cvrp_v3_rep1`, i447 | endpoint, base=808；Q/P/D=.920/.149/.088, V=1.202；四分均为负，E=-.014 略高 | demand-weighted proximity；-8.8129/-8.8785 improve A，另一个 regress A | -8.8388 -> -8.8129 | 同时包含较好轨迹、具体 action 和随机生成，不能归因给 endpoint score。 |
| plateau 后改进 | `.../cvrp_v3_rep1`, i217 -> i218 | i217 endpoint plateau；i218 endpoint, base=423，Q/P/D=.954/.218/.094 | i218 demand-weighted neighborhood density；-9.0388/-9.1057 improve A | -9.0483 -> -9.0388 | 一步 plateau 后相邻一步仍刷新 best。 |

### knapsack_construct

| 类型 | run / iteration | 轨迹与算子状态 | 关键 action 与候选结果 | best 前 -> 后 | 观察 |
| --- | --- | --- | --- | --- | --- |
| 评分似乎帮助 | `.../20260720_233339_kp_v3_rep2`, i5 | novelty fresh-start；Q/P/D=.963/0/.145, V=1.117；N=.642 最高 | action=missing；545.7/555.7 regress A，555.8/555.7 improve A | 555.6875 -> 555.8438 | 高分 novelty 后有 best，但仅一个初期样本。 |
| 可能偏置 | `.../kp_v3_rep2`, i4 | crossover, base=10；Q/P/D=.977/-.014/.181；C=.631 高于 E=.438、N=.607 | action 日志为 crossover transplant；两候选 521.625/545.656 regress A，reward=-.0557 | 545.6563 -> 545.6563 | 仅初期就出现高分失败，样本不足以称长期偏置。 |
| 难以归因 | `.../kp_v3_rep1`, i22 | crossover, base=41；Q/P/D=.996/.202/.265, V=1.364；C=.279 略高 | recursive-greedy-potential transplant；557.594/557.344 improve A | 557.3438 -> 557.5938 | 由 parent、动作和 LLM 共同决定。 |
| plateau 后改进 | `.../kp_v3_rep1`, i5 -> i9 | i5 plateau；i9 novelty fresh-start，Q/P/D=.995/.253/.444 | action=missing；555.938/555.688 improve A | 555.6875 -> 555.9375 | 中间四次 attempt 后出现 best；不能称 plateau 结束。 |

### online_bin_packing

| 类型 | run / iteration | 轨迹与算子状态 | 关键 action 与候选结果 | best 前 -> 后 | 观察 |
| --- | --- | --- | --- | --- | --- |
| 评分似乎帮助 | `.../20260720_233339_obp_v3_rep3`, i10 | endpoint, base=24；Q/P/D=1/0/.329, V=1.246；E=.357 高于 C=.322、N=.316 | residual-capacity linear bonus；-4123/-2087 regress A，-2086/-2087 improve A | -2086.6 -> -2086.4 | 最优 UCB 后有小 best 更新，但 batch reward 仍为负。 |
| 可能偏置 | `.../obp_v3_rep2`, i234 | endpoint, base=442；Q/P/D=.902/.394/.110, V=1.121；E=.012 最高 | dynamic best-fit penalty；-2092/-2047 regress A，-2053/-2047 regress R | -2038.8 -> -2038.8 | 相对最高分并未阻止失败；不能断言是 endpoint 的因果缺陷。 |
| 难以归因 | `.../obp_v3_rep2`, i397 | endpoint, base=766；Q/P/D=1/.101/.210；E=-.046 略高 | linear inverse-distance score；-2035/-2036 improve A，另一候选 regress A | -2036.2 -> -2035.2 | score 差距小，具体 action/父程序解释同样成立。 |
| plateau 后改进 | `.../obp_v3_rep1`, i5 -> i6 | i5 plateau；i6 crossover，Q/P/D=1/.260/.212，C=.505 最高 | fragmentation-aware harmonic score；-2044/-2047 improve A | -2046.4 -> -2044.0 | plateau 后 crossover 路线仍产生 best。 |

### orienteering_construct

| 类型 | run / iteration | 轨迹与算子状态 | 关键 action 与候选结果 | best 前 -> 后 | 观察 |
| --- | --- | --- | --- | --- | --- |
| 评分似乎帮助 | `.../20260720_233339_op_v3_rep1`, i9 | backtrack, base=23；Q/P/D=.904/-.031/.151, V=.958；B=.600 高于 E=.306 | budget-feasible cluster value；8.442/14.09 regress A，16.038/14.09 improve A | 14.3831 -> 16.0381 | 较大 UCB margin 与 best 同时出现，仍只是关联。 |
| 可能偏置 | `.../op_v3_rep3`, i206 | crossover, base=399；Q/P/D=.944/.329/.131，V=1.472；C=.011 最高 | net-value cluster score；11.41/16.01 regress A，16.01/16.01 plateau R | 16.2456 -> 16.2456 | 高分调用落在非改善 batch；说明 score 不能稳定预测。 |
| 难以归因 | `.../op_v3_rep1`, i361 | endpoint, base=654；Q/P/D=.885/.146/.099；E=-.024 略高 | return-path safety margin；16.31/16.26 improve A | 16.3094 -> 16.3100 | 极小更新可能更受 action、评估尺度与随机性影响。 |
| plateau 后改进 | `.../op_v3_rep2`, i68 -> i69 | i68 backtrack plateau；i69 同算子，base=10；Q/P/D=.978/.240/.150 | dynamic detour penalty；12.634/12.433 improve A，13.572/12.433 improve A | 12.6344 -> 13.5719 | 单步 plateau 不等于 backtrack 路线无效。 |

### tsp_construct

| 类型 | run / iteration | 轨迹与算子状态 | 关键 action 与候选结果 | best 前 -> 后 | 观察 |
| --- | --- | --- | --- | --- | --- |
| 评分似乎帮助 | `.../20260720_233339_tspc_v3_rep1`, i4 | endpoint, base=7；Q/P/D=.924/-.019/.153, V=1.069；E=.583 略高于 B=.548 | conditional NN/destination strategy；-8.073/-7.412 regress A，-6.989/-7.412 improve A | -7.2505 -> -6.9888 | 评分和 best 同步出现，但两个 action 结果相反。 |
| 可能偏置 | `.../tspc_v3_rep1`, i399 | backtrack, base=241；Q/P/D=.571/.189/.427, V=1.068；B=.001 高于其余负分 | straggler-awareness branch；-6.767/-6.429 regress A，-7.261/-6.429 regress A | -6.0924 -> -6.0924 | 高分选择进入困难 base 后双 regress，显见上下文混杂。 |
| 难以归因 | `.../tspc_v3_rep2`, i482 | endpoint, base=889；Q/P/D=1/.287/.176；E=-.101 略高 | crossing-elimination 近似 2-opt；-5.9723/-5.9799 improve A | -5.9799 -> -5.9723 | 最终 best 是高质量 endpoint、具体 action 和采样共同结果。 |
| plateau 后改进 | `.../tspc_v3_rep1`, i13 -> i14 | i13 endpoint plateau；i14 backtrack, base=14；Q/P/D=1/.223/.160 | continuous NN/destination interpolation；-6.8202/-6.8240 improve A | -6.8240 -> -6.8202 | 路线可在 plateau 后经回退分叉继续改善。 |

## 10. 当前能够支持的结论

**已有过程证据支持。**

1. 平均收益 UCB 确实实质改变了算子选择分布：从初期较均衡探索收缩为 endpoint 主导，novelty 在后期近乎消失。
2. 当前分布不支持“所有 task 存在同一最优算子”。选择率、mean reward、best 更新贡献在 task 与 run 间不同；例如 OP 的 backtrack 贡献 13 次 best，TSP 仅 9 次，CVRP 仅 6 次。
3. 算子收益高度依赖选择上下文。backtrack 被系统性派往 `endpoint_not_best`、`last_regressed`、`recent_plateau`；因此低均值不能被直接解释为其内在能力低。这与把主要探索/利用放在轨迹选择层的科学认识一致。
4. 固定单步 plateau 不能表示轨迹或全局停滞。五个 task 都有 plateau 后的有效改善个案，且 task 内 delta 尺度差异很大。

**有初步迹象，但样本不足。**

1. OP、TSP、OBP 中，高 UCB 三分位的后续 batch reward 更高；CVRP 没有该关系，Knapsack 过短且相反。因此没有跨 task 稳定性。
2. 后期 endpoint 偏好可能延续早期收益形成的归纳偏置，特别是 OP 的 94.1%。但没有重放/对照，不能断言它压制了本可产生更好结果的替代路线。

**与原假设不一致。**

“仅根据全局平均 operator reward 就能稳定识别更有效算子”的强版本不被当前数据支持：同一算子在不同 task、阶段和 base reason 下表现改变，fresh-start novelty 几乎总 regress，backtrack 的难上下文又使均值混杂。把它作为强主机制会与轨迹为主要搜索单位的定位不一致。

## 11. 仍无法回答的问题

1. 平均收益调度是否优于简单轮换、均匀采样或固定概率：没有 A/B/C/D 对照，当前不能回答，更不能声称替代调度一定更好。
2. 轨迹选择与算子选择各自的独立贡献：没有固定 trajectory/base/action 分布的反事实记录。
3. 算子 UCB 是否在跨 task 泛化：各 rep 使用的模型端点并不完全相同，且多数 run 尚未结束。
4. visit count、完整 operator decision context、fresh-start 的 action/edge delta 及 child 到样本程序的稳定 node-id 映射缺失，限制了逐决策因果诊断和轨迹访问饱和度分析。
5. 当前记录没有针对连续 plateau 长度、后续改善概率、任务内相对尺度的专门字段，不能拟合可靠的“路线停滞”判定。

## 12. 对下一版机制和对照实验的建议

1. **弱化而非作为主机制保留 operator reward 调度。** 保留算子作为深化、回退、组合、独立探索的动作语义；让轨迹 value/访问不足/多样性继续承担主要探索与利用。若保留 UCB，应把它定位为轻量 tie-break 或有下限的探索配额，而不是跨上下文的能力估计。
2. **执行固定预算、固定任务评估、固定模型/seed 的四臂对照。** 至少比较：均匀轮换、均匀随机、固定语义概率、当前 UCB。每 task 多 rep，报告最终训练与测试、每阶段样本效率、active diversity、有效率、best 更新和完整过程分布；不要只比较一个最终 best。
3. **做条件化消融。** 固定同一 target trajectory/base reason 分层比较 operator；至少分 endpoint、last_regressed、endpoint_not_best、recent_plateau、fresh-start，并报告 Q/P/D 与访问次数分布。这样才能区分算子效果与调用条件。
4. **把 plateau 改为相对、多步信号。** 单步仍可保留 outcome 标签；路线/全局停滞应使用 task 内滚动分位或 active-pool 尺度、连续窗口、后续改善概率和新颖性共同判断，而非固定绝对 delta。
5. **补齐审计日志。** 每个 selection 写入 target 的 visit count、Q/P/D 的命名字段、完整 portfolio attempts/mean/UCB 组成项、base/parent 评分、fresh-start 的生成意图与 anchor delta、child node-id 到 sample program 的稳定映射，以及连续 plateau/后续改善标记。这样下一次可以验证条件关联，但仍需对照实验回答因果问题。

---

## 附录 A：第二位分析者的交叉验证补充

> 以下由另一位分析者用独立脚本（`/tmp/traceaad_audit/analyze.py` + `case_picker.py`，统计 4,959 次 `operator_selection`，快照 16:11–16:17）得出。口径与主文一致；数字因快照时刻略早于主文（主文 4,975 次，16:18）而略少，方向完全一致。仅补充主文未量化的几点，**不改动主文 1–12 节的结论**。

### A.1 各算子"每次被选刷新 best"的命中率趋同（补充主文 §6）

主文 §6 用 UCB 三分位检验"高分是否关联更好结果"。换一个口径——直接看**每被选一次刷新 run 内 best 的概率 `best_refresh / selected_times`**（基于 4,959 次选择）：

| 算子 | selected | best 刷新 | best/selected |
| --- | ---: | ---: | ---: |
| endpoint_refine | 2822 | 110 | **0.039** |
| backtrack_branch | 884 | 31 | **0.035** |
| mechanism_crossover | 1000 | 32 | **0.032** |
| novelty_jump | 253 | 8 | **0.031** |

四算子命中率落在 0.031–0.039、差异不显著。endpoint 贡献最多 best 主要是"被选最多 × 普通命中率"的结果，而非它每次更易命中 best。这是"评分未在 best 产出层聚焦"的更直接反例，与主文 §6 同向；**同样不能推出"评分无用"**（见主文 §12 的对照建议）。

### A.2 被选轨迹 Q 随阶段下降：reward 后期变负的上下文证据（补充主文 §7）

主文 §7 指出算子调用上下文不同、reward 不可直接比较。补充各算子被选时轨迹终点质量 Q 随阶段的变化：

| 算子 | early Q | mid Q | late Q |
| --- | --- | --- | --- |
| endpoint_refine | 0.940 | 0.849 | 0.765 |
| mechanism_crossover | 0.951 | 0.858 | 0.764 |
| novelty_jump | 0.968 | 0.838 | 0.774 |
| backtrack_branch | 0.937 | 0.777 | **0.453** |

后期被选轨迹质量全面下降，backtrack 在 late 的 Q 暴跌至 0.453。因此 §5 中 reward 随阶段变负（endpoint −0.070→−0.136）**至少部分来自后期可用好轨迹被耗尽（上下文变差），而非算子能力下降**——强化主文 §7"reward 差异不能因果归因给算子"的判断（核心认识 3/4）。

### A.3 plateau 后短期改善率的跨 task 量化（补充主文 §8）

主文 §8 给出 plateau 后改善的个案反例。补充每个 plateau 之后 5 步内出现 improve 的比例（`child_accepted` 序列）：

| task | plateau% | plateau 后 5 步内 improve% |
| --- | --- | --- |
| cvrp_aco | 13.9% | **67.6%** |
| online_bin_packing | 25.3% | 59.2% |
| orienteering_construct | 19.3% | 54.1% |
| knapsack_construct ⚠️ | 31.0% | 47.7% |
| tsp_construct | 20.3% | 47.1% |

固定 ±1e-6 阈值判出的 plateau 率跨 task 差近一倍（cvrp 13.9% vs obp/kp 25–31%），且 47–68% 的 plateau 在 5 步内被 improve 跟随。这量化支撑主文 §8："单步 plateau 不代表路线/全局停滞"，且固定阈值的停滞含义跨 task 不可比（核心认识 5/6）。

### A.4 OBP 上 novelty 的过度探索特例（补充主文 §6）

主文 §6 指出 novelty 几乎总 regress、被低频调用。一个反方向特例：在 obp 上 novelty 被选 **46 次**（其余 task 仅 6–18 次），因 obp 归一化使 novelty 的 batch reward 常接近 0、未被快速压低；但 46 次仅 **1 次刷新 best**（命中率 2.2%）。即 reward 估计在 obp 上系统性高估了 novelty 的价值，使 12.6% 的选择份额只换来 1 次 best——与其它 task 上 novelty 被"抑制"相反的另一种偏置形态。归因仍需谨慎（obp 整数适应度＋活跃池 scale 的尺度效应待查，见主文 §11）。

### A.5 交叉验证的一致点

独立复算确认主文核心数字与结论：选择集中于 endpoint（~57%）、reward 全为负且排序 endpoint > crossover ≈ backtrack > novelty、探索算子 best 贡献集中于 early、5 个完成 run 中 3 个最终 best 落在 late。两位分析者独立得出一致结论，增强以下三点的可信度：① 算子评分未在 best 产出上体现明确聚焦；② 主要探索/利用应放在轨迹选择层；③ 无对照实验时不可声称替代调度更优。

*本附录的分析脚本为一次性产物，存放于 `/tmp/traceaad_audit/`，未写入仓库；追加过程未修改主文 1–12 节，也未修改任何代码、配置、实验结果或既有文档。*
