# TraceAAD V9.5 终局复盘

> 分析对象：正式批次 `20260811_171029`。分析日期：2026-08-12。
>
> **2026-08-12 记录：** 本文原含预算计数口径的"纠正"与公平性论述，该议题已裁定关闭——
> 固定 1000 次评价即为统一公平口径，相关论述已删除，不再讨论。
>
> Allocation / Evidence / Generation / duplicate / cache 只用于 V9.5 内部机制诊断。所有选择行为读取 `artifacts/decisions.jsonl` 当时的
> `(q,n,S)` snapshot，不用最终 forest 反推。搜索分数越大越好；held-out 保留任务原生方向。
>
> 本文是对联合机制和已完成运行的终局层复盘，不是四任务完整三重复的正式性能结论；
> TSP/CVRP 的 replacement repeat 完成前，二者的均值、方差和排名均为暂定。

## 0. 数据完整性（先读）

| Task | r1 | r2 | r3 | held-out |
| --- | --- | --- | --- | --- |
| TSP | **623 responses，configuration_failure** | 1000 responses finished | 1000 responses finished | 仅 r2/r3 |
| CVRP | 1000 responses finished | **580 responses，configuration_failure** | 1000 responses finished | 仅 r1/r3 |
| OP | 1000 responses finished | 1000 responses finished | 1000 responses finished | 3/3 |
| OBP | 1000 responses finished | 1000 responses finished | 1000 responses finished | 3/3 |

原批次失败原因已定位为 tokenizer 暂时不可用后的假 context overflow（见 worklog）。`contextfix_*`
是**新开**的补跑，不是 resume；截至本次审计，TSP 补跑在 51 responses / 50 evaluator calls、
CVRP 补跑在 55 / 55 时因 `tokenizer_retry_exhausted` 进入 `infrastructure_failure`，仍未形成替代
repeat，**不进入本复盘**。

因此：

- **V9.5 finished-run 描述**：TSP/CVRP 用 finished 2-run；OP/OBP 用 3-run。
- Incomplete run 的过程统计仍报告，但单独标注预算。
- 凡写“排名 / 相对旧版本”，仅指 finished run 的描述性比较。

---

## 1. 最终性能（最高优先级）

### 1.1 Search best（directed，越大越好；finished-run endpoint）

| Task | r1 | r2 | r3 | finished mean ± SD | vs V9.4 | vs V9.3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TSP | −5.707@623 | −6.106 | **−5.703** | **−5.905 ± 0.285** (n=2) | 更好 (+0.228) | 更好 (+0.194) |
| CVRP | −9.057 | −8.842@580 | **−8.906** | −8.982 ± 0.106 (n=2) | 近似持平 (+0.007) | 更差 (−0.224) |
| OP | 14.626 | 14.544 | **14.678** | 14.616 ± 0.068 (n=3) | 更差 (−0.201) | 更差 (−0.148) |
| OBP | **−725.25** | −731.00 | −729.25 | −728.50 ± 2.95 (n=3) | 更好 (+5.08) | 更差 (−2.33) |

同场 search 排名（finished；对照 V8.3 / V9.1-Traj / V9.2 / V9.3 / V9.4）：

| Task | V9.5 排名 | 当前最强对照 | 三 repeat 一致性 |
| --- | ---: | --- | --- |
| TSP | **1** | 自身；次为 V9.1-Traj (−5.908) | 分化大：强 r3 ≈ V9.4 最强，弱 r2 明显落后 |
| CVRP | 4 | **V9.3 (−8.758)** | finished 两路接近；incomplete r2 当时更强 |
| OP | 5 | **V9.4 (14.817)** | 三路窄（14.54–14.68），稳定但整体偏低 |
| OBP | 3 | **V9.3 (−726.17)** | 无 V9.4 那种 −744.5 崩塌；中等稳定 |

### 1.2 Held-out（原生 objective）

**TSP ↓（n=2）**

| Scale | r2 | r3 | mean ± SD | vs V9.4 | vs V9.3 | 局内大致排名 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TSP50 | 6.108 | **5.706** | **5.907 ± 0.284** | 更好 | 更好 | 1（优于 V9.1-Traj / V6 均值口径） |
| TSP100 | 8.649 | **7.897** | **8.273 ± 0.532** | 更好 | 更好 | 1 |
| TSP200 | 12.638 | **11.322** | 11.980 ± 0.931 | 更好 | 更好 | 2（次于 V9.1-Traj 11.743） |

**CVRP ↓（n=2）**

| Scale | r1 | r3 | mean ± SD | vs V9.4 | vs V9.3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CVRP50 | 9.397 | 9.467 | 9.432 ± 0.049 | 更差 | 更差 |
| CVRP100 | 15.832 | 15.824 | 15.828 ± 0.006 | 略好 | 更差 |
| CVRP200 | 28.073 | **27.859** | **27.966 ± 0.151** | 更好 | 近似持平 |

**OP ↑（n=3）**

| Scale | r1 | r2 | r3 | mean ± SD | vs V9.4 | vs V9.3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OP50 | 14.982 | 14.706 | **15.120** | 14.936 ± 0.211 | 更差 | 更差 |
| OP100 | 29.473 | 29.332 | **30.628** | 29.811 ± 0.711 | 更差 | 更差 |
| OP200 | 51.185 | 51.904 | **55.583** | 52.891 ± 2.359 | 更差 | 更差 |

**OBP ↓（n=3）**

| Scale | mean ± SD | vs V9.4 | vs V9.3 |
| --- | ---: | ---: | ---: |
| 1k/100 | 425.4 ± 12.9 | 更差 | 更差 |
| 5k/100 | 2036.7 ± 16.9 | 更好 | 更差 |
| 10k/100 | 4062.7 ± 38.5 | 更好 | 更差 |
| 1k/500 | 80.8 ± 0.0 | 略好/持平 | 持平 |
| 5k/500 | 402.67 ± 0.76 | 略好 | 持平 |
| 10k/500 | 804.87 ± 1.62 | 更好 | 略差 |

### 1.3 四任务总体（不平均原始 fitness）

| 维度 | 事实 |
| --- | --- |
| 四任务 search 相对 V9.4 | **2 胜 1 负 1 平**（TSP/OBP 胜，OP 负，CVRP 平） |
| 四任务 search 相对 V9.3 | **1 胜 3 负**（仅 TSP 胜） |
| Held-out 主档相对 V9.4 | TSP 三档更好；CVRP 混；OP 三档更差；OBP capacity=100 中大档更好、1k 更差 |
| Held-out 主档相对 V9.3 | TSP 更好；CVRP/OP/OBP 整体更差或持平 |
| Task-dependent pattern | **TSP 明显受益；OP 明显受损；CVRP≈V9.4 且弱于 V9.3；OBP 去掉 V9.4 崩塌但仍弱于 V9.3** |

**判断（仅基于 finished-run 最终质量）：** V9.5 值得作为“统一 Quality+Opportunity allocation + Evidence”的可运行对象继续研究，但**还不能**宣称相对旧版本的联合升级。当前最强信号在 TSP；OP 是最清楚的弱势任务。当前强版本参照应是 V9，而不是 V9.3/V9.4；TSP/CVRP replacement repeat 完成前，不对 V9 与 V9.5 作最终排名结论。

---

## 2. Best-so-far 曲线

截点按实际 evaluator call 累计（含初始化）。下表为 **finished repeats 的均值**；若某路未达到所列
截点，旧统计使用其最终可用值延伸展示。表格仅保留为该批次内部曲线形态记录。

### 2.1 均值曲线（directed）

| Task / t | 100 | 200 | 300 | 500 | 750 | 1000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TSP V9.5 | −6.177 | −6.067 | −6.044 | −6.044 | −5.927 | **−5.905** |
| TSP V9.3 | −6.290 | −6.137 | −6.133 | −6.117 | −6.099 | −6.098 |
| TSP V9.4 | −6.699 | −6.464 | −6.445 | −6.377 | −6.256 | −6.133 |
| CVRP V9.5 | −9.510 | −9.260 | −9.140 | −9.064 | −9.017 | −8.982 |
| CVRP V9.3 | −9.284 | −8.964 | −8.870 | −8.826 | −8.761 | **−8.758** |
| CVRP V9.4 | −9.494 | −9.156 | −9.017 | −9.012 | −8.995 | −8.989 |
| OP V9.5 | 13.991 | 14.099 | 14.136 | 14.247 | 14.597 | 14.616 |
| OP V9.3 | 14.204 | 14.630 | 14.657 | 14.710 | 14.725 | 14.764 |
| OP V9.4 | 14.221 | 14.494 | 14.712 | 14.794 | 14.817 | **14.817** |
| OBP V9.5 | −741.3 | −735.3 | −734.3 | −729.5 | −728.5 | −728.5 |
| OBP V9.3 | −733.3 | −731.9 | −726.8 | −726.2 | −726.2 | **−726.2** |
| OBP V9.4 | −744.4 | −744.4 | −744.4 | −739.9 | −733.6 | −733.6 |

### 2.2 各 run：末次刷新、两阶段增益、早熟

| Run | last refresh | 100→500 | 500→1000 | 早熟？ |
| --- | ---: | ---: | ---: | --- |
| TSP r2 | 904 | +0.068 | +0.063 | 否，后半仍动 |
| TSP r3 | 860 | +0.199 | +0.215 | 否；后半更强 |
| TSP r1@623 | 613 | +0.363 | (+0.023 to end) | 未完成 |
| CVRP r1 | 803 | +0.236 | **+0.009** | **后半近似停滞** |
| CVRP r3 | 812 | +0.656 | +0.156 | 否 |
| CVRP r2@580 | 574 | +0.500 | (+0.022) | 未完成；当时更强 |
| OP r1 | 639 | 0.000 | **+0.962** | 前半停滞、后半大跳 |
| OP r2 | 909 | +0.316 | +0.038 | 轻度后半减速 |
| OP r3 | 933 | +0.454 | +0.106 | 否 |
| OBP r1 | 564 | +6.5 | +3.0 | 否 |
| OBP r2 | 350 | +13.5 | **0** | **500 后完全停滞** |
| OBP r3 | 226 | +15.25 | **0** | **更早停滞** |

### 2.3 对阶段性现象的终局核对

| 阶段性观察 | 1000 后是否仍成立 |
| --- | --- |
| TSP 强 | **成立**（search/held-out 均强；强 run 后半继续突破） |
| OP 分化 | **收敛但仍在**：三路 search 差距缩小到 ~0.13；held-out 仍由 r3 拉高 |
| CVRP 弱 run 后半可能追赶 | **不成立于 r1**：500→1000 仅 +0.009；真正更强的是未完成的 r2 |

---

## 3. Allocation（完整预算重算）

公式不变：\(S=q+s/\sqrt{n+1}\)。

### 3.1 全预算主表

| Run | s | sels | changed | n=0/1/≥2 | gap/s med | s_crit Q25/med/Q75 | med s/s_crit | s>s_crit |
| --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: |
| TSP r1@623 | 2.487 | 606 | **95.0%** | 61/35/3% | 0.065 | .17/.35/.78* | 6.16 | 97.1% |
| TSP r2 | 0.720 | 984 | **93.4%** | 64/33/3% | 0.087 | — | 3.76 | 95.9% |
| TSP r3 | 1.384 | 983 | **94.3%** | 66/32/2% | 0.038 | — | 7.72 | 96.6% |
| CVRP r1 | 0.750 | 984 | **88.6%** | 90/9/1% | 0.124 | — | 3.99 | 94.7% |
| CVRP r2@580 | 0.730 | 563 | **85.8%** | 85/13/1% | 0.095 | — | 3.43 | 93.1% |
| CVRP r3 | 0.859 | 983 | **90.9%** | 87/12/1% | 0.120 | — | 3.95 | 95.5% |
| OP r1 | 0.088 | 984 | **2.1%** | 14/12/74% | 0 | — | 0.47 | 2.4% |
| OP r2 | 0.194 | 983 | **14.4%** | 19/17/64% | 0 | — | 0.81 | 16.6% |
| OP r3 | 0.035 | 984 | **2.5%** | 6/5/89% | 0 | — | 0.61 | 2.7% |
| OBP r1–r3 | **0** | ~984 | **0%** | 见下 | — | — | 0 | 0% |

OBP n 比：r1 5/3/92%；r2 32/20/49%；r3 23/22/55%。

### 3.2 分阶段 changed rate

| Run | 1–250 | 251–500 | 501–750 | 751–1000 |
| --- | ---: | ---: | ---: | ---: |
| TSP r2 | 85.9% | 94.8% | 98.4% | 94.0% |
| TSP r3 | 83.7% | 97.6% | 96.0% | **99.2%** |
| CVRP r1 | 78.6% | 96.4% | 92.0% | 86.8% |
| CVRP r3 | 81.5% | 92.0% | 93.2% | 96.4% |
| OP r1 | 0% | 0.4% | 6.4% | 1.6% |
| OP r2 | 7.7% | 9.6% | 7.6% | **32.4%** |
| OP r3 | 1.7% | 0.4% | 7.6% | 0.4% |
| OBP all | 0 | 0 | 0 | 0 |

回答：

1. **TSP/CVRP 后半程 optimism 仍 active**——changed 维持高位，甚至略升。
2. **OP 大体仍在 greedy regime**；只有 r2 在 751–1000 升到 32%，仍远低于 TSP/CVRP。
3. **OBP 始终 s=0**。
4. changed-rate：**TSP/CVRP 稳定高；OP 稳定低（r2 末段例外）；OBP 恒为 0**。

三种 operational regime 到终点**仍然存在**。

---

## 4. 搜索结构

正式搜索（`candidate_order>16`）clade 预算：

| Run | formal clades | Top-1 share | unique anchors | states | arts | sel depth mean/max | best depth | init→final clade | lineage len | #BT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TSP r2 | 1 | 100% | 638 | 878 | 878 | 85 / — | 139 | 1→**6** | 140 | 9 |
| TSP r3 | 6 | 98.8% | 656 | 824 | 824 | 66 / — | 109 | 5→**4** | 110 | 18 |
| CVRP r1 | 2 | 99.9% | 893 | 951 | 951 | 155 / — | 254 | 6→**4** | 255 | 33 |
| CVRP r3 | 1 | 100% | 863 | 954 | 954 | 173 / — | 294 | 1→1 | 295 | 39 |
| OP r1 | 1 | 100% | 141 | 901 | 901 | 21 / — | 36 | 1→3 | 37 | 9 |
| OP r2 | 2 | 99.9% | 196 | 900 | 899 | 34 / — | 52 | 7→4 | 53 | 12 |
| OP r3 | 1 | 100% | 63 | 979 | 979 | 23 / — | 30 | 6→6 | 31 | 16 |
| OBP r1 | 7 | 97.4% | 54 | 936 | 934 | 11 / — | 16 | 1→2 | 17 | 10 |
| OBP r2 | 8 | 74.5% | 320 | 936 | 936 | 21 / — | 21 | 1→5 | 22 | 1 |
| OBP r3 | 6 | 97.4% | 233 | 912 | 911 | 30 / — | 19 | 1→1 | 20 | 3 |

拓扑事实归纳：

- 多数 run 的正式预算在生成拓扑上集中于单个 root clade（Top-1 ≥ 98%）。
- 部分 run 先改变主导 root，之后主要沿该生成分支继续修改。
- TSP 强 run（r3）具有更多 breakthrough 和更长的连续修改历史；OP 强 run（r3）anchor 更少、重访更多。
- OBP r2 是唯一 Top-1 明显低于 90% 的 run，同时几乎无后期 breakthrough。

Clade 只表示 parent-child provenance，不表示算法思想、semantic basin 或自然搜索区域。以上统计不能
证明同一 clade 的特殊价值，也不能把 root switching 当作语义探索。此前的 “within-clade success
mechanism” 表述作废。

---

## 5. Evidence composition（实际进入 prompt）

| Run | direct mean/med | formation mean/med | form_missing | squeezed | 主导模式 |
| --- | --- | --- | --- | --- | --- |
| TSP r2/r3 | ~0.39 / 0 | ~7.4 / 8 | ~1% | ~0 | **formation-heavy，全程稳定** |
| CVRP r1/r3 | ~0.11–0.15 / 0 | ~7.7 / 8 | ~1% | 0 | **更极端 formation-heavy** |
| OP r1 | 3.95 / 4 | 3.87 / 4 | 22% | 21% | 混合 |
| OP r2 | 2.96 / 2 | 4.90 / 5 | 10% | 9% | 偏 formation |
| OP r3 | **6.21 / 8** | **1.63 / 0** | **65%** | **64%** | **direct-heavy** |
| OBP r1 | 6.86 / 8 | 0.80 / 0 | 80% | 78% | direct-heavy |
| OBP r2/r3 | ~2.3 / 1–2 | ~5.2 / 6 | 7–9% | 6–7% | 偏 formation |

分阶段：TSP/CVRP **没有**从 formation 转向 direct；反而后半 formation 更满。OP r3 / OBP r1 的 direct-heavy 随搜索加深更明显。模式与 allocation freshness 同向：n≈0 选择多 → formation-heavy；反复打磨成熟锚点 → direct-heavy。

**不作好坏预设。** exact dedup、diff available、idea missing、invalid exposure 见原始 JSON；健康项未见异常堵塞搜索。

---

## 6. Allocation 之后的 generation productivity（观察性）

| Run | 类型 | samples | valid | improve rate | Δq med | regress | BT | BT/sample |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| TSP r3 | optimism | 927 | 757 | **33.0%** | −0.004 | 52% | 15 | 0.016 |
| TSP r3 | greedy | 56 | 51 | 5.9% | −0.16 | 82% | 1 | 0.018 |
| TSP r2 | optimism | 919 | 805 | 23.4% | −0.013 | 58% | 6 | 0.007 |
| CVRP r3 | optimism | 894 | 854 | **42.2%** | −0.012 | 55% | 30 | 0.034 |
| CVRP r1 | optimism | 872 | 852 | ~similar | — | — | 21 | 0.024 |
| OP r3 | greedy | 959 | 938 | 1.6% | −0.59 | 95% | 15 | 0.016 |
| OP r3 | optimism | 25 | 25 | 4.0% | −0.55 | 96% | 1 | 0.040 |
| OP r2 | optimism | 142 | 133 | 9.0% | −0.26 | 75% | 2 | 0.014 |

Optimism n=0 vs n≥1：TSP/CVRP 上 n=0 仍是主要 optimism 样本来源，一步 improve 更高；这是观察关联，**不是** optimism 因果收益。

---

## 7. Immediate vs future lineage；最终 best 的非单调性

Optimism-induced child 的延迟突破（finished 主 run）：

| Run | imm BT | delayed BT children | delayed rate | regress children | regress→desc BT |
| --- | ---: | ---: | ---: | ---: | ---: |
| TSP r3 | 15 | 高（深树） | ~0.2+ | 多 | 非零 |
| CVRP r3 | 30 | 260 | **0.30** | 466 | **0.26** |
| OP r3 | 1 | 0 | 0 | 24 | 0 |

**最终 best lineage 的 transition 构成：**

| Run | improve | plateau | regress | 结论 |
| --- | ---: | ---: | ---: | --- |
| TSP r3 | 47 | 19 | **43** | **远非单调** |
| TSP r2 | 55 | 33 | **51** | 远非单调 |
| CVRP r3 | 150 | 9 | **135** | 远非单调 |
| OP r3 | 15 | 15 | **0** | 本 run 近乎单调+平台 |
| OP r2 | 13 | 35 | 4 | 平台为主 |
| OBP r1 | 10 | 6 | 0 | 短链路、少退步 |

这对 TraceAAD 动机很关键：**TSP/CVRP 的最终优秀算法 lineage 大量经过 regress**；OP/OBP 则更多平台或短改进链。保留 regression state 与“历史作为决策信息”在 TSP/CVRP 上仍有结构对应，但仍不是因果证明。

---

## 8. 健康诊断（不参与版本公平比较）

| Run | invalid | dup | ancestral | cache | status |
| --- | ---: | ---: | ---: | ---: | --- |
| TSP r2 | 6.4% | 0.1% | 0.2% | 5.8% | finished |
| TSP r3 | 14.3% | 0.1% | 0.3% | 3.3% | finished |
| CVRP r1/r3 | ~0.8–0.9% | ~0 | 0.7–2.5% | 3.7–4.1% | finished |
| OP all | 0.3–6.2% | ~0.1–0.2% | ≤0.2% | 1.8–5.4% | finished |
| OBP all | ≤1% | 0.2–2.9% | ≤0.4% | 5.7–7.9% | finished |
| TSP r1 / CVRP r2 | — | — | — | — | **configuration_failure** |

未见后半程 no-op/duplicate 爆炸导致“搜索饱和”的主导证据。OP/OBP 后半停滞更像 **best-so-far 不再刷新**，而不是提案管道堵塞。真正的硬失败是两路 context 假溢出。

---

## 9. Case study：TSP/OP 强弱 run

### 9.1 TSP 最强 r3（search −5.703；BT=18；末次 860）

- 几乎全程锁在 root clade **4**；深度推进到 109。
- 多数后期 breakthrough 是 **optimism-induced + n=0 + formation=8**。
- 关键窗口：656–662 连续刷新；最终 860。
- Lineage：47 improve / 43 regress / 19 plateau。

### 9.2 TSP 最弱 finished r2（search −6.106；BT=9；末次 904）

- 正式预算 **100%** 在 clade 6（从 init-best clade 1 切走后锁死）。
- Optimism 同样活跃（changed 93%），但一步 improve 与 delayed value 低于 r3。
- 差异更接近 **lineage 起点 + generation realization**，而不是 allocation regime（两路同属 active optimism）。

### 9.3 OP 最强 r3（14.678；BT=16；末次 933）

- changed 仅 2.5%；anchors 仅 63；反复打磨同一成熟锚点。
- Breakthrough 几乎全是 **greedy**；末次 n=39、direct=8。
- Lineage：**0 regress**，15 improve + 15 plateau。

### 9.4 OP 最弱 r2（14.544；BT=12）

- changed 14.4%（三路最高），仍远低于 TSP。
- 更深（best depth 52）但最终更差；末段仍有小幅突破。
- Lineage 平台更多（35 plateau）。

强弱对照：**同任务内 allocation regime 相近时，差距主要由实现路径与提案兑现率拉开；跨任务则是 regime 本身不同。**

---

## 10. 事实总结（不提新机制）

1. **相对强参照 V9 强不强？**  
   Finished-run held-out：V9.5 的 TSP 三档更好；CVRP 仅 200 略好；OP 三档更差，OBP 也多数弱于
   V9。由于 TSP/CVRP 仅两路完成，这只是当前方向，不是最终排名。V9.5 **不是**全面升级，
   但 TSP 是明确正信号。

2. **三种 allocation regime 是否仍在？**  
   **是。** TSP/CVRP active optimism；OP near-greedy；OBP pure-q（s=0）。

3. **TSP 优秀程序是否具有较长的连续修改历史？**  
   **是。** 其生成拓扑深度超过 100，且预算高度集中；但这只说明 provenance，不能证明长历史或
   same-clade 是成功原因。

4. **OP 强弱是否仍 route-sensitive？**  
   Search 差距缩小，但 held-out 仍明显（尤其 OP200：55.6 vs ~51–52）。强 run 是成熟锚点 greedy refinement；弱 run 更深却更差。

5. **CVRP 弱 r1 后半是否追上？**  
   **否**（500→1000 仅 +0.009）。未完成的 r2 在 580 时反而更好——补跑结果出来前不能下最终 CVRP 结论。

6. **OBP 是否始终 s=0？表现？**  
   **是。** 在 pure-q + tie-break 下：无 V9.4 崩塌，search/held-out 中等，整体仍弱于 V9.3；两路 500 后 search 完全停滞。

7. **强弱差异更接近什么？**  
   - 跨任务：allocation regime。  
   - 同任务（TSP/OP）：**generation productivity + lineage 起点/路径**；allocation 本身不足以区分强弱。  
   - 不能把差异单归因于 Evidence composition。

8. **阶段性现象：消失 vs 稳定**  
   - 消失/削弱：OP search 极端分化；“CVRP 后半追赶”预期。  
   - 稳定：三 regime；TSP deep continuation；OBP s=0；TSP/CVRP formation-heavy；最终优秀 lineage 在 TSP/CVRP 非单调。

9. **只能作 observation、不能作 causal claim 的结果**  
   - optimism-induced 的一步/延迟突破率  
   - formation-heavy vs direct-heavy 与最终质量  
   - regress child 的后代突破  
   - clade 数 / Top-1 share 与好坏  
   - 单次强 run（TSP r3 / OP r3）推广为机制胜利  

---

## 11. 分层机制判断

### 11.1 State / Search Forest：工程上通过

正式完成运行中没有出现 duplicate explosion、ancestral cycle、cache 异常或后半程
proposal 管道崩坏，100--300 层的 lineage 也能稳定形成。现有证据不支持继续修改这一层。

### 11.2 Evidence：工程上通过，科学价值尚未识别

Formation/direct correction、actual diff、dedup 与上下文构造均正常工作；两路中断来自旧的
tokenizer 失败回退，不是 EvidenceBuilder 自身膨胀。当前联合结果只能证明
`Current Code + Evidence` 的 V9.5 能够运行，不能证明 Evidence 相对 `Current Code Only`
改善了生成。TSP 的正信号也不能单独归因于历史。

### 11.3 Generation：机会能否兑现的主要随机来源

TSP r2/r3 的 allocation regime 与 Evidence 组成相近，但最终质量和
optimism-induced improvement rate 差异明显；OP 内部也不能仅靠 allocation 差异解释强弱。
因此更符合当前证据的表述是：搜索机制决定哪些程序获得生成机会，LLM 决定这些机会能否
转化成有效 modification 与突破。27B 模型的生成方差是联合系统的一部分，目前没有证据支持
用 critic、planner 或 operator portfolio 覆盖它。

### 11.4 Allocation：确实改变行为，尚未证明净收益

TSP/CVRP 的 optimism changed-rate 长期处于高位，OP 大体接近 greedy，OBP 因 `s=0`
成为 pure-`q`/tie-break 的自然对照。这证明公式产生了三种不同的运行形态，但四任务最终质量
与 changed-rate 没有单调关系：active optimism 在 TSP 对应强结果，在 CVRP 没有形成升级；
near-greedy OP 结果偏弱，`s=0` 的 OBP 居中。

因此 changed-rate 和非零 `s` 都只是干预强度，不是优化目标。Allocation 的目标仍是
best-at-budget；不能因为某任务的 optimism 太活跃或不活跃，就据此调整 `s`。

---

## 12. 两个研究含义

### 12.1 最终优秀 lineage 可以高度非单调

TSP/CVRP 的最终 best lineage 包含大量 regress transition，说明贪心地要求每一步 fitness
改善会删掉实际可达的成功路径。这个结果支持保留真实可执行的 regression state，让后续搜索
仍有机会从它继续；它不支持给 regression 自动加信用。一个节点事后位于成功谱系上，不等于
在当时或现在具有可预知的前瞻预算价值。

### 12.2 历史价值可能依赖任务的算法改进几何

TSP/CVRP 呈现 100--300 层的深链路以及 improve/regress 交错，更像连续积累、反复重构的
非单调改进；OP/OBP 的最终链路较短，并更多由 improvement/plateau 构成。这提示一个新的
待验证假设：需要长期结构积累的任务可能更依赖 formation history，围绕强 heuristic 反复试验
局部变体的任务可能更依赖 exact-state direct trials。

这只是从 V9.5 联合运行得到的相关性观察。它既不能证明 Evidence 对任何任务有效，也不能据此
为不同任务预设不同 Evidence 配额。

---

## 13. 下一阶段决策

1. 等待 TSP r1 与 CVRP r2 replacement repeat 完成。
2. 以 V9 而非 V9.3/V9.4 为强版本参照重算 search 与 held-out。
3. 优先做固定真实 anchor 的生成接口配对：`Current Code Only`、V9 式 concise history、V9.5
   式 correction diff evidence；先识别信息是否可用，再运行整棵搜索。
4. 再做 Allocation 消融：固定 Evidence 与 Generation，对比 pure `q(a)`、V9 选择器与
   `q(a)+s/sqrt(n(a)+1)`。
5. 在这两个基础假设被识别前，不加入 `0.5s/2s`、formation quota、descendant evidence、
   adaptive `s`、credit、critic 或 operator。

**当前总判断：** V9.5 不是 V9 的全面升级，也不应被归类为失败。它已经成为一个
机制清楚、工程稳定、能产生显著任务依赖搜索动力学的研究基座。TSP 是强正信号，OP 是稳定
负信号，CVRP/OBP 居中；其主要科学价值是把“历史是否改善生成”和“Quality + Opportunity
是否改善有限预算搜索”暴露成了两个可以独立检验的问题。

---

## 复核入口

```text
experiments/<task>/traceaad_v9_5/v9_5_20260811_171029_<task>_rep{1,2,3}/
experiments/<task>/traceaad_v9_5/eval_best_20260812_v95/results.json
```

V9/V9.5 生成接口与预算审计脚本：`experiments/analysis/analyze_v9_v95_generation_interface.py`；
完整聚合见 `traceaad_v9_v95_generation_interface/summary.json`。

19:10 截点的前置阶段性分析已由本文在完整预算上重算，不再单独保留。
