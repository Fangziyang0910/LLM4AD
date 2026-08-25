# TraceAAD V9.17 自适应调度单因素消融

## 研究问题

检验一次运行中观察到的正 block gain，是否足以支持继续发展当前假设，而不是立即重新 Discovery。

对照使用相同的 V9.17 搜索状态、生成机制与竞争机制，只改变全体 active sweep 结束后的调度规则：

| 实验臂 | sweep 结束后的规则 |
| --- | --- |
| Adaptive | `g>0` 的假设进入下一 sweep，直到一轮没有成功假设 |
| FixedCycle | 一轮全体 sweep 后立即进入 Discovery |

FixedCycle 的每个周期为：全部 active hypotheses 各执行一个三步 Refine block，随后 Discover；有效新 hypothesis 完成三步 maturation 后按前沿质量参加前八竞争。

两臂共同使用八个初始 hypothesis、三步 block、冻结的一步尺度、假设内父节点选择、parent-path prompt、最高质量 active frontier 的 Explore 来源、两次 bounded repair 和 1000 个 primary slots。

## 配对

每个 `task × repeat` 形成一个配对 block，共 5 个任务、3 个重复和 15 对运行。同一对运行使用相同 seed，并共享首次 Development 决策前的完整 checkpoint：

- 八个有效 roots 及其程序；
- 每个 root 的三步初始化 maturation；
- 冻结的 `s_R`；
- 初始 active set、前沿节点和竞争次序；
- 已消耗的 primary slots 与 repair 调用事实。

共享 checkpoint 必须处于 `cycle=1`、`sweep=1` 且尚未选择第一个 Development 父节点的状态。FixedCycle 从该 checkpoint 继续计数到 1000 primary slots。

## 预先记录的过程量

- Refine、Explore 的实际 primary-slot 数量和比例；
- 相邻 Discovery 的 slot 距离；
- 每个周期实际执行的 sweep 数和连续成功长度；
- 正 block gain 后下一 block 再次产生正增益的比例；
- Adaptive 在第二及后续 sweep 中追加的 blocks 数量；
- 追加 blocks 推进 hypothesis frontier 和 global frontier 的数量；
- Refine slots 在 active hypotheses 之间的最大份额和集中度。

## 结果量

- 100、250、500、750、1000 slots 的 search best；
- 1000-slot 最终 search best；
- 各任务同规模 held-out；
- 各任务跨规模 held-out。

所有比较先保留 `task × repeat` 配对，再报告三重复的实际差值和方向。

## 结果

覆盖范围：5 任务 × 3 重复共 15 对全部完成并完成 held-out。VRPTW held-out 为 50/100/200 三规模（16 实例 × seed 2025）。配对差为 Adaptive − FixedCycle，TSP/CVRP/VRPTW/OBP 越低越好（正差 = FixedCycle 更优），OP 越高越好（正差 = Adaptive 更优）。

### Online Bin Packing

训练口径两臂接近不可分：1000 slots 的搜索 best 差 +1.0、−1.5、−0.75，方向不一致；250 slots 三重复同向，Adaptive 少 3.0、3.0、10.25。held-out 上差距拉开且按容量分化：容量 100 的规模 Adaptive 占优，1k_100 三重复同向少 6.6、20.2、15.6；容量 500 的规模 FixedCycle 占优，5k_500 三重复同向少 1.2、0.8、1.0，跨规模 10k_500 三重复同向少 1.0、1.4、1.6。

搜索 best-at-budget（训练 bins，越低越好）：

| slots | rep1 A | rep1 F | rep2 A | rep2 F | rep3 A | rep3 F |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 730.50 | 731.00 | 731.25 | 731.00 | 730.00 | 744.50 |
| 250 | 727.00 | 730.00 | 727.75 | 730.75 | 730.00 | 740.25 |
| 500 | 726.50 | 726.25 | 727.50 | 730.25 | 728.50 | 730.00 |
| 750 | 726.00 | 725.00 | 726.50 | 727.00 | 727.50 | 728.75 |
| 1000 | 726.00 | 725.00 | 725.50 | 727.00 | 726.00 | 726.75 |

held-out（bins 均值，越低越好；前四行为同规模，后两行为跨规模）：

| 规模 | rep1 A | rep1 F | rep2 A | rep2 F | rep3 A | rep3 F | 配对差 | 方向 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1k_100 | 408.6 | 415.2 | 409.0 | 429.2 | 414.2 | 429.8 | −6.6 / −20.2 / −15.6 | 三重复同向，Adaptive 更少 |
| 5k_100 | 2023.4 | 2018.6 | 2019.8 | 2037.6 | 2025.4 | 2038.8 | +4.8 / −17.8 / −13.4 | 不一致 |
| 1k_500 | 81.0 | 81.0 | 80.6 | 80.8 | 81.0 | 80.8 | 0.0 / −0.2 / +0.2 | 不一致，幅度 ≤0.2 |
| 5k_500 | 403.0 | 401.8 | 402.6 | 401.8 | 402.8 | 401.8 | +1.2 / +0.8 / +1.0 | 三重复同向，FixedCycle 更少 |
| 10k_100 | 4039.6 | 4028.2 | 4035.6 | 4065.6 | 4042.0 | 4064.6 | +11.4 / −30.0 / −22.6 | 不一致 |
| 10k_500 | 804.2 | 803.2 | 804.4 | 803.0 | 804.6 | 803.0 | +1.0 / +1.4 / +1.6 | 三重复同向，FixedCycle 更少 |

### VRPTW（两臂 3 对完整）

held-out 三规模（均值总距离，越低越好）：vrptw100 与 vrptw200 三重复同向 Adaptive 更低（100：少 0.454、0.830、1.383；200：少 1.445、0.475、3.987），vrptw50 方向不一致（−0.091、+0.030、−0.972）。训练 best 方向不一致（rep1/2 FixedCycle 更低，rep3 Adaptive 更低）。跨规模泛化 Adaptive 占优与 TSP200 的三重复同向一致。

| 重复 | 50 A | 50 F | 100 A | 100 F | 200 A | 200 F |
| --- | --- | --- | --- | --- | --- | --- |
| rep1 | 19.183 | 19.274 | 32.246 | 32.700 | 53.705 | 55.150 |
| rep2 | 19.626 | 19.596 | 32.129 | 32.959 | 53.895 | 54.370 |
| rep3 | 18.443 | 19.415 | 31.585 | 32.968 | 52.449 | 56.436 |

对应搜索 train best：Adaptive 19.368、19.044、18.779；FixedCycle 19.162、18.953、19.697。

### TSP（两臂 3 对完整）

训练口径：250 slots 三重复同向 FixedCycle 更低（少 0.061、0.143、0.170），1000 slots 方向不一致（rep1 Adaptive 少 0.247，rep2/3 FixedCycle 少 0.100、0.290）。held-out 按规模分化：TSP200 三重复同向 Adaptive 更低（少 0.139、1.084、0.478），TSP50 与 TSP100 方向不一致。训练中期 FixedCycle 推进更快、跨规模泛化 Adaptive 占优。

搜索 best-at-budget（训练总距离，越低越好）：

| slots | rep1 A | rep1 F | rep2 A | rep2 F | rep3 A | rep3 F |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 6.255 | 6.175 | 6.262 | 6.224 | 6.262 | 6.262 |
| 250 | 6.151 | 6.090 | 6.169 | 6.026 | 6.252 | 6.082 |
| 500 | 5.985 | 6.090 | 5.946 | 5.953 | 6.171 | 6.010 |
| 750 | 5.838 | 6.078 | 5.875 | 5.775 | 6.171 | 5.974 |
| 1000 | 5.831 | 6.078 | 5.875 | 5.775 | 6.138 | 5.848 |

held-out（均值总距离，越低越好）：

| 重复 | TSP50 A | TSP50 F | TSP100 A | TSP100 F | TSP200 A | TSP200 F |
| --- | --- | --- | --- | --- | --- | --- |
| rep1 | 5.831 | 6.173 | 8.055 | 8.561 | 11.787 | 11.925 |
| rep2 | 5.835 | 5.778 | 8.126 | 8.563 | 11.536 | 12.620 |
| rep3 | 6.231 | 5.930 | 8.559 | 8.452 | 11.902 | 12.380 |

Adaptive rep1 的 TSP200 首评 1000 秒超时，3000 秒复测得 11.787 计入 n=3。

### CVRP（两臂 3 对完整）

held-out 三规模 × 三重复 9 格全部同向 Adaptive 更低，是全消融最强的任务级结果；搜索 train best 与 500 slots best-at-budget 也三重复同向 Adaptive。

| 重复 | 50 A | 50 F | 100 A | 100 F | 200 A | 200 F |
| --- | --- | --- | --- | --- | --- | --- |
| rep1 | 8.896 | 9.130 | 15.138 | 15.411 | 27.517 | 27.598 |
| rep2 | 9.232 | 9.334 | 15.641 | 15.887 | 28.543 | 28.975 |
| rep3 | 9.145 | 9.366 | 15.364 | 16.169 | 27.742 | 29.287 |

搜索 train best：Adaptive 8.578、8.850、8.747；FixedCycle 8.848、9.025、9.080。

### OP（两臂 3 对完整）

三规模方向均不一致，由 Adaptive rep2 单路偏弱驱动（train 14.534 为全部六路最低，三档 held-out 也最弱）；rep1/rep3 在 OP200 上 Adaptive 高出 1.077、3.804，rep2 反向低 3.829。

| 重复 | 50 A | 50 F | 100 A | 100 F | 200 A | 200 F |
| --- | --- | --- | --- | --- | --- | --- |
| rep1 | 15.134 | 15.198 | 30.482 | 30.411 | 54.761 | 53.684 |
| rep2 | 14.724 | 14.867 | 29.014 | 30.195 | 50.017 | 53.846 |
| rep3 | 15.187 | 14.964 | 30.770 | 29.961 | 56.070 | 52.266 |

搜索 train best：Adaptive 14.868、14.534、14.866；FixedCycle 14.830、14.733、14.646。

### 过程量

Adaptive 的实际调度形态因任务分化（各路数字见 `docs/analysis/v917_scheduler_ablation/summary.json`）：

- 每周期平均 sweep 数 1.59–3.25：CVRP 最高（2.29 / 2.73 / 3.25），TSP rep2 最低（1.59）；FixedCycle 恒为 1。
- Adaptive 每路追加（第 2 及以后 sweep 的）development blocks 25–103 个；其中推进 hypothesis frontier 的 2–47 个（CVRP 30 / 38 / 47 最高），推进 global frontier 的 0–8 个。
- 正 block gain 后下一 block 再成功的比例全部 ≤ 0.47：CVRP 0.42 / 0.47 / 0.46 最高，TSP rep2 与 OP rep3 仅 0.08 / 0.06。
- Discovery 间隔：Adaptive 平均 30.0–41.3，FixedCycle 27.6–27.9。

### 判读

预注册判定的两个事实在任务间分化成立。事实一（不同的连续发展长度）五任务全部出现；事实二（追加 blocks 提高相应结果）只在追加量与命中率同时最高的 CVRP 上完整成立（held-out 9 格 + 训练三重复同向），并在 TSP200、vrptw100/200 的跨规模侧三重复同向兑现。OP 与 TSP 的训练侧、OBP 的容量 500 侧方向不一致，且这两类位置对应的正增益再成功率恰为最低档（0.06–0.24）：正 block gain 对下一段预算投资价值的预测力有限，兑现依赖任务的可连续改进结构。

## 判定

支持自适应调度需要同时出现两个事实：不同任务或运行形成不同的连续发展长度；这些追加发展 blocks 提高相应的 best-at-budget 或最终结果。

若 Adaptive 使用大量追加 blocks，但这些 blocks 很少推进 hypothesis frontier 或 global frontier，并伴随 Discovery 数量下降，则正 block gain 不能充分预测下一段预算的投资价值。

## 运行入口

- Adaptive：`experiments.runners.traceaad.run --version v9_17`
- FixedCycle：`experiments.runners.traceaad.run --version v9_17_fixed_cycle`
- 初始化配对：`experiments.runners.traceaad.capture_v917_initializations`
- FixedCycle 补位：`experiments.runners.traceaad.launch_v917_fixed_cycle`

搜索并发使用 server3/server3b 18 路、local 3 路和 server1 9 路，共 30 路。
