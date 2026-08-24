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

覆盖范围：OBP 两臂各 3 个重复；VRPTW Adaptive 3 个重复；TSP Adaptive rep2、rep3 与 FixedCycle rep1、rep3；OP Adaptive rep1、rep3。OBP 的 bins 数值越低越好，配对差为 Adaptive − FixedCycle，负值表示 Adaptive 更少 bins。

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

### VRPTW（Adaptive 三重复）

held-out test split（均值总距离，越低越好）：rep1 19.183、rep2 19.626、rep3 18.443，均值 19.084 ± 0.598；对应搜索 train best 19.368、19.044、18.779。FixedCycle 侧无已完成重复，无配对差。

### TSP（Adaptive rep2、rep3；FixedCycle rep1、rep3）

held-out（均值总距离，越低越好）：

| 重复 | 臂 | train best | TSP50 | TSP100 | TSP200 |
| --- | --- | --- | --- | --- | --- |
| rep2 | Adaptive | 5.875 | 5.835 | 8.126 | 11.536 |
| rep3 | Adaptive | 6.138 | 6.231 | 8.559 | 11.902 |
| rep1 | FixedCycle | 6.078 | 6.173 | 8.561 | 11.925 |
| rep3 | FixedCycle | 5.848 | 5.930 | 8.452 | 12.380 |

rep3 为同 seed 配对：FixedCycle 的 train best 与 TSP50/100 更低（6.138 对 5.848、6.231 对 5.930、8.559 对 8.452），Adaptive 的 TSP200 更低（11.902 对 12.380）；跨规模方向不一致，完整配对差待 Adaptive rep1 与 FixedCycle rep2。

### OP（Adaptive rep1、rep3）

held-out（均值收益，越高越好）：rep1 15.134 / 30.482 / 54.761，rep3 15.187 / 30.770 / 56.070（OP50/100/200）；对应 train best 14.868、14.866。FixedCycle 侧无已完成重复，无配对差。

## 判定

支持自适应调度需要同时出现两个事实：不同任务或运行形成不同的连续发展长度；这些追加发展 blocks 提高相应的 best-at-budget 或最终结果。

若 Adaptive 使用大量追加 blocks，但这些 blocks 很少推进 hypothesis frontier 或 global frontier，并伴随 Discovery 数量下降，则正 block gain 不能充分预测下一段预算的投资价值。

## 运行入口

- Adaptive：`experiments.runners.traceaad.run --version v9_17`
- FixedCycle：`experiments.runners.traceaad.run --version v9_17_fixed_cycle`
- 初始化配对：`experiments.runners.traceaad.capture_v917_initializations`
- FixedCycle 补位：`experiments.runners.traceaad.launch_v917_fixed_cycle`

搜索并发使用 server3/server3b 18 路、local 3 路和 server1 9 路，共 30 路。
