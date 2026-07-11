# TraceAAD 完整实验：机制、算子与参数审计

日期：2026-07-10
对象：`LLM4AD/experiments/tsp_construct/traceaad/20260708_203505/logs/`（完成 run）

## 结论先行

这次实验说明 TraceAAD **已经有一个能产生突破的核心搜索链条，但调度与信用系统没有把突破转化为持续搜索能力**。

真正 work 的不是某个孤立算子，而是：

> fresh basin（novelty） -> 容忍一次轻微退步 -> simplify/rewrite 找到密度-稀疏结构 -> crossover 注入 sparsified-candidate 机制

直接证据是最终最优路径：`-6.9701 -> -6.9867 -> -6.4275 -> -6.4068 -> -6.3713`。其中两次 simplify 和最终 crossover 分别贡献 `+0.5592`、`+0.0207`、`+0.0355`。

目前最主要的不 work 是：

1. **最优解没有被继续开发。** sample 318 产生 final best 后，该节点作为 endpoint/base 的后续选择次数均为 0；之后 659 次评估没有任何全局改进。
2. **portfolio 信用量纲不一致，后期反而越来越偏 novelty。** 后期 novelty 占 51.5% iteration，找到 final best 的 crossover 只占 4.1%。
3. **island migration 产生的是 trajectory ID 多样性，不是程序多样性。** 368 个 fork 重置 visit count；后期 82 次真实 refine 看似使用 77 个 trajectory ID，实际只有 18 个 endpoint。
4. **机制信用并不可信。** distill 重复计算 fork/branch 共享的 edge，保留历史最大 improve rate；关键词标签还会把“移除 randomization”的改进记成 randomization 成功。
5. **novelty 后期退化为 adaptive-exponent 重试循环。** 252 个有效 fresh start，均值 `-11.1579`，最好 `-6.8986`，没有一个超过初始化最好 `-6.8596`。
6. **generalization 当前没有被真实评估。** 所有有效程序的 robustness 都固定为 1.0，scale-transfer 因而总是可触发；本 run 也没有跨规模或 held-out 指标。

因此，这个 run 不是“方法整体不 work”，而是 **生成/组合机制已经能 work，selection、operator credit、knowledge credit 和 population bookkeeping 还没有 work**。后期乏力不是因为 LLM 完全无法再生成局部改进，而是 57 次后期局部改进没有被组织成新的全局突破。

## 审计边界与方法

- 主分析只用完成 run 的原始 `method_events.jsonl`、`method_state.jsonl`、`llm_calls.jsonl`、`samples_*.json`、`run_log.txt` 和 `run_summary.json`，不把已有总结文档当证据。
- 以 `profiler_sample_order` 连接评估、child、LLM call 和 sample code；按源码的 portfolio phase 划分 early `0..164`、mid `165..328`、late `329..497`。
- 重新计算每个算子的 outcome、global-best event、机制标签、AST 节点数、迁移 fork ID、选择集中度、LLM 时间和 portfolio 累积统计。
- `run_config.json` 是迁移后恢复的配置；实际启动参数以 [run_log.txt](../../LLM4AD/experiments/tsp_construct/traceaad/20260708_203505/logs/run_log.txt#L1) 为准。当前 TraceAAD 核心源码与首次集成提交 `f2fade6` 内容一致，但 run artifact 没有记录 commit hash，这是复现信息缺口。
- `logs_run1_crash_at_212/` 是 **pre-fix TraceAAD**，不是 旧版 TraceAAD；其 [run_log.txt](../../LLM4AD/experiments/tsp_construct/traceaad/20260708_203505/logs_run1_crash_at_212/run_log.txt#L40) 明确记录 `Method: TraceAAD`。

## 1. 总体结果与搜索时间线

[run_summary.json](../../LLM4AD/experiments/tsp_construct/traceaad/20260708_203505/logs/run_summary.json#L1) 给出的最终状态为：

| 指标 | 结果 | 判断 |
| --- | ---: | --- |
| wall time | 43,125 s（11.98 h） | 稳定完成 |
| evaluation attempts | 977 | 低于配置的 1000 |
| valid programs | 942 | 96.4% of evaluations；94.2% of program draws |
| eval failed | 35 | 搜索未因此中止 |
| program draws | 1000 | 4 init + 498 iterations x 2 |
| parse failed | 23 | 9 code + 14 novelty，不计 sample budget |
| total LLM calls | 1322 | 1000 program + 322 action calls |
| refine edges / fresh roots | 620 / 318 | 后者不含 4 init |
| trajectories | 1310 | 942 generated + 368 migration forks |
| final best | **-6.371328** | sample 318 / node 308 |
| final logged stagnation | 338 iterations | state 在本轮更新前记录；实际结束为 339 轮无刷新 |

全局 best 只刷新了 10 次：

| sample | operator | score | 对前 best 增益 | gate |
| ---: | --- | ---: | ---: | --- |
| 1 | init | -6.986714 | - | accepted |
| 4 | init | -6.859626 | +0.127088 | accepted |
| 133 | backtrack | -6.823656 | +0.035970 | accepted |
| 141 | endpoint | -6.821159 | +0.002497 | **rejected** |
| 163 | novelty | -6.707123 | +0.114036 | accepted |
| 209 | crossover | -6.702942 | +0.004181 | accepted |
| 267 | simplify | -6.427479 | +0.275463 | accepted |
| 268 | simplify | -6.409565 | +0.017914 | **rejected** |
| 269 | simplify | -6.406812 | +0.002753 | accepted |
| 318 | crossover | **-6.371328** | +0.035484 | accepted |

这些 record 可在 [method_events.jsonl](../../LLM4AD/experiments/tsp_construct/traceaad/20260708_203505/logs/method_events.jsonl#L286)、[sample 267 附近](../../LLM4AD/experiments/tsp_construct/traceaad/20260708_203505/logs/method_events.jsonl#L575) 和 [final best](../../LLM4AD/experiments/tsp_construct/traceaad/20260708_203505/logs/method_events.jsonl#L686) 直接核对。

### 后期不是没有运动，而是运动没有转化

sample 319..977 共消耗：

- 659 次评估、633 个有效 candidate、877 次 LLM call、约 7.42 LLM-hours；
- 381 个 refine child 中仍有 57 improve、127 plateau、197 regress；
- 另外生成 252 个 novelty fresh root；
- 有 20 个 candidate 达到 `>= -6.5`，但没有一个超过 `-6.371328`；
- 继续执行 34 次 distill、42 次 reflect、68 次 migrate，也没有刷新 best。

每 200 samples 的区间最好值为：`-6.7071`（1..200）、`-6.3713`（201..400）、`-6.4096`、`-6.4096`、`-6.4106`。也就是说，sample 318 后搜索仍能回到一个约 `-6.41` 的强 basin，却无法继续开发已经找到的 `-6.3713` basin。[method_state.jsonl 最后一行](../../LLM4AD/experiments/tsp_construct/traceaad/20260708_203505/logs/method_state.jsonl#L498) 记录了 338 轮停滞。

与目录内 pre-fix TraceAAD backup 做 matched-budget 检查也能看出修复的边界：前 106 iterations，pre-fix 是 novelty 43 / backtrack 1 次，完成 run 变为 novelty 26 / backtrack 14 次，说明 trigger 修复确实改变了调度；但 pre-fix 在 sample 212 前已达到 `-6.45395`，完成 run 同期只有 `-6.70294`，到 sample 267 才反超。也就是说，修复让 backtrack 真正触发、压低了早期 novelty，却没有提高 early sample efficiency；完成 run 的优势来自后续 simplify+crossover 突破，而不是前半程全面更强。pre-fix 原始事件见 [backup method_events.jsonl](../../LLM4AD/experiments/tsp_construct/traceaad/20260708_203505/logs_run1_crash_at_212/method_events.jsonl#L1)。

## 2. 最优路径说明了什么

| node / sample | operator / tag | score / delta | AST nodes | 解释 |
| --- | --- | ---: | ---: | --- |
| p223 / 232 | novelty / edge_contrast | -6.970101 | 185 | fresh start 本身不强，但进入了新 basin |
| p224 / 233 | endpoint / local_density | -6.986714 / -0.016613 | 203 | 一次轻微退步被保留，后续可恢复 |
| p257 / 267 | simplify / local_density | -6.427479 / **+0.559235** | 277 | 最大突破；实际是机制重写，复杂度反而 +74 |
| p259 / 269 | simplify / other | -6.406812 / +0.020667 | 238 | 这一步同时提分并减少 39 AST nodes |
| p308 / 318 | crossover / sparsified_candidate | **-6.371328 / +0.035484** | 265 | 在强 base 上加入 dead-end/gateway 机制 |

final crossover 的 action prompt 确实携带三步 trajectory 因果叙事、distilled patterns、contrast 和 sparsified donor，第二个 action 产出最终最好解；可核对 [llm_calls.jsonl](../../LLM4AD/experiments/tsp_construct/traceaad/20260708_203505/logs/llm_calls.jsonl#L443)。这至少证明：

- trajectory context 真实进入了 LLM prompt；
- crossover 的“保留 base、只移植一个机制”约束在这个关键 case 中得到遵守；
- 过程信息有一个明确的正例，但单个正例还不能证明每个 context 字段都有因果贡献。

这个 lineage 也说明 **允许保留退步分支是有价值的**。但它不是 backtrack 的成功证据：真正恢复 p224 的是后续 simplify。

## 3. 算子效力

表中 `I/P/R` 只对有 parent 的 refine child 定义；gate reject 已包含在 valid child 中。

| operator | iterations | valid child | I / P / R | improve rate | gate reject | best | global records | 判断 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| distill_simplify | 91 | 173 | 45 / 55 / 73 | **26.0%** | 48 | -6.406812 | **3** | 有效的质量重写器；复杂度控制仅部分有效 |
| mechanism_crossover | 56 | 110 | 21 / 13 / 76 | 19.1% | 17 | **-6.371328** | **2** | 高风险、重尾突破算子，不能用均值回报评价 |
| endpoint_refine | 85 | 163 | 26 / 46 / 91 | 16.0% | 42 | -6.409565 | 1 | 能局部深化，但 regress 55.8%，exploit 精度偏低 |
| backtrack_branch | 55 | 107 | 14 / 59 / 34 | 13.1% | 30 | -6.592675 | 1 | trigger 已运转；真正内部回溯大多只复制 plateau |
| scale_transfer | 35 | 67 | 9 / 23 / 35 | 13.4% | 10 | -6.396324 | 0 | 当前任务分数有少量收益；泛化未被验证 |
| novelty_jump | 176 | 318 | fresh roots | - | 0 | -6.499879 | 1 | 早期有必要，后期严重过量；均值 -11.4826 |

### 3.1 Simplify：质量搜索 work，复杂度目标不稳定

- 173 个 child 中 99 降低 AST、7 不变、67 增加；median change `-15`，但 mean `+9.47`，说明失败尾部会大幅膨胀代码。
- 45 个 fitness improve 中 30 同时降低复杂度，14 增加复杂度，1 不变。
- improve child 的平均复杂度变化是 `-17.7`，regress child 是 `+46.7`：成功 simplify 确实倾向于更简洁，失败 simplify 则倾向于代码膨胀。
- 后期 50 个 valid child 有 32 regress，平均 AST change `+58.9`，后期已经不再像“simplify”。
- best lineage 第一处最大跃迁虽然名为 simplify，却从 203 增到 277 AST nodes；所以它的实际价值更接近 **structured rewrite**。

结论：保留该算子，但信用应同时使用 fitness 与真实 complexity delta，名称和触发条件也应反映“重写”而非假定一定简化。

### 3.2 Crossover：平均很差，但负责最终突破

- 69.1% child regress，median delta `-0.5401`，但贡献 2 个 global record 和 final best。
- `crossover x sparsified_candidate` 是强组合：20 次中 8 improve，mean delta `+0.0487`，并产生 final best；`crossover x local_density` 则 mean delta `-4.8802`。
- `actions_per_iteration=2` 在这里是有效 hedge：iteration 158 第一个 child `-0.580`，第二个 `+0.0355` 并成为 final best。

当前 portfolio 按每个 child 的 lifetime mean 更新，因此这个关键 iteration 的净信用仍是负数。对 best-so-far 搜索，应至少记录 batch max、global-record event 或 upper-tail success；否则会系统性压制这种必要的高方差算子。

### 3.3 Backtrack：机制触发 work，真正 path correction 基本没 work

[backtrack.py](../../LLM4AD/llm4ad/method/traceaad/operators/backtrack.py#L23) 已能主动从 pool 选退步/饱和 trajectory，因此不再是“完全不触发”。但 55 次选择中：

- 36 次 base 真正在 endpoint 之前（29 `endpoint_not_best` + 7 `recent_plateau`），产生 71 valid child；
- 相对内部 base 仅 2/71 improve；相对被放弃 endpoint 仅 9/71 严格变好；
- `recent_plateau` 13 个 child 中 0 improve；
- 其余 19 次标成 `backtrack_internal`，实际 `base == endpoint`，产生了 12/14 个表面 backtrack improve，本质上是 endpoint rewrite。

因此可以说“独立选题修复让 backtrack 运转起来了”，但不能说 path correction 已有效。它目前主要重新生成相同 plateau，只有 sample 133 是一次真正的内部回溯 record。

### 3.4 Scale-transfer：不能从这次 run 宣称泛化有效

[scale_transfer.py](../../LLM4AD/llm4ad/method/traceaad/operators/scale_transfer.py#L17) 用 `robustness >= 0.5` 触发，但评估代码对所有 valid program 写死 `robustness=1.0`（[traceaad.py](../../LLM4AD/llm4ad/method/traceaad/traceaad.py#L473)），所以该算子总是 eligible，而不是被泛化证据触发。

它在 train aggregate score 上有 9/67 improve、最好 `-6.3963`；但没有 per-instance vector、held-out split 或跨规模评估。51/67 child 还增加 AST，mean `+49.5`。目前最多只能说“generalize prompt 偶尔带来当前 task 的好改写”，不能说 scale-transfer 或泛化机制 work。

## 4. 阶段行为与后期乏力

### 4.1 分配方向与设计目标相反

| phase | novelty | simplify | endpoint | crossover | backtrack | scale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| early, 165 iter | 22.4% | 24.8% | 19.4% | 12.1% | 12.7% | 8.5% |
| mid, 164 iter | 31.7% | 14.0% | 15.9% | 17.7% | 14.0% | 6.7% |
| late, 169 iter | **51.5%** | 16.0% | 16.0% | **4.1%** | 6.5% | 5.9% |

源码的 role bonus 明确想让 explore 由 `0.2 -> 0.05`，让 exploit/simplify 后期升到 `0.5/0.4`（[portfolio.py](../../LLM4AD/llm4ad/method/traceaad/portfolio.py#L42)）。实际却是探索越到后期越多。

原因不是单纯随机波动，而是 credit 的量纲和定义不同：

- refine gain 是原始 fitness delta，可轻易出现 `-1..-10`；
- novelty gain 是当前 fitness range 内的归一化 relative quality，范围 `[-1, 0]`；新 best 又在计算前先更新 incumbent，因此不可能得到正 gain（[traceaad.py](../../LLM4AD/llm4ad/method/traceaad/traceaad.py#L274)）；
- novelty fresh child 只要 valid 就记 `novel=True`，从不记 regress；
- portfolio 再对 lifetime mean 做 `tanh` 并加 valid/novel bonus（[portfolio.py](../../LLM4AD/llm4ad/method/traceaad/portfolio.py#L85)）。

按真实事件重建的最终 late portfolio value 约为：

| novelty | simplify | endpoint | backtrack | scale | crossover |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **+0.597** | +0.215 | +0.076 | -0.162 | -0.249 | **-0.412** |

`tau_end=0.3` 会进一步锐化这个错误排序。也就是说，portfolio 把“质量很差但语法有效且结构新”的 fresh start 看得远高于“平均回撤很大但能创造 global best”的 crossover。

### 4.2 各算子后期也在退化

refine outcome 的 early / mid / late improve counts：

- simplify：`29/78 -> 7/45 -> 9/50`，late regress 32/50；
- endpoint：`13/62 -> 5/50 -> 8/51`，late regress 39/51；
- crossover：`9/40 -> 9/57 -> 3/13`，有效率尚有 23.1%，但 allocation 被压缩；
- backtrack：`4/42 -> 9/45 -> 1/20`；
- scale：`4/27 -> 2/22 -> 3/18`。

后期确实更难，但不是所有突破算子都失去能力：crossover 晚期仍有 3/13 local improve，主要问题之一是它只获得了 7 个 iteration，而 novelty 获得 87 个。

## 5. Trajectory selection 与 islands

### 5.1 final best 从未被选择

final best node 308 / trajectory 404 通过 gate 后，后续 endpoint 选择 0 次、base 选择 0 次。accepted record node 157（sample 163）和 200（sample 209）也都是 0 次；另有两个被 gate 拒绝的 record node 136、258 同样无法继续发展。

与之相反，late phase 最常被选的是：

- node 257 (`-6.4275`)：60 次；
- node 52 (`-6.9867`)：50 次；
- node 881 (`-7.6607`)：16 次；
- final best node 308 (`-6.3713`)：**0 次**。

这直接否定了“当前 trajectory value 能稳定 exploitation best”的假设。

可能的实现与参数原因都有原始证据：

1. quality 只占 scalar 的 0.30，potential/generalization 合计 0.50，diversity/novelty 0.20（[value.py](../../LLM4AD/llm4ad/method/traceaad/value.py#L30)）。
2. min-max normalization 被极差 outlier 压缩。最终 `fmin=-35.5446` 时，baseline `-6.9867` 的 normalized quality 已是 `0.9789`，与 best 的加权 scalar 差仅约 `0.0063`。
3. 只在 scalar top-5 内以 temperature 0.8 抽样；掉出 top-5 就没有机会（[value.py](../../LLM4AD/llm4ad/method/traceaad/value.py#L118)）。
4. survival 试图保 best，但把 `best_node.id` 与 `trajectory.id` 比较（[traceaad.py](../../LLM4AD/llm4ad/method/traceaad/traceaad.py#L394)），并不能保证包含 best node 的 trajectory 存活。
5. 配置的 `sampling_strategy` 只被保存，主循环无条件调用 trajectory-UCB（[traceaad.py](../../LLM4AD/llm4ad/method/traceaad/traceaad.py#L238)）；`best/random` 实际不可切换。

这里还有一个设计与实现的差距：`ValueVec` 最终被 scalarize，island survival 也直接按 `scalar_value` 排序截断（[traceaad.py](../../LLM4AD/llm4ad/method/traceaad/traceaad.py#L394)），并没有执行设计说明中的 non-dominated survival；runtime 和 complexity 也不在 [ValueVec](../../LLM4AD/llm4ad/method/traceaad/schema.py#L96) 中。因此当前所谓“多目标”实际是五个代理分量的固定加权和，尚不是 Pareto survival，也没有把真实计算成本纳入 trajectory 价值。

`V_potential` 还展示出一个具体偏差：498 次 selection 中有 260 次（52.2%）选中的 endpoint 分数精确等于 baseline `-6.9867137`。例如 node 52 的 path 从 `-7.7371` 恢复到 baseline，单步得到 `+0.7504`，因此 path return/positive ratio 很好；它在 final best 出现后仍被选 83 次。node 104 也来自一条大幅恢复到 baseline 的 path，并被反复选择。也就是说，当前 value 会奖励 **“从很差起点英勇恢复、但终点仍普通”** 的轨迹；在 absolute quality 又被 outlier 压缩时，这类 potential 足以压过真正 best。修复方向不是删除 trajectory value，而是给 potential 加 endpoint-quality floor 或采用 lexicographic guard。

### 5.2 Migration 扩大了 clone，不是有效覆盖

- 97 次 migrate 共 fork 368 条 trajectory，正好解释 `1310 - 942 = 368`；[islands.py](../../LLM4AD/llm4ad/method/traceaad/islands.py#L21) 复制同一路径到其他岛。
- `fork()` 会把 `visit_count` 重置为 0（[trajectory_memory.py](../../LLM4AD/llm4ad/method/traceaad/trajectory_memory.py#L88)），因此相同程序 clone 会重新获得 UCB 探索优势。
- 322 次真实 refine 中，224 次（69.6%）选择的是 migration-only trajectory ID。
- late refine 的 clone-ID 比例 86.6%；82 次 refine 使用 77 个 nominal trajectory ID，却只有 18 个 endpoint，top-5 endpoint 占 79.3%。
- 整个 run 有 416 个不同 selected trajectory ID，但只有 84 个 endpoint node。236/938 child 甚至得到与 baseline 完全相同的 aggregate score，其中 164 个仍通过 novelty gate。

所以，islands/migration 的机制“有运行”，但本 run 展示的是 **ID 层扩张 + 程序层收缩**。迁移是否带来跨岛机制组合没有正证据；clone 重置 UCB 反而有明确的干扰证据。

`max_active_trajectories=1000` 也没有实际约束力。4 islands x `max_per_island=40` 使有效 cap 约为 160；iteration 161 首次达到 160，之后 state 在 survival 前/迁移后偶尔为 164。真正起作用的是 `max_per_island`，不是全局 1000。

## 6. Novelty 与 novelty gate

### 6.1 Fresh exploration 早期有价值，后期策略失效

novelty 产生过一个 global record（sample 163，`-6.7071`），final-best lineage 也起源于 sample 232 的 fresh root。因此“需要能跳出当前 path 的 fresh exploration”是 work 的。

但最终分配明显失控：

| requested tag | valid roots | mean score | best | 超过 init best `-6.8596` |
| --- | ---: | ---: | ---: | ---: |
| adaptive_exponent | **252** | -11.1579 | -6.8986 | **0** |
| nn_rank | 24 | -13.1830 | -7.0428 | 0 |
| local_density | 22 | -11.1642 | -6.7071 | 1 |
| other | 13 | -12.4879 | -6.4999 | 2 |
| randomization | 5 | -16.0211 | -7.2931 | 0 |
| edge_contrast | 2 | -17.6090 | -6.9701 | 0 |

[novelty.py](../../LLM4AD/llm4ad/method/traceaad/operators/novelty.py#L29) 在没有可用 top mechanism 时按 **active endpoint count 最少** 选 family。低质 adaptive roots 很快被 survival 淘汰，所以 active count 一直低，系统便把“持续失败”误认为“仍未充分探索”，从 iteration 131 到 487 重试 252 次。

这也纠正了一个容易混淆的旧判断：pre-fix backup 中 adaptive-exponent 的成功主要发生在 simplify/endpoint/crossover 的有 parent 改写；完成 run 把这份 family credit 转移成 fresh-start 任务后完全失败。**机制信用是 operator/context 条件化的，不能只按 family 全局迁移。**

### 6.2 Gate 在过滤相似路径，但没有保护高价值路径

- 620 个 refine child 中拒绝 147（23.7%），包括 25 个 improve、67 plateau、55 regress；两个当时的 global record（sample 141、268）也被拒绝。
- 24 个 score `>= -6.5` 的 refine candidate 中拒绝 5 个。
- 318 个 valid novelty fresh root 全部通过，尽管均值只有 `-11.4826`。

这不代表阈值 0.92 单独“太高或太低”；它说明结构相似度 gate 与 best-so-far 目标没有对齐。当前应先增加 **quality override / elite exception** 和 behavioral equivalence，再考虑调 threshold。仅调 0.92 无法解决低质 fresh root 全过、高质相似 child 被封的问题。[gate 实现](../../LLM4AD/llm4ad/method/traceaad/traceaad.py#L413) 只看最大相似度。

## 7. 机制族与 PatternMemory

### 7.1 原始 tag 统计（只能作为弱证据）

| tag | refine n | I / P / R | improve rate | fresh roots | best | 判断 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| row_normalize | 24 | 9 / 2 / 13 | 37.5% | 0 | -6.4096 | 小样本 promising，未产生 record |
| nn_rank | 38 | 10 / 9 / 19 | 26.3% | 24 | -6.4280 | parent rewrite 有潜力；fresh generation 差 |
| sparsified_candidate | 141 | 33 / 57 / 51 | 23.4% | 0 | **-6.3713** | 与 crossover 组合是最强直接证据 |
| local_density | 209 | 35 / 78 / 96 | 16.7% | 22 | -6.4096 | best lineage 核心中间机制，但总体噪声大 |
| other | 97 | 14 / 15 / 68 | 14.4% | 13 | -6.4068 | 110/938 child 无法可靠归类 |
| generalize | 85 | 11 / 23 / 51 | 12.9% | 0 | -6.3963 | 这是 operator objective 标签，不是实证机制族 |
| edge_contrast | 17 | 1 / 11 / 5 | 5.9% | 2 | -6.9701 | direct effect 弱；提供 final lineage root |
| adaptive_exponent | 1 | 0 / 0 / 1 | 0% | 252 | -6.8986 | 当前 fresh-start 用法明确失败 |
| randomization | 8 | 2 / 1 / 5 | 25.0% | 5 | -6.9867 | 两个“成功”实际都在移除随机化 |

标签可靠性有四个限制：

1. [infer_mechanism_tag](../../LLM4AD/llm4ad/method/traceaad/operators/base.py#L21) 只是按固定顺序做关键词 first-match；244/620 action（39.4%）同时命中多个 family。
2. 46/209 个 `local_density` action 同时明确提到 sparsified，标签无法表达混合机制。
3. crossover 强制继承 donor tag，scale 强制写 `generalize`，novelty 强制写 requested family，均未验证生成代码是否实现该语义。
4. 唯二 `randomization`-tagged improve（[event line 22](../../LLM4AD/experiments/tsp_construct/traceaad/20260708_203505/logs/method_events.jsonl#L22)、[line 263](../../LLM4AD/experiments/tsp_construct/traceaad/20260708_203505/logs/method_events.jsonl#L263)）的 action 都明确是 replace/eliminate randomization。关键词系统把反机制改动记成了该机制成功。

因此当前最可信的机制结论来自 **action + parent/child + operator 的具体组合**，不是 family aggregate：

- `simplify -> density/sparsity rewrite`：产生最大单步突破；
- `crossover x sparsified_candidate`：20 次、8 improve、mean delta `+0.0487`、产生 final best；
- `simplify x row_normalize`：12 次、6 improve、mean delta `+0.3371`，值得定向复验；
- `adaptive_exponent x fresh_start`：252 次、0 次超过 init best，明确应停。

### 7.2 Distill/reflect 已进入 prompt，但信用失真

49 次 distill（iteration 10..490）和 59 次 reflect（8..495）都真实触发；311/322 个 action prompt 显示 recurring mechanisms，308 个显示 lessons。说明知识回路的 **plumbing 是 work 的**。

但内容质量不 work：

- 297/308 个含 lesson 的 prompt，其 top lessons 内部已有重复文本；主要反复显示 sparsified-candidate 和 nn-rank。
- prompt 从 iteration 11 到结束一直显示 `randomization generalization=1.00 support=1`；local-density 后期也固定为 1.00，而 unique refine improve rate 只有 25.0% 和 16.7%。可在 [iter 20 prompt](../../LLM4AD/experiments/tsp_construct/traceaad/20260708_203505/logs/llm_calls.jsonl#L62)、[iter 140](../../LLM4AD/experiments/tsp_construct/traceaad/20260708_203505/logs/llm_calls.jsonl#L395) 和 [iter 489](../../LLM4AD/experiments/tsp_construct/traceaad/20260708_203505/logs/llm_calls.jsonl#L1302) 对照。
- 56 个 action 明确引用 `generalization` evidence，只有 6 improve、37 plateau、13 regress，最好仅 `-6.8090`；错误信用不是无害展示，它确实进入了决策上下文。

实现原因明确：

- distill 遍历 `memory.trajectories()`，包含 archived trajectory 和 migration fork，并对共享 edge 重复计数（[reflection.py](../../LLM4AD/llm4ad/method/traceaad/reflection.py#L14)）；
- `upsert_mechanism` 使用历史 `max(old, new)`，早期幸运 rate 永不下降；`support_id=-1` 又使可见 support 永远为 1（[pattern_memory.py](../../LLM4AD/llm4ad/method/traceaad/pattern_memory.py#L35)）；
- reflect 每次新增模板化 lesson，不按语义或 support node 去重。

所以“蒸馏与反思触发了”成立，“它们形成了可靠跨轨迹知识”不成立。

所谓 robust comparative feedback 也没有真正发挥设计中的作用。[RankingModel](../../LLM4AD/llm4ad/method/traceaad/feedback.py#L16) 会更新 Elo-like `_scores`，但 `contrast()` 最后仍按原始 endpoint fitness 排序选择 best/worst；仓库中没有消费者读取 `rank()`。因此当前 prompt 里的 contrast 是近期 active endpoint 的普通 fitness 对比，不是经过相对排名或置信度校正后的鲁棒反馈。

## 8. 参数配置审计

实际参数见 [run_log.txt method block](../../LLM4AD/experiments/tsp_construct/traceaad/20260708_203505/logs/run_log.txt#L41)；恢复版配置见 [run_config.json](../../LLM4AD/experiments/tsp_construct/traceaad/20260708_203505/run_config.json#L25)。

| 参数 | 实际行为 | 结论 |
| --- | --- | --- |
| `max_sample_nums=1000` | 恰好做 1000 次 program draw，但 23 parse fail 不计 budget；固定 498 iterations 后以 977 eval 结束 | **不符合 sample-budget 语义**；应补抽直到 eval=1000，或改名 draw budget |
| `n_init=4` | 4 个 root 正常生成；tag 只有 `other, local_density, nn_rank, other`，只占 3 islands | 小预算启动 work；“强制四机制族”未完全实现 |
| `actions_per_iteration=2` | 322 个 action prompt 全部解析出恰好 2 个 action；两次 record iteration 都由第二个 action 抵消第一个失败 | **work，建议保留**；portfolio 应按 batch max/record 给信用 |
| `max_trajectory_length=8` | action prompt 硬截到最近 5 条 edge；graph ancestry 有 64 个 node 到达 depth 8、没有 depth > 8 | 上限可能已成为实际边界，但日志不足以区分“参数限制”与“depth-8 trajectory 被 selection/gate 淘汰”；需定向 ablation |
| `max_active=1000` | 被 `4 x max_per_island=40` 提前限制在约 160 | 1000 是 inert 参数；明确写 160 或重新设计两层 cap |
| `n_islands=4`, `max_per_island=40` | survival cap 生效；migration 产生 368 clone 并重置 UCB visits | cap work，migration 效果不 work |
| `novelty_threshold=0.92` | refine reject 23.7%，fresh accept 100%；拒绝 25 improve | gate 机械生效，但与质量目标失配；不宜先盲调阈值 |
| `k_distill=10` | 准确触发 49 次 | 频率 work，统计对象/去重不 work；修复信用前可降低频率 |
| `patience_reflect=8` | 触发 59 次，大量重复 lesson | trigger work，信息增量低；应按新 evidence 触发而非固定重复 |
| value weights `0.30/0.25/0.10/0.10/0.25` | quality 被 outlier min-max 压缩；伪 generalization 与 clone-UCB 可压过真实 best | 当前组合 **不 work**；先修 normalization/credit，再做权重 ablation |
| `top_k=5`, trajectory temp `0.8`, `c0=0.4` | hard top-5 使 best trajectory 掉队后无恢复机会；migration fork 重置 visit 放大 UCB | 与 clone migration 组合不 work |
| portfolio `tau 1.0 -> 0.3` | 后期把错误 credit 排序变得更尖锐 | 调度不 work；根因是 reward scale/目标，不只是 tau 数值 |
| novelty trigger stagnation `>=5` | 很早进入长期 eligible 状态，最终占 176 iterations | 对当前 portfolio 太激进；需 cooldown、family failure memory 和 late cap |
| `num_evaluators=4` | 每次 submit 后立即 `future.result()`，candidate 间没有并发 | method-level 4 workers inert；但 eval 只占 0.35% wall，不是主要瓶颈 |
| LLM temp `1.0` / max tokens `16384` | program parse 97.7%，action parse 100%；无对照 run | 工程可用，质量影响 **未检验**；不能从单 run 判最优 |
| task/method random seed | task log 为 `None`，operator/softmax/migration 也未记录固定 seed | 不利于科研对比与归因，应固定并做多 seed |

另外，wall time 的 97.7% 来自 LLM generation（`llm_calls.sample_time` 合计 42,144 s），evaluation 仅 151.3 s；但 portfolio 的 cost 只累计约 0.15 s 的 evaluator runtime，不统计 LLM action/code generation。`delta_c=0.05` 因而没有优化真正成本。

## 9. Work / Not work / Unproven

### 有直接证据 work

1. **跨算子路径组合**：fresh basin + 可恢复退步 + density/sparsity rewrite + crossover 能形成 final best。
2. **Simplify 作为 structured rewrite**：贡献最大单步增益和 3 个 records；成功时通常同时减复杂度。
3. **Crossover x sparsified-candidate**：高方差但正 mean delta，并产出 final best。
4. **两 action hedge**：关键 iteration 的第二个 proposal 两次救回第一个 proposal 的失败。
5. **trajectory narrative / operator constraint 的 plumbing**：关键 prompt 中可见，且 final action 遵守约束。
6. **工程鲁棒性**：1000 program draws、1322 LLM calls、一个 HTTP 502 自动重试，搜索 0 fatal error 完成。

### 已触发但效果弱或当前不 work

1. **Elite exploitation / trajectory selection**：final best 后 0 次选择，是最直接的失败证据。
2. **Operator portfolio**：不同比例 reward 混算，后期把 novelty 放大、crossover 压低。
3. **Novelty family targeting**：active-count 逻辑形成 adaptive-exponent 失败循环。
4. **Islands migration**：制造 fork-ID/UCB 优势，程序 endpoint 反而高度集中。
5. **True backtrack**：36 次内部回溯 iteration 只有 2 次 child 超过内部 base。
6. **Novelty gate**：能过滤结构相似，但会切断 improving/record path，也挡不住行为等价和低质 fresh root。
7. **PatternMemory credit**：clone-weighted、max-so-far、support=1、标签可语义反转。
8. **Simplify 的复杂度控制**：质量改写有效，但 38.7% child 增加 AST，后期明显膨胀。
9. **Robust ranking / non-dominated survival**：Elo score 没有进入 contrast 或 selection，survival 仍是 scalar top-k；设计中的鲁棒相对反馈和 Pareto 生存尚未落地。

### 这次 run 不能证明

1. scale-transfer 或 `V_generalization` 的真实跨 instance / 跨规模泛化；
2. 0.30/0.25/... value weights、top-k=5、temperature=0.8 的最优性；
3. `max_trajectory_length=8` 的价值；
4. Pattern/reflect 相对“只给 trajectory + operator prompt”的净贡献；
5. 单次 stochastic run 相对其他 method 的统计优势。

## 10. 下一轮最小验证顺序

优先级应按因果链修复，而不是先做大范围参数 sweep：

1. **先修 elite 与重复对象**：显式保留/抽样 best trajectory；migration fork 不重置或共享 visit；selection/value/distill 按 unique path/edge 去重。
2. **统一 portfolio reward**：所有算子使用同一归一化目标；按 iteration batch max、global-record/near-record 和真实 LLM cost 记信用；新增 operator x mechanism 条件信用。
3. **修 novelty failure memory**：按历史 fresh attempts/success 计数，不按 surviving active count；对 family 设置 cooldown/失败上限，late novelty allocation cap。
4. **修 knowledge credit**：unique edge 统计、可下降的后验估计、真实 support、add/remove 方向标签、lesson 去重。
5. **再调 value 参数**：用 robust/rank normalization，比较更高 quality weight、移除伪 generalization、top-k 5 vs 10；此时参数 ablation 才有解释力。
6. **最后验证泛化**：加入 held-out/per-instance/多规模 fitness vector 后再启用 scale-transfer 与 robustness trigger。

最小对照至少应固定 LLM、task、sample/draw budget 和随机种子，跑：当前配置、`elite+dedup`、`elite+dedup+unified-credit` 三组；重点看 `best sample order`、post-best conversion rate、operator x mechanism upper-tail success、unique endpoint coverage 和 LLM-hours per record，而不只看最终 best。

## 11. 实施状态（2026-07-10）

第 10 节的代码级修复已经按因果顺序落地到 `LLM4AD/llm4ad/method/traceaad/`；正式 runner 也已改为明确的 active cap、quality-first 权重、关闭伪泛化、固定 task/search seed，并记录 value 与 portfolio 参数。详细映射、参数表和新实验矩阵见 [`docs/worklog/traceaad-audit-driven-redesign.md`](../worklog/traceaad-audit-driven-redesign.md)。

当前没有 TraceAAD run 在运行。`20260708_203505` 仍是旧机制的只读历史 artifact，不能用于验证本轮修复；新机制是否改善后期转化、elite 利用和 LLM-hours per record，必须由下一轮固定预算、多 seed 实验回答。

实现完成后的验证结果为 TraceAAD 专项 `74 passed`、全仓 `171 passed, 2 subtests passed`；standards 与 spec 双轴最终 re-review 均 clean。代码层结论已经闭环，效果层结论仍以新实验为准。
