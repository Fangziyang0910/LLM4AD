# TraceAAD V9.8 机制识别实验分析

> 状态：Stage P 已完成。P1/P2 批次为 `20260815_221500_v98_p1_p2`，P3 批次为 `20260816_001100_v98_p3`。本文只分析固定锚点生成与强制续段；1000-eval 正式批次 `20260815_225000` 仍在运行，不能用本文结果替代完整搜索或 held-out 结论。

## 1. 核心判断

1. **Refine 与 Explore 产生了可区分的单步 proposal behavior。** 在 code-only 和 parent-path 两种上下文下，Explore 的静态宏簇切换率与代码修改比例几乎都高于 Refine，但即时有向质量变化更低。V9.8 把两者定义为“发展当前方向”与“提出替代方向”具有行为依据，不只是算子重命名。该实验只测量了若干边际统计，不声称已识别完整 kernel 或真实 family transition。
2. **Parent path 对 Refine 的作用具有跨任务一致方向。** 在四个任务中，`parent_path - code_only` 的锚点级配对即时质量均值都为正，且正向锚点多于负向锚点。它同时使 Refine 的修改更集中、静态宏簇切换率下降。来时路作为当前方向 development context 得到支持。
3. **Parent path 对 Explore 的作用没有同样稳定。** 四任务的条件配对均值都略为正，但 CVRP 为 7 正 / 7 负，OBP 为 7 正 / 9 负 / 2 平，且方差很大；有效率也没有统一方向。当前证据不支持把“来时路帮助 Refine”直接外推为“同样帮助 Explore”。
4. **Explore child 存在可观测的短期延迟发展，但高度任务异质。** 五步 Refine 后，OBP 与 OP 的 parent recovery 明显增加，TSP 仅小幅增加，CVRP 基本没有越过原父代。短续段机会能够救回一部分 child，但不是普遍有效，也不等于长期潜力。
5. **Hypothesis-level 局部重选没有稳定优于 child-chain 深挖。** H1/H3 时 CVRP、TSP 的配对均值有时偏向区域级重选，但 H5 时四任务的均值都略偏向 child-chain。当前局部锚点规则是可运行的 baseline，不是已识别的最优 development policy。这一结果只比较 hypothesis 内的两种续段协议，不评价 hypothesis boundary 或聚合统计是否有用。

这些结果只达到“机制运行并改变行为”。边界宽限 $C$、历史发展项 $M$、hypothesis 聚合以及完整 $Q+U+C+M$ 是否改善有限预算搜索，仍必须由 Stage A 消融和完成的正式搜索回答。

## 2. 协议完成与数据完整性

### 2.1 P1/P2：History × Intent

- 72 个固定锚点，覆盖四任务、low / middle / high 三个质量层；
- 每个锚点三次重复、四个条件，共 864 个响应；
- 216 个 `anchor × replicate` block 全部包含四条件，每条件恰好 216 条；
- 864 个唯一 trial ID 与 864 个原始模型调用一一对应；
- 746 个有效响应，118 个无效响应，850 次真实 evaluator 调用；
- 主统计先在同一锚点内平均重复观测，再以源锚点为独立单位。

### 2.2 P3：Explore child 强制续段

- 179 个有效且非 no-op 的 `parent_path × Explore` 观测中，按预注册规则为每个源锚点选择第一个可用 child；
- 69 个源锚点进入 P3：CVRP 15 个，其余三任务各 18 个；
- 每个 child 配对运行 child-chain 与 hypothesis-level 两种协议，每种协议五步，共 690 个响应；
- 138 条 continuation 全部包含恰好 step 1–5，690 个唯一 `(continuation_id, step)` 与 690 个原始调用一一对应；
- 545 个有效响应，145 个无效响应，659 次真实 evaluator 调用。

两阶段都采用同进程逐响应流水：一次生成落盘后立即解析和评价，再进入下一次生成。上述完成数来自冻结后的事实层验收，不来自控制台近似计数。

## 3. P1：Operator 是否改变单步生成

下表的即时差为锚点内配对的 `Explore - Refine` 有向质量变化。正 / 负表示锚点级配对均值的方向；不同任务的 $q$ 尺度不同，不能横向比较差值幅度。宏簇切换率和修改比例展示 parent-path 条件下的锚点级均值。

| 任务 | Code-only 即时差，正/负 | Parent-path 即时差，正/负 | Parent-path 宏簇切换率 R / E | Parent-path 修改比例 R / E |
| --- | ---: | ---: | ---: | ---: |
| TSP | -1.527，4/12 | -1.345，4/12 | 0.222 / 0.435 | 0.736 / 0.917 |
| CVRP | -4.171，2/12 | -7.063，1/13 | 0.204 / 0.583 | 0.375 / 0.814 |
| OP | -1.238，3/15 | -1.374，1/17 | 0.185 / 0.685 | 0.652 / 0.843 |
| OBP | -804.083，2/16 | -749.567，1/17 | 0.148 / 0.444 | 0.476 / 0.801 |

Explore 在所有任务和两种上下文下都更常产生即时退步，同时在 parent-path 条件下都更常切换静态机制代理并形成更大代码变化。Code-only 条件也保持相同总体结构。由此可确认 operator 指令改变了生成分布；不能把 Explore 的更高切换率解释为真实 family discovery，也不能因即时质量更低判定其后续价值为零。

## 4. P2：Parent path 分别怎样作用于两种 Operator

下表报告锚点级配对的 `parent_path - code_only` 即时有向质量，形式为均值 ± 样本标准差；方向计数为正 / 负 / 平。有效率是三次重复先在锚点内求比例后的均值。

| 任务 | Refine 配对差，方向 | Refine 有效率 C / H | Explore 配对差，方向 | Explore 有效率 C / H |
| --- | ---: | ---: | ---: | ---: |
| TSP | +0.574 ± 1.047，9/5/2 | 0.870 / 0.852 | +0.756 ± 4.567，11/5/0 | 0.870 / 0.852 |
| CVRP | +3.080 ± 4.241，12/2/0 | 0.667 / 0.778 | +0.188 ± 3.898，7/7/0 | 0.574 / 0.556 |
| OP | +0.515 ± 0.686，13/5/0 | 0.981 / 0.981 | +0.379 ± 1.980，12/6/0 | 0.963 / 0.944 |
| OBP | +36.590 ± 103.804，14/3/1 | 1.000 / 0.981 | +91.106 ± 648.913，7/9/2 | 0.963 / 0.981 |

其中 C 表示 code-only，H 表示 parent-path。Refine 的四任务均值与方向计数一致支持 parent path；CVRP 的有效率也明显增加。Explore 的条件均值虽都为正，但 CVRP 与 OBP 的方向计数不稳定，标准差远大于均值，有效率也有升有降。因此，第一版完整 V9.8 可以把 parent path 保留为共同 prompt baseline，但关于 Explore 的独立科学主张应保持未验证；后续可以单独比较 Explore 的局部来时路、机制摘要与全局已探索区域提示。

## 5. P3：短续段能否发展或救回 Explore Child

下表给出 H5 时相对 Explore 入口的平均 internal gain，以及相对原父代的 recovery rate 从 H0 到 H5 的变化。H0 已包含 Explore child 自身，所以其 recovery rate 可以非零。

| 任务 | $n$ | Child-chain H5 internal gain | Child-chain recovery H0 → H5 | Hypothesis-level H5 internal gain | Hypothesis-level recovery H0 → H5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TSP | 18 | 0.884 | 0.389 → 0.444 | 0.767 | 0.389 → 0.389 |
| CVRP | 15 | 1.543 | 0.067 → 0.067 | 1.355 | 0.067 → 0.067 |
| OP | 18 | 0.587 | 0.222 → 0.389 | 0.372 | 0.222 → 0.278 |
| OBP | 18 | 585.347 | 0.111 → 0.611 | 584.722 | 0.111 → 0.500 |

到 H5 时，child-chain 观察到 internal gain 的 child 数为 TSP 10/18、CVRP 10/15、OP 14/18、OBP 14/18；hypothesis-level 分别为 8/18、6/15、16/18、15/18。发展发生不等于越过原父代：CVRP 两协议都只有 1/15 最终 recovery，而 OBP 分别达到 11/18 与 9/18。

Hypothesis-level 减去 child-chain 的 H5 internal gain 配对均值为：TSP -0.117 ± 0.843、CVRP -0.188 ± 1.946、OP -0.215 ± 0.937、OBP -0.625 ± 6.324。H1/H3 的部分任务方向相反，因此不能形成“区域重选优于单链”或相反的统一结论。P3 支持的是短续段机会具有任务相关价值，而不是某种固定 continuation policy 已经最优。

## 6. 回到原子搜索循环的解释

V9.8 的基本操作仍然是“选一个锚点和 operator，生成一个 child，评价后更新状态”。Stage P 没有评价整个调度器，它只识别这个原子循环中的三个问题。

### 6.1 Operator 决定要采样的 transition 类型

记给定锚点、上下文和 operator 时的生成分布为：
$$
K_o(x'\mid a,h)=P(x'\mid a,h,o).
$$

P1 表明 $K_R$ 与 $K_E$ 在即时质量、静态机制代理切换和修改规模上有一致的可观测差异。这是后续区分 development 与 departure 的必要条件：若 operator 不改变 proposal behavior，则 operator-specific 上下文与分配都没有可利用的差异。当前数据不识别分布全貌，也不识别 Explore 是否进入更好的真实算法族。

### 6.2 历史不具有脱离 Operator 的统一价值

P2 识别的是 interaction：parent path 对 Refine 的即时质量呈跨任务正向，对 Explore 则不稳定。同时观察到 Refine 修改更集中与质量更高，但本实验没有对“理解当前方向 → 修改集中 → 质量改善”做中介分析，因此该链条只是候选解释。对 Explore 而言，parent path 也可能同时提供问题信息并形成 trajectory anchoring。这同样尚未识别。

因此更基本的设计对象不是“轨迹有无价值”，而是 operator-conditioned context policy：
$$
h_t=\operatorname{Ctx}(a_t,o_t,\mathcal H_t).
$$

第一版仍可使用共享 parent path 作为可比 baseline；是否应让 Explore 读取更弱的局部路径、机制摘要或已探索区域，需要新的单因素实验。

### 6.3 即时质量不能直接等同于有限续段价值

记对入口 child $x'$ 强制运行 $H$ 步的协议为 $\pi$，其有限续段价值为：
$$
V_H^{\pi}(x')=\max_{0\leq k\leq H}q(x_k)-q(\operatorname{parent}(x')).
$$

P3 观察到一部分 $q(x')<q(\operatorname{parent}(x'))$ 的 child 在五步内恢复，因此将跨边界的低即时质量直接映射为零后续机会会产生 false negative。但 $V_H^{\pi}$ 同时依赖任务、horizon、proposal kernel 和续段协议，它不是 child 的内在不变潜力。CVRP 的低 parent recovery 只表明该样本与五步协议下的固定投入可能低效，不足以把全部内部发展定义为浪费。

这导向一个分配原则：**development opportunity 不等于 guaranteed development budget**。Stage P 说明预算分配不应把即时回撤当作必然死刑，但没有告诉系统应为哪个 child 支付多少观测成本。这一问题仍必须由每步重新选择的 allocation policy 回答。

### 6.4 Hypothesis 表示与局部 Development Policy 必须分开识别

Hypothesis 至少包含三个可分问题：Explore boundary 如何划分 trajectory segment，哪些统计在 segment 内聚合，以及选中 segment 后如何选锚点。P3 只比较了第三个问题的两种强制续段，因此 hypothesis-level 未胜出不能推出 hypothesis abstraction 无价值。

H1/H3 与 H5 方向的变化可以产生一个新的探索性假设：早期在 segment 内回访多个锚点，出现清晰领先谱系后转向 child-chain commitment。当前样本量、嵌套 horizon 和协议差异都不支持将它写成已观察 dynamics；它只是后续 adaptive local policy 的预注册候选。

## 7. 对 V9.8 设计的直接影响

1. 保留 Refine / Explore 的搜索角色划分。它已改变静态机制切换、修改规模和即时质量分布。
2. 保留 parent path 作为 Refine 的默认生成条件。对 Explore 则只作为当前 baseline，不把其作用写成已确认机制；后续以 operator-specific context 作为独立提议层问题。
3. 不恢复 protected probe。P3 说明部分 child 需要额外机会，但 CVRP 的低 recovery 与任务差异也说明固定赠送深度可能低效；V9.8 用逐步宽限参与每次在线重选仍是更小的待验证机制。
4. 不把 P3 结果当作 $C$ 或 $M$ 的效果。P3 强制给定续段机会，没有比较有无 $C$ 时完整搜索实际选择了谁，也没有检验历史平均 gain 是否预测下一份计算的收益。
5. 不把特定 hypothesis-level 局部规则的负结果外推为 hypothesis 表示的负结果。Boundary、聚合统计和局部锚点 policy 需要分开识别。

## 8. 证据边界与后续识别

- 本文的准确定位是 **mechanism identification / behavioral validation**：识别指令、上下文和强制续段是否改变所测行为。Stage A 才是 **search-policy effectiveness / causal ablation**；1000-eval 完整批次是 **end-to-end search performance**，held-out 则评价最终程序的泛化。这四种证据不互相替代。
- P1/P2 是固定锚点重复观测，P3 是条件于有效、非 no-op 的 parent-path Explore child 的强制干预；两者都不是独立完整搜索重复。P3 recovery rate 只适用于该选入 cohort 与强制协议，不是全部 Explore attempt 的无条件概率。
- 静态宏簇是代码规则代理，不是真实算法 family；更高切换率只说明 proposal 的机制代理分布改变。
- P3 的 horizon 是同一生成前缀的嵌套读取，H1、H3、H5 不是独立样本。
- 本文没有回答 hypothesis boundary、$C$ 或 $M$ 是否提高 best-at-budget，也没有 held-out 结果。
- 下一层证据来自 Hypothesis-Uniform、Route-$Q+U$、Hypothesis-$Q+U$、$Q+U+C$、$Q+U+C+M$ 的 Stage A 对照；完整正式 V9.8 只能评价联合协议。
- “区域内早期回访，出现领先谱系后承诺”只是由 H1/H3/H5 方向变化产生的探索性假设。若继续检验，必须预注册切换信号、切换时点和总 horizon，不能在当前结果上事后择时。

## 9. 事实工件

- P1/P2 协议：`experiments/generation_probe/20260815_221500_v98_p1_p2/probe_config.json`；
- P1/P2 冻结分析：`experiments/generation_probe/20260815_221500_v98_p1_p2/analysis/summary.json`；
- P3 协议：`experiments/generation_probe/20260816_001100_v98_p3/probe_config.json`；
- P3 冻结分析：`experiments/generation_probe/20260816_001100_v98_p3/analysis/summary.json`；
- 预注册与入口说明：[TraceAAD V9.8 机制识别实验](../experiments/TraceAAD-V9.8机制识别实验.md)；
- 完整方法边界：[TraceAAD V9.8 完整机制](../methods/TraceAAD-v9.8完整机制设计.md)。
