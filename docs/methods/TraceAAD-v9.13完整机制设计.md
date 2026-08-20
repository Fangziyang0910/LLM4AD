# TraceAAD V9.13：代理区域前沿条件化探索（修订 r3）

> V9.13 以 [V9.7](TraceAAD-v9.7完整机制设计.md) 为直接基线，研究搜索全局已经形成的区域前沿能否改善 Explore 的下一步提议。r2 依据第一轮 Stage P（[机制识别](../analysis/TraceAAD-V9.13-StageP机制识别.md)）把完整前沿表改为地板表（其它区域只给质量）；第二轮 Stage P 修复了有效率（最大降幅 3.7pp）但均值口径的单步质量只余 2/4 任务。r3 解混 r2 捆绑的三个变量：**保留全部区域的机制标签，只改语义框架（地板措辞、本区域置顶、全局最好一行）**——检验呈现方式能否把标签的价值与风险分开。设计依据还包括[研究认识](../knowledge/研究认识.md)、[V9.7 机制诊断](../analysis/TraceAAD-V9.7机制诊断.md)、[V9.7 机制区域重访分析](../analysis/TraceAAD-V9.7机制区域重访分析.md)、[L2 预算分配](../research/L2-预算分配.md)与[L3 单步生成](../research/L3-单步生成.md)。

## 0. 修订依据

第一轮 Stage P（1080 trial，五条件）给出四条可复用事实：

1. 前沿表使四任务 Explore 单步质量均值全部提高（+0.65s 至 +1.47s）、亚前沿重入全部下降、archive duplicate 零上升、next_selection 3/4 任务改善——信息的生成杠杆真实。
2. 唯一失败项是 CVRP 有效率 −7.4pp（门槛 5pp）；失效全部为运行期失败而非解析失败，且修改行数未增大——代价来自更冒险的结构重写，不是更大的改写。
3. 失效与"跨区域攀比"通道一致：表内其它区域的机制标签 + 更高质量诱发朝强区域的结构性复写，在容量约束最紧的 CVRP 上最易失效；完整参考代码（FC）在 OP 上产生 5.6% 参考复制，增量方向混合，该候选关闭。
4. 被选中的锚点几乎全部贴近本区域前沿（V9.7 锚点分数必然选高质量锚点），因此任何"锚点低于前沿才交付"的条件化都会退化为永远交付；修订只能改信息内容。

r2 的改动由此确定：**本区域前沿完整展示（机制标签 + 质量），承载反劣质重建；其它区域只给质量水平，不给机制标签与代码**，移除结构模仿的线索。FC 从候选集中删除。

第二轮 Stage P（864 trial，四条件）的结果：有效率门槛通过（最大降幅 3.7pp，v1 的 CVRP −7.4pp 消失），但均值口径的单步质量仅 2/4 任务提高（TSP +2.50s、OP +0.31s；CVRP −0.52s、OBP −3.06s），后者由重尾离群主导（OBP 单点 −109.6s 贡献约 2.2s 均值差，中位数两组持平；CVRP 中位数地板表略优）。合并两轮：**跨区域机制标签同时承载质量收益（v1 的 CVRP/OBP 增益）与有效率风险（v1 的 CVRP 运行期失败）；移除标签把两者一起削掉。** r2 同时修改了措辞、排序与标签三个变量，无法归因。

r3 的改动只做一件事：**恢复全部区域的机制标签，保留 r2 的地板语义框架**（反劣质重建措辞、本区域置顶、其它区域按质量降序、全局最好一行、不展示程序数量与代码）。检验"标签的价值是否依赖成就式呈现"。六项门槛与阈值在新数据生成前原样冻结；若 r3 仍不通过，V9.13 以三轮识别证据终止。

## 1. 核心判断

V9.7 当前最清楚的损失发生在 Explore 的提议质量。Explore 经常改变静态机制宏簇，但其换簇 child 中 `97.9%–99.8%` 落入已经访问的宏簇，只有 `5.2%–7.4%` 出生即达到目的宏簇当时的前沿；无新机制标签的 Explore child 占全部搜索响应的 `23.4%–27.8%`。模型知道当前程序怎样形成，却不知道全局已经实现过哪些机制区域以及这些区域达到过多高。

路线分配目前没有同等强度的新设计依据。路线是根来源，尚不是算法族；路线乐观项在 V9.7 中只改变 `1.07%` 的路线选择；被放弃路线又缺少可比较的 continuation observations。V9.13 因此冻结预算分配，只研究一个问题：

> **给定同一锚点、同一父代来时路和同一 Explore 指令，加入以"反劣质重建"形式构造的代理区域前沿信息，能否在不过度损失有效率的前提下减少低质量重建并提高候选进入后续竞争的能力？**

V9.13 采用两阶段协议：Stage P 在固定决策快照上比较候选上下文，选择满足预设条件的最小处理；Stage A 从共同 V9.7 前缀分叉，对冻结后的处理做完整搜索与 held-out 验证。Stage P 负责设计选择，Stage A 负责评价冻结版本。

## 2. 科学主张与适用范围

### 2.1 研究对象

对当前锚点 $a_t$，V9.7 Explore 的生成分布为

$$
x_{t+1}
\sim
P_E(\cdot\mid x(a_t),h_t,o_E),
$$

其中 $h_t$ 是与锚点匹配的父代形成路径，$o_E$ 是固定 Explore 指令。修订后的 FP 只追加搜索全局视图 $g_t$：

$$
x_{t+1}
\sim
P_E(\cdot\mid x(a_t),h_t,o_E,g_t).
$$

V9.13 不修改 $o_E$。Stage P 比较的是信息条件，不把新指令或参考程序混入处理。

### 2.2 代理边界

V9.13 的 `mechanism_tags` 与 `macro_family` 是根据实际代码构造、在实验前冻结的任务内静态代理。它们承担两项职责：压缩已评价程序形成的搜索全局视图；支持可审计的过程诊断。它们不是语义真值，也不是 TraceAAD 已经学会的通用算法簇表示。第一轮 Stage P 前的盲评审计记录了已知误差（[代理标签盲评审计](../analysis/TraceAAD-V9.13代理标签盲评审计.md)，最主要：TSP 字面 `two_opt` 拼写漏标影响 25.1% 程序）；条件间对比对该误差配对不变。

V9.13 的四任务结果最多支持：**在这四个任务及冻结代理定义下，以地板形式提供的区域前沿信息是否具有生成和搜索价值。** 同任务 held-out 评价检验最终程序对新规模和实例的泛化，不检验代理定义向新任务的迁移。

## 3. 保留的 V9.7 主干

以下机制完全保留：

1. 初始化 $K=8$ 个代码互异且有效的根，每个根进行一次 Refine bootstrap；
2. 从有效 bootstrap 的一步有向质量变化估计固定尺度 $s$；
3. 先按路线分数选择路线，再按锚点分数选择形成状态；
4. 固定 $P(\mathrm{Refine})=0.7$、$P(\mathrm{Explore})=0.3$；
5. Refine 与 Explore 均读取当前完整程序和至多 8 条匹配的父代形成事件；
6. 每次只生成一个 `Idea + Code`，立即评价并重新选择；
7. 有效 improvement、plateau 与 regression state 均作为形成事实保留；
8. 1000 次真实评价后返回全局最好唯一程序。

路线与锚点分数仍为

$$
S_t^{route}(r)=q_t^*(r)+\frac{s}{\sqrt{N_t(r)+1}},
\qquad
S_t^{anchor}(a)=q(a)+\frac{s}{\sqrt{n_t(a)+1}}.
$$

V9.13 不加入路线票据、长期信用、算子后验、固定 rollout 或多后代批次。处理差异只进入 Explore 的上下文。

## 4. 代理区域前沿视图

### 4.1 时间边界

一次生成决策只能使用该决策开始前已经完成真实评价的唯一程序。对程序 $p$，定义其真实评价次序为 $e(p)$；令 $b_t$ 为第 $t$ 次决策开始前已完成的真实 evaluator 调用数。可用程序集合为

$$
\mathcal P_t=\{p:e(p)\le b_t\}.
$$

模型响应序号、程序创建序号和 evaluator 调用数分别记录，不能相互替代。Stage P 必须重放真实搜索决策发生前的状态；锚点创建时的状态不能代替该锚点后来被选择时的状态，未来程序也不能回填到历史快照。

### 4.2 固定代理

对每个已评价程序的实际 evaluator input code，使用冻结的任务内规则得到：

- `mechanism_tags(p)`：去除注释与字符串后，从代码 token 识别的机制标签；
- $F(p)$：由标签归并得到的 `macro_family` 代理区域。

标签不读取声明 Idea，不使用后续结果训练，不在实验过程中修改。r2 沿用 r1 冻结的词表与盲评审计；r2 的表格构造不改变标签本身。

### 4.3 区域前沿与地板表

对决策前已经访问的代理区域 $f$，定义

$$
q_t^*(f)=\max_{p\in\mathcal P_t:F(p)=f}q(p),
$$

并令 $p_t^*(f)$ 为对应前沿程序。完全同分时偏好代码更短、真实评价更早的程序。

FP 的上下文是一张**地板表**，语义是"已达到的水平，重入低于它的区域是浪费"，不是"待追赶的成就"。构造规则：

1. **本区域块置顶且完整**：当前锚点所在区域 $F(a_t)$ 的前沿程序以机制标签 + 有向质量展示，并显式标注为当前算法所在区域；
2. **其它区域给标签与水平**：其余已访问区域按 $q_t^*(f)$ 从高到低列出区域序号、前沿程序机制标签与有向质量，不展示程序代码、程序数量、访问次数或声明 Idea；
3. 表尾给出全局最好有向质量一行；
4. 展示冻结代理能够区分的全部已访问区域，不静默截断。

```text
[Searched Proxy Regions]
Earlier in this search the following proxy mechanism regions were already
implemented and evaluated. A candidate that merely rebuilds a region below
its recorded level wastes budget.

[Current Algorithm's Region]
Observed tags of frontier program: ...
Directed quality: ...                  # higher is better

[Other Searched Regions]
Region 2 observed tags of frontier program: ...
Region 2 directed quality: ...
Region 3 observed tags of frontier program: ...
Region 3 directed quality: ...

Global best directed quality across all regions: ...
```

### 4.4 启用条件

本区域块的存在只要求锚点程序已被评价，因此在 200 次真实评价的启用边界后 FP 恒可构造，不再存在"无替代区域"的回退分支。若决策前仅访问过一个区域，其它区域清单为空行，FP 仍交付本区域块。

## 5. 候选 Explore 上下文

Stage P 比较两个 Explore 上下文：

| 名称 | 内容 | 识别对象 |
| --- | --- | --- |
| `PP` | 当前代码 + parent path | V9.7 基线 |
| `FP` | `PP` + 代理区域地板表 | 地板表信息的作用 |

两种条件共享任务说明、当前 fitness、当前完整代码、parent path、V9.7 Explore 指令、输出契约、temperature、最大输出 token 和块内采样 seed。FP 的新增内容放在独立的全局事实区，不伪装成当前锚点的形成历史。

V9.7 Explore 指令保持不变：

> Seek a materially different way to improve the current algorithm. Do not merely tune parameters or make a small local modification. You may replace or substantially restructure an important part of the current design.

模型仍只输出一个简短 Idea 与一份完整可执行 Code。系统不要求目标区域、计划、置信度或结构化机制声明。

任务说明、当前代码、输出预算和整张地板表是处理的硬组成。上下文不足时，只按 V9.7 规则从最早的 parent-path 事件开始删除。仍无法容纳完整处理时，本次运行报错；正式协议不降级为另一处理。Stage P 和长跑 smoke 必须事先确认所有计划条件可以完整构造。

## 6. 启用边界与完整循环

V9.13 在 200 次真实评价完成后启用冻结的地板表处理，前 200 次评价完全执行 V9.7。200 是由开发数据确定的四任务协议边界，作用是把处理集中于亚前沿重建阶段；它不是在线饱和估计，也不声称适用于新任务。

```text
Input: task, evaluator, LLM, real evaluator budget B = 1000

Run the V9.7 protocol through exactly 200 real evaluator calls.

While real evaluator budget remains:
    Select route by q_best(route) + s / sqrt(N(route) + 1).
    Select anchor in that route by q(anchor) + s / sqrt(n(anchor) + 1).
    Draw Refine with probability 0.7, otherwise Explore.

    If Refine:
        Build the unchanged V9.7 Refine prompt.
    Else:
        Snapshot all programs evaluated before this decision.
        Build the frozen Stage-P-selected context for Explore.

    Generate one Idea + one complete program.
    Increment the selected anchor's response count.
    Evaluate a new program or reuse an existing cached result.
    Record prompt treatment, snapshot, actual change, outcome and costs.
    Create a child anchor for a valid new relation.
    Update the global view only after the response is complete.
    Reselect immediately.

Return the globally best unique program by the true task objective.
```

第 200 次评价前后不重置森林、尺度、访问计数或随机数日程。地板表处理不提供额外评价、保护期或后续 Refine；候选必须通过 V9.7 的普通竞争获得后续发展。

## 7. Stage P：固定真实决策快照实验

### 7.1 实验单位与抽样

Stage P 从 V9.7 正式搜索的真实决策日志中抽取 Explore 决策快照。每个快照包含当时实际选中的路线与锚点、决策前 evaluator 调用数、当前森林、parent path、已评价程序集合、区域前沿和 V9.7 选择分数。

快照满足：

1. 决策开始前已经完成至少 200 次真实评价；
2. 当前锚点具有可重放的有效 parent path；
3. 决策前至少存在两个代理区域；
4. 处理上下文能够完整放入上下文窗口；
5. 完全相同的决策状态——同一来源运行、锚点、parent path、评价时点与全局前沿——只保留一次。

每任务从 3 个来源运行的三个真实评价区间中抽样：`[200,466]`、`[467,733]`、`[734,999]`。每个来源运行 × 区间选择两个快照，一个来自该单元所选锚点质量的下半区，一个来自上半区；单元内使用冻结 seed 随机抽取。于是每任务得到 18 个快照，四任务共 72 个。任一单元不足两个合格快照时，Stage P 准备阶段报错并在生成数据前修订抽样协议。

处理以决策快照为单位重复施加。每个快照—条件生成 3 次响应；三次响应是同一实验单位上的技术重复，先在快照内聚合。快照嵌套于来源运行，来源运行是任务内的独立搜索重复。条件顺序在任务 × 来源运行 × 评价区间内平衡，块内共享采样 seed，并在同一服务块内完成。完整 Stage P 共 $72\times4\times3=864$ 次响应。

模型传输失败按冻结的次数重试同一 trial 与 seed；重试耗尽后保留为缺失 trial，不以新锚点或新 seed 替换。解析失败、no-op、重复、运行失败和 evaluator 失败属于已经发生的响应。全部计划 trial 完成或明确缺失后才执行正式分析。

### 7.2 条件与对比

四个条件为：

| 条件 | Intent | 上下文 |
| --- | --- | --- |
| `pp_refine` | Refine | PP |
| `pp_explore` | Explore | PP |
| `fp_refine` | Refine | FP |
| `fp_explore` | Explore | FP |

设计对比为：

1. `fp_explore - pp_explore`：地板表的作用（主对比，承担候选选择）；
2. `fp_refine - pp_refine`：地板表对 Refine 的影响，用于判断信息作用是否依赖 operator（只作机制解释）。

第一轮已完成的 `fc_explore` 增量对比支持关闭参考代码候选，r2 不再设置。r2 使用新生成数据；r1 的数据不与本协议合并。

### 7.3 结果变量

**质量与安全结果**不依赖代理区域，承担主要决策：

- `valid_rate`：候选代码可解析，并通过新评价或缓存复用得到有限有向质量的比例；
- `evaluable_novel_rate`：形成代码新颖且获得有效 evaluator 结果的比例；
- `conditional_delta_q_over_s`：有效候选的 $(q_{child}-q_{parent})/s$；
- `parent_improvement_rate`：严格超过当前父代的比例；
- `global_gap_over_s`：$(q_{child}-q_{global,t}^*)/s$；
- `archive_duplicate_rate`：候选与决策前任一已评价程序代码相同的比例；
- `no_op_rate`：候选与当前程序相同的比例；
- `next_selection_rate`：将本次响应按真实 V9.7 更新规则写回快照后，新 child 是否成为紧邻下一次选择的锚点。分母是全部响应；没有形成 child 的响应取 0，重复程序是否形成新锚点严格服从 V9.7 规则。

正式 Stage P 要求所有来源运行的 $s>0$。invalid、解析失败、运行失败、no-op 与历史重复分别报告，不从有效条件均值推断总体成功。

**代理机制结果**解释信息怎样改变 proposal：

- 当前区域、其他已访问区域与新代理区域的目的地分布；
- proxy-region switch rate；
- 已访问区域的连续 `frontier_gap_over_s`；
- `sub_frontier_response_rate`：在全部响应中，形成有效 code-novel candidate、落入已访问区域且低于目的区域前沿超过 $1s$ 的比例；
- 在落入已访问区域的有效 code-novel candidate 中，推进目的区域前沿的比例。

**成本结果**包括 prompt token、response token、LLM 调用、真实 evaluator 调用和墙钟时间。

### 7.4 分析单位与不确定性

每个快照—条件的 3 次响应先求均值或比例，再计算条件间的快照级配对差。任务内同时报告：18 个快照的配对均值、方向计数与区间；每个来源运行内 6 个快照的配对均值；按"来源运行 → 快照"重采样的层级 bootstrap 区间。来源运行只有 3 个，因此区间用于描述不确定性，不据此构造高精度显著性结论。四任务结果分别报告。

## 8. Stage P 的候选选择规则

候选选择规则在新的 Stage P 数据生成前冻结，选择完成后不再修改 V9.13 的处理。门槛与 r1 相同：资源受限下的开发决策规则，不是统计显著性界限。

FP 相对 PP 需要同时满足：

1. `conditional_delta_q_over_s` 的任务内均值在至少 3 个任务上提高；
2. 任一任务的均值下降不超过 `0.5`；
3. 任一任务的 `valid_rate` 下降不超过 5 个百分点；
4. 任一任务的 `archive_duplicate_rate` 上升不超过 5 个百分点；
5. `next_selection_rate` 在至少 3 个任务上改善；
6. `sub_frontier_response_rate` 在至少 3 个任务上不增加。

FP 满足时，FP 冻结为 V9.13 处理并进入 Stage A；不满足时，V9.13 停止在机制识别阶段，不启动正式搜索。所有条件和结果均完整报告。

## 9. Stage A：共同前缀的配对完整搜索

Stage A 使用并行、同期的 V9.7 行为控制，不以历史批次代替正式控制。

对每个任务和重复：

1. 按 V9.7 从头运行至恰好 200 次真实 evaluator 调用（V9.13 处理 pp，行为与 V9.7 逐位一致）；
2. 保存一个通过完整恢复测试的共同 checkpoint；
3. 从该 checkpoint 复制出控制分支和 V9.13 处理分支；
4. 两个分支使用相同的后续 intent 随机数、模型采样 seed、评价预算与恢复规则；
5. 控制分支继续 PP，处理分支只在 Explore 使用 Stage P 冻结的 FP；
6. 分叉后的森林、程序缓存、区域视图、日志与 checkpoint 完全隔离；
7. 两个分支分别运行到 1000 次真实 evaluator 调用；
8. 对两个分支的最终最好程序执行相同正式 held-out。

四任务各使用 3 个独立共同前缀，共 12 个配对实验单位。条件效应按同一前缀内的 `V9.13 - V9.7` 差值报告；三次重复给出实际均值差、逐重复方向和样本标准差。

Stage A 的主要终点是 1000 eval 的 search best 和正式 held-out。过程终点包括：地板表处理的实际激活与上下文完整率；Explore 的有效率、标准化父代变化、archive duplicate 和 next-selection；亚前沿重入、区域前沿推进和实际后续 Refine；最终最好程序的形成意图、深度与区域迁移；路线和锚点集中形状；evaluator 调用、LLM 调用、prompt/response token 与墙钟成本。

正式性能预算仍为每分支 1000 次真实评价。若 V9.13 因重复或无效响应需要更多 LLM 调用才能达到 1000 eval，这部分额外生成成本必须单独报告。另提供按共同 LLM 调用数截断的搜索曲线作为成本敏感性分析，不替代正式 evaluator-budget 结果。

## 10. 实现审计与失败可见性

正式 Stage P 和 Stage A 启动前必须通过以下审计：

1. 处理配置同时出现在运行配置、真实方法对象和提示审计中；
2. 快照使用真实 evaluator 次序，重放时不存在未来程序；
3. 地板表每一行的标签与质量来自该区域的前沿程序，不展示代码与数量；
4. 本区域标注与锚点所在区域一致；
5. PP 与 FP 除全局事实区外完全一致；
6. Refine 提示与 V9.7 字节级一致；
7. 200-eval checkpoint 两个分支恢复后拥有相同森林、计数、尺度与下一随机状态；
8. 分叉后两个分支的缓存、日志和全局视图相互隔离；
9. reference copy、archive duplicate、no-op、invalid、缓存复用和 evaluator 失败均能区分；
10. 所有正式上下文完整放入窗口，没有静默截断或处理降级；
11. checkpoint 恢复不会改变区域快照或 intent 日程。

任一条件失败时停止对应 smoke 或正式运行。

## 11. 证据层次与允许的结论

1. **代理审计**：说明静态规则如何标记代码及其误差，不证明真实算法族；
2. **机制运行**：说明 FP 内容实际进入 Explore 提示；
3. **固定快照提议**：说明给定真实决策状态时，处理怎样改变下一候选；
4. **完整搜索**：说明冻结处理是否改善 1000-eval 可达前沿；
5. **held-out 与成本**：说明最终程序在同任务新实例上的表现及生成代价。

Stage P 选择的候选只能评价 proposal；FP 被选择时可归因到地板表处理。Stage A 的联合结果评价冻结后的 V9.13 整体。

V9.13 不支持以下主张：静态代理等于真实算法簇；200 eval 是通用饱和时点；held-out 证明代理可迁移到新任务；减少重访必然改善最终质量；地板表单独造成完整搜索收益。r1 已关闭的主张不再进入 r2：完整参考代码作为 Explore 上下文的增量价值（OP 参考复制与混合增量）。

## 12. 预期失败方式

1. **降权不完全**：仅给水平的其它区域仍诱发攀比，有效率损失未消除；
2. **信息不足**：移除机制线索同时削弱反劣质重建作用，质量收益缩小到门槛以下；
3. **上下文挤占**：全局事实削弱当前代码与 parent path 的作用；
4. **分配不兑现**：候选质量提高但仍不足以进入 V9.7 的下一步竞争；
5. **任务异质**：OP、CVRP、TSP 与 OBP 对地板信息形成不同反应；
6. **成本转移**：评价预算相同，但提示和无效生成增加实际模型成本；
7. **代理过拟合**：四任务上有效的手工词表无法支持新任务。

诊断按层次报告，不能统一写成"全局知识有效"或"全局知识无效"。

## 13. 与相关工作及预算分配的关系

[MCTS-AHD](../references/LLM自动算法设计方法阅读笔记/22-MCTS-AHD.md)表明优秀程序可以作为不同生成操作的参考，[SMCEvolve](../references/LLM自动算法设计方法阅读笔记/48-SMCEvolve.md)将有无 inspiration 与修改范围拆成不同 proposal kernels。r1 的证据显示完整参考代码在本协议中引入参考锚定，r2 只检验地板表形式的前沿信息，不移植 UCT、价值回传、粒子重采样或 Thompson 调度。

预算分配继续作为独立研究线。新的分配设计需要先获得路线或其他投资单位的匹配 continuation observations，并固定 V9.13 的生成协议单独比较分配策略。V9.13 的区域代理不进入选择分数、候选奖励或最终排序。

## 14. 两句话方法说明

TraceAAD V9.13 保留 V9.7 的局部深精炼与即时反馈，在 200 次真实评价后只为 Explore 追加由既有评价轨迹构造的代理区域地板表：当前区域的前沿完整展示，其它区域只给水平。固定真实决策快照实验先检验该上下文，通过后从同一 V9.7 前缀分叉控制与处理，分别识别地板信息对 proposal、完整搜索和 held-out 的作用。
