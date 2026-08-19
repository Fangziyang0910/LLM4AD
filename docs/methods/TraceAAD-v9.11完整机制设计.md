# TraceAAD V9.11：轨迹体制切换与探索着陆

> 实现入口为 `llm4ad/method/traceaad_v9_11/`，统一 runner 使用 `--version v9_11`。V9.11 以 [V9.7](TraceAAD-v9.7完整机制设计.md) 的稳定搜索骨架为直接实现基座，以 [V9.9](TraceAAD-v9.9完整机制设计.md) 的“从哪里继续—怎样走到这里—这次怎么改”为研究语义，并吸收 [V9.10 机制诊断](../analysis/TraceAAD-V9.10机制诊断.md) 暴露的反馈稀释问题。

## 1. 核心主张

算法改进包含两种时间尺度：Refine 通常在当前算法方向内产生较快反馈；Explore 尝试改变核心决策原则，第一次实现常处于尚未稳定的着陆状态。若两者都被当成相互独立的一步生成，Explore 会因即时质量较低而迅速失去后续机会；若为 Explore 建立长期信用、后验或独立预算容器，又会引入比目标现象更复杂的控制系统。

V9.11 只引入一个主要机制：**由全局搜索停滞触发一次 Explore，并给有效 Explore child 紧接着的一次 Refine 着陆机会。**

完整节律为：

```text
搜索仍在产生严格全局突破
        -> 沿高质量轨迹继续 Refine

连续 H 个正式搜索响应没有严格全局突破
        -> 从当前应投资的锚点执行一次 Explore

Explore 形成有效 child
        -> 下一次响应固定从该 child 执行一次 Refine

着陆 Refine 完成
        -> 取消全部特殊待遇，重新进行全局选择
```

其中固定停滞窗口为

$$
H=8.

$$

`H=8` 是首版协议选择，不是科学常数。它使 Explore 在持续停滞时保持稀疏，同时与当前最多 8 条形成事件的历史视野处于同一短时间尺度。

V9.11 的直觉是：仍在改善的方向值得继续发展。它的反直觉是：Explore child 即使即时变差，也固定获得一次紧邻的 Refine；这一次预算表达的是“结构变化需要落地”，不表达该 child 已经具有较高长期价值。

## 2. 研究对象与设计原则

V9.11 保留 TraceAAD 的两个核心研究对象。

### 2.1 轨迹条件生成

$$
x_{t+1}
\sim
P(\cdot\mid x(a_t),h(a_t),o_t),

$$

其中 $a_t$ 是当前锚点，$x(a_t)$ 是当前完整程序，$h(a_t)$ 是与该锚点严格匹配的父代形成路径，$o_t\in\{R,E\}$ 分别表示 Refine 与 Explore。

轨迹的作用是改变下一步候选分布。模型每次只完成一个新决策：联合生成一条 `Idea` 和一份完整 `Code`。历史中的继续、修复、回退和换方向都是已经发生的事实或本次 Idea 的语义内容，不被实现为额外模型动作。

### 2.2 轨迹感知的计算分配

常规状态下，V9.11 沿用 V9.7 的路线到锚点两级质量—机会选择。Explore 着陆时，下一份预算临时覆盖常规选择，直接分配给刚形成的 Explore child：

$$
a_t
=
\begin{cases}
a_{\mathrm{landing}}, & \mathrm{landing\_anchor}\neq\varnothing,\\
\operatorname{Select}_{V9.7}(\mathcal H_t), & \mathrm{landing\_anchor}=\varnothing.
\end{cases}

$$

着陆覆盖只持续一个模型响应。完成后锚点重新与全部历史状态按同一规则竞争，不获得信用继承、额外分数或固定续段。

### 2.3 三条设计原则

1. **轨迹首先是生成条件。** 父代来时路进入提示；后代成功不回传为祖先信用。
2. **探索是一个“改方向—落地”的短事件。** Explore 与紧邻的一次 Refine 构成最小完整结构变化，不扩展成长期 rollout。
3. **分配保持可直接解释。** 常规选择只读当前质量和已获得机会；停滞只决定何时 Explore，不进入锚点评分。

## 3. 在线事实对象

### 3.1 程序

程序 $x$ 是 evaluator 实际执行的一份唯一代码。记原始 fitness 为 $f(x)$，任务方向为 $d\in\{+1,-1\}$：

$$
q(x)=d f(x),

$$

搜索内部统一为 $q$ 越大越好。程序保存规范化代码、代码哈希、真实 fitness、有向质量、发现顺序和评价成本。完全同分时只用代码长度与发现顺序确定最终 tie-break；复杂度不进入在线分配。

### 3.2 锚点

锚点表示程序在一条具体形成路径中的位置：

$$
a=\langle x(a),p(a),e(a),n(a),r(a)\rangle.

$$

$p(a)$ 是结构父锚点，$e(a)$ 是形成事件，$n(a)$ 是从该锚点获得的完整模型响应数，$r(a)$ 是根来源。相同代码沿不同路径到达时可以形成不同锚点；它们共享程序评价，但保留不同来时路与机会计数。

### 3.3 形成事件

每条有效父子边保存：

$$
e=\langle a_p,o,a_c,\mathrm{Idea},\mathrm{Change},\Delta q,\mathrm{Outcome},t\rangle.

$$

其中 `Change` 是父子 evaluator input 的确定性实际差异，`Outcome` 为 `improve / plateau / regress`。模型声明的 Idea 是语义标签，实际修改与真实评价是事实。invalid、no-op、重复和祖先返回也保存为从起始锚点发出的响应事实，但没有形成 child 时不创建形成边。

### 3.4 路线

路线只表示根来源：

$$
r(a)=r(p(a)).

$$

路线不是算法簇、hypothesis 或语义区域。它只为 V9.7 两级选择提供一个稳定的深度投资容器，不保存独立的长期信用。

### 3.5 体制状态

V9.11 只增加三个轻量运行状态：

- `last_progress_order`：最近一次严格刷新全局最好程序的正式搜索响应序号；
- `last_explore_order`：最近一次 Explore 响应序号；
- `landing_anchor`：等待一次 Refine 着陆的 Explore child，默认为空。

它们不参与程序或锚点质量计算。运行统计中的 Develop、Explore、Landing 次数和有效 Explore child 数只用于诊断，也不参与在线控制。

## 4. 初始化

初始化沿用 V9.7：

1. 独立生成 8 个有效且代码互异的根；生成一个根时不展示其他根或已有搜索历史。
2. 每个根恰好执行一次 Refine bootstrap；成功形成的 child 与根全部保留，invalid、no-op、重复边或祖先返回不重试。
3. 以有效 bootstrap 的一步有向质量变化绝对值估计共享尺度：

$$
s_0
=
\operatorname{median}
\left(
\left\{|q(x_c)-q(x_p)|\right\}
\right).

$$

持平的有效 child 以零变化进入中位数；invalid、no-op、重复和祖先返回不进入尺度估计。若所有有效变化均为零，则 $s_0=0$，常规分配自然退化为纯质量选择，不添加人工下界。

初始化结束时：

- `landing_anchor = null`；
- `last_progress_order = 0`；
- `last_explore_order = 0`；
- 正式搜索响应序号从 1 重新计数。

根生成与 bootstrap 中的真实 evaluator 调用全部计入统一的 1000 次评价预算。重新计数的只是用于体制切换的正式搜索响应序号，不是评价预算。

初始化的 8 个根是来源覆盖，不被解释为 8 个算法簇。

## 5. 常规路线—锚点选择

V9.11 不重新设计 V9.7 的常规分配器。

### 5.1 路线选择

记路线 $r$ 当前达到的最好有向质量为

$$
q_t^*(r)=\max_{a:r(a)=r}q(x(a)),

$$

该路线从初始化至当前已经获得的完整响应数为 $N_t(r)=\sum_{a:r(a)=r}n_t(a)$；因此每个根的 bootstrap 也计入对应路线和根锚点的机会计数。路线优先级为

$$
S_t^{route}(r)
=
q_t^*(r)
+
\frac{s_0}{\sqrt{N_t(r)+1}}.

$$

选择分数最高的路线。完全同分时依次选择响应更少、根创建更早、根 ID 更小的路线。

### 5.2 锚点选择

在选中路线内，对每个锚点计算

$$
S_t^{anchor}(a)
=
q(x(a))
+
\frac{s_0}{\sqrt{n_t(a)+1}}.

$$

选择分数最高的锚点。完全同分时依次选择响应更少、创建更早、锚点 ID 更小的锚点。

该分数只表达“已经达到多高”和“从这里获得过多少生成机会”。它不估计路线趋势、成熟度、平均增益、长期潜力或算法簇价值。

### 5.3 着陆覆盖

若 `landing_anchor` 非空，则本轮跳过路线和锚点选择，直接选择该锚点。此时仍记录“常规选择原本会选谁”，仅用于离线理解着陆覆盖改变了哪一次决策，不参与在线机制。

## 6. 停滞触发的生成意图

记当前已经完成的正式搜索响应数为 $m$，最近一次能够重置 Explore 时钟的序号为

$$
c_m
=
\max
\left(
\mathrm{last\_progress\_order},
\mathrm{last\_explore\_order}
\right).

$$

V9.11 在发出下一个请求前按以下唯一优先级决定意图：

$$
o_{m+1}
=
\begin{cases}
R, & \mathrm{landing\_anchor}\neq\varnothing,\\
E, & m-c_m\ge H \ \land\ B_{\mathrm{remain}}\ge 2,\\
R, & \text{otherwise}.
\end{cases}

$$

$B_{\mathrm{remain}}$ 是尚未消耗的真实 evaluator 调用数。Explore 只在至少余下两次评价时可触发，从而保证它若用一次评价形成新 child，仍有预算发出紧邻的 Landing 响应。这是停止边界，不是新的分配分数。

这里故意使用全局突破而不是锚点或路线的局部趋势。最终目标是全局最好程序，而稀疏锚点的局部无改善难以区分方向成熟与偶然波动。停滞时钟只读取“最近是否产生了新的全局最好”，不估计潜力。

因此存在三种可观察体制：


| 体制    | 触发条件                                         | 锚点          | 意图    |
| --------- | -------------------------------------------------- | --------------- | --------- |
| Develop | 搜索未达到停滞窗口                               | 常规两级选择  | Refine  |
| Explore | 连续`H=8` 个响应无严格全局突破，且没有待着陆锚点 | 常规两级选择  | Explore |
| Landing | 上一次 Explore 形成有效 child                    | Explore child | Refine  |

严格全局突破定义为新程序的 $q$ 严格大于此前全部程序；持平和同分择简不重置 `last_progress_order`。

每次 Explore 响应完成后，无论是否形成 child，都把 `last_explore_order` 更新为该响应的序号，避免生成失败导致连续 Explore。只有形成有效 child 时才设置 `landing_anchor`。传输失败未获得完整响应时，$m$ 不增加，原请求按恢复协议重试，不重新选择。

## 7. 生成上下文与意图契约

### 7.1 共同上下文

Refine 与 Explore 都读取：

1. 任务定义与不可违反的执行契约；
2. 当前程序的真实 fitness；
3. 当前完整代码；
4. 与当前锚点匹配的最近 8 条父链形成事件。

形成事件按真实发生顺序展示，每条只包含 `Intent + Idea + Compact Actual Change + Result`。从当前锚点发出的已有子代尝试、其他路线代码、全局 Idea Bank、算法簇标签和模型生成的全局总结不进入默认提示。

上下文超限时从最早形成事件开始删除；任务契约、当前 fitness、当前完整代码和本轮意图始终保留。

### 7.2 Refine

Refine 的职责是发展当前算法方向。提示要求模型：

- 延续当前核心决策原则；
- 利用来时路中已经形成的有效结构；
- 修正仍未稳定的实现或局部决策；
- 提交一个连贯的新 `Idea + Code`。

Landing 不增加第三种算子。Explore child 的形成事件已经进入 parent path；紧邻的 Refine 因而自然读取“刚刚改变了什么”，并负责把该变化稳定为可竞争的程序。

### 7.3 Explore

Explore 的职责是提出替代算法方向。提示要求模型：

- 改变当前算法的核心决策原则、搜索结构或主要信息利用方式；
- 避免只做参数微调、局部常数替换或同一机制的小修补；
- 保持任务接口与执行契约；
- 一次输出完整可执行的 `Idea + Code`。

Explore 仍读取当前来时路，因为它需要知道当前方向已经怎样形成；意图契约负责要求模型离开当前核心机制。V9.11 不增加 Explore 专用摘要器或第二次反思调用。

## 8. Explore 着陆

### 8.1 创建着陆资格

若 Explore 响应形成有效 child 锚点 $a_c$，则

$$
\mathrm{landing\_anchor}\leftarrow a_c.

$$

Explore child 是否改善父代、是否刷新全局最好、修改幅度多大，都不影响这一资格。资格来自生成意图这一事实，而不是来自语义分类器或质量预测器。

### 8.2 消耗着陆资格

下一次响应固定满足：

$$
a_{t+1}=a_c,
\qquad
o_{t+1}=R.

$$

该 Refine 响应完成后立即执行

$$
\mathrm{landing\_anchor}\leftarrow\varnothing.

$$

即使 Refine invalid、no-op、重复或再次退步，也不重试、不延长保护。若它形成新 child，新 child 作为普通锚点进入森林；下一轮恢复全局常规选择。

### 8.3 着陆的语义边界

一次着陆预算只表达三个判断：

1. Explore 的目标是结构性改变，其第一次实现可能尚未稳定；
2. 没有紧邻发展机会时，proposal 与 allocation 会共同造成 Explore child 早夭；
3. 一次机会足以表达容忍，继续投入仍应由常规竞争决定。

着陆不被解释为 rollout、局部预算池、后验成功样本或长期 continuation value。

## 9. 评价、去重与事实更新

每个模型响应完成后：

1. 起始锚点的 $n(a)$ 加一，所属路线的 $N(r)$ 加一；invalid、no-op 和重复响应同样计入生成机会。
2. 提取完整代码并按任务协议构造 evaluator input。
3. 对全局新代码启动真实 evaluator；已有代码复用缓存评价。
4. 新代码得到有限 fitness 后创建程序和 child 锚点；已见代码若形成新的父子关系，可复用程序并创建新的历史锚点。
5. 记录 Idea、实际修改、有效性、评价结果、父子关系和响应顺序。
6. 若新程序严格刷新全局最好，则更新 `last_progress_order`。
7. 若本轮是 Explore，则更新 `last_explore_order`，并按是否形成 child 设置着陆资格。
8. 若本轮是 Landing，则无条件清空着陆资格。

传输错误未获得完整模型响应时不增加机会计数；解析失败获得了完整响应，因而增加机会计数但不消耗 evaluator 预算。

## 10. 完整原子循环

```text
Initialize 8 code-unique roots.
Refine each root once and estimate the shared scale s0.
Keep every valid root and bootstrap child.

Set last_progress_order = 0.
Set last_explore_order = 0.
Set landing_anchor = null.
Set completed_search_responses = 0.

While real evaluator budget remains:
    If landing_anchor exists:
        Select landing_anchor.
        Set intent = Refine.
    Else:
        Select one route by q_best + s0 / sqrt(N + 1).
        Select one anchor in that route by q + s0 / sqrt(n + 1).
        If completed_search_responses
           - max(last_progress_order, last_explore_order) >= 8
           and at least 2 real evaluator calls remain:
            Set intent = Explore.
        Else:
            Set intent = Refine.

    Build Current Code + Parent Improvement Path.
    Generate one Idea + Code response.
    Parse, evaluate or reuse, and record all facts.
    Increment completed_search_responses and set t to its new value.

    If a strict global best is formed:
        Set last_progress_order = t.

    If intent is Explore:
        Set last_explore_order = t.
        If a valid child anchor is formed:
            Set landing_anchor = that child.

    Else if this was a landing response:
        Clear landing_anchor.

Return the globally best unique program by the true objective.
```

每次模型响应仍然只产生一个候选并立即更新事实。Explore 与 Landing 在叙事上构成两步结构事件，在执行上仍是两个独立、可中断、可记录的原子决策。

## 11. 主动删除的机制

V9.11 明确不包含以下对象：

- V9.8 的 hypothesis、边界信用、发展均值和独立 hypothesis 预算；
- V9.9 的中秩 `Q`、算子欠尝试 `U_R/U_E`、回撤宽限 `C_R`、softmax 和几何秩概率；
- V9.10 的锚点—意图联合臂、Beta 后验、Thompson 抽样、近期折扣、父链伪计数、用于延迟信用结算的 pending action、child-depth 结算和 response-age 结算；
- 趋势、动量、成熟度、路线推进率、祖先信用、subtree max-backup 和 learned critic；
- 在线语义聚类、embedding、算法簇 judge、全局 Idea Bank 和模型生成的长期总结；
- 固定三步 rollout、每个 Explore 的多步预算承诺和额外反思模型调用。

保留这些历史事实用于离线分析，不让它们进入在线控制。

## 12. 预期搜索行为

### 12.1 有进展时形成深轨迹

只要严格全局最好在 8 个响应内持续刷新，搜索保持 Develop 体制，所有普通生成均为 Refine。V9.7 的质量—机会选择继续决定在哪条路线和哪个锚点发展。V9.11 因而允许高产方向自然形成长的非单调改进链。

### 12.2 停滞时稀疏地改变方向

持续停滞时，首次 Explore 在已完成 8 个无突破响应后触发；之后两次 Explore 之间至少完成 8 个其他响应。因此在无突破的稳态中，Explore 约占正式搜索响应的 $1/9$，而不是固定 30%。搜索越有进展，Explore 越少；搜索越停滞，Explore 越接近这一上界。

### 12.3 Explore 获得最小兑现机会

每个有效 Explore child 固定获得一次紧邻 Refine。低即时质量不会在它完成最小落地前使其失去机会；一次 Refine 后也不会因为“曾经被 Explore 创建”而长期占用预算。

### 12.4 失败方向自动退出

Explore 或 Landing 没有产生竞争性程序时，常规选择会自然回到已有高质量锚点。系统无需失败惩罚、后验降权或显式淘汰。

## 13. 直觉与反直觉

V9.11 的论文叙事围绕一组直接矛盾展开：当前质量适合判断已经完成的算法，结构性探索却可能产生尚未完成的算法。


| 行为                   | 直觉解释                   | 反直觉含义                                             |
| ------------------------ | ---------------------------- | -------------------------------------------------------- |
| 改善期持续 Refine      | 有产出的方向继续发展       | 探索比例可以随成功下降，而不是始终保持固定多样性       |
| 停滞触发 Explore       | 当前方向暂时没有兑现新收益 | Explore 由搜索状态触发，不由固定概率或算子 bandit 触发 |
| Explore child 固定着陆 | 大改动需要一次稳定实现     | 一个更差的 child 可以覆盖全局选择并获得下一份预算      |
| 着陆只持续一次         | 容忍结构变化的短期回撤     | 允许退步不等于奖励退步，也不等于相信长期潜力           |
| 着陆后重新竞争         | evaluator 重新接管选择     | stepping stone 身份不会永久附着在节点上                |

## 14. 与相近工作的关系

BaSE 的核心启发是深度与广度的价值取决于轨迹间可利用异质性；V9.11 不把不断增长的锚点当作 bandit 臂，也不估计跨轨迹后验。它把分配问题收缩为一个更贴近 TraceAAD 生成语义的事件：何时暂时离开局部发展，以及新方向是否获得一次落地。

A2DEPT 在全局停滞时 re-anneal，并通过 Boltzmann 选择保留 stepping stone。V9.11 同样承认停滞与结构变化的关系，但不引入温度、模拟退火接受、Boltzmann 父代池和算子权重继承。TraceAAD 的差异点仍是：真实父代改进来时路直接进入下一次生成，Explore child 的形成事件直接条件化紧邻 Refine。

因此，V9.11 借鉴的是两条结构认识：停滞应改变搜索体制，结构性改变存在延迟效用；具体机制保持为 TraceAAD 自身的轨迹条件生成与一次着陆分配。

## 15. 设计假设与失败边界

V9.11 包含三个设计假设，当前均不写成实验结论：

1. 严格全局突破的短期缺席足以作为打开 Explore 的粗粒度信号。
2. 一次紧邻 Refine 能提高部分 Explore 提议转化为可竞争算法的概率。
3. V9.7 的常规两级分配能够在不增加新信用的情况下承接着陆后的方向。

可能的失败方式包括：

- `H=8` 使某些任务 Explore 过频或过稀；
- 全局停滞不能区分“当前路线成熟”和“生成模型暂时波动”；
- Explore 仍受 parent path 锚定，无法产生真正的结构变化；
- 一次 Refine 不足以完成大型改写，或已经足以浪费过多预算；
- V9.7 路线层进入吸收态，使 Explore 始终从同一来源发生；
- 常规质量—机会选择在着陆后仍过快放弃尚未完成的新方向。

这些是完整版本可能呈现的行为边界。首轮设计不为每个边界预先增加修补机制。

## 16. 实现不变量

实现必须满足：

1. 初始化为 8 个代码互异根，每个根一次 Refine bootstrap。
2. 每个普通循环恰好选择一个锚点、一个意图并请求一个 `Idea + Code`。
3. Refine 是默认意图；只有停滞窗口到达且至少余下两次真实评价时产生 Explore。
4. 只有形成有效 child 的 Explore 创建着陆资格。
5. 每个 Explore child 最多获得一次着陆 Refine；着陆完成后无条件清除资格。
6. Landing 覆盖路线—锚点选择，但不修改任何在线分数。
7. 严格全局突破与 Explore 响应共同重置 Explore 冷却；plateau 不重置突破时钟。
8. 父代来时路最多展示 8 条形成事件，不展示已有子代尝试。
9. 形成历史、invalid、no-op、重复与评价失败均按事实落盘。
10. 程序事实持久化代码哈希、真实 fitness、有向质量、发现顺序和 evaluator 耗时。
11. 最终程序只按真实任务目标选择，完全同分时择短、再择早。
12. 正式预算为 1000 次真实 evaluator 调用，包含根与 bootstrap；额外响应数、token 和墙钟成本单独记录。
13. 默认协议中不存在后验、信用回传、语义簇或隐藏的多步 rollout。

## 17. 首轮运行协议

V9.11 先作为完整联合版本运行，不同步展开消融矩阵：

1. 先在一个任务上完成小预算 smoke，确认 Develop、Explore、Landing 三种体制均可进入，checkpoint 可恢复，着陆资格不会重复消费。
2. smoke 通过后，按统一模型和每次 1000 次真实评价预算运行四任务三重复。
3. 三重复搜索完成后统一进行 held-out 评价，再判断 V9.11 是否值得进入正式结果页。
4. 首轮只附带记录 Explore 周期数、有效 Explore child 数、Landing 完成数、Landing 后的即时恢复和全局突破来源；这些量用于理解完整机制，不作为调参目标。
5. 在完整版本显示竞争力之前，不启动 `H`、着陆长度或常规分配器的并行消融。

V9.11 的首要评价对象是完整搜索行为：它能否保留 V9.7 的深度发展能力，同时让少量结构性 Explore 真正经过一次轨迹条件化的落地。单组件归因属于该版本形成稳定结果之后的工作。

## 18. 两句话方法说明

LLM 自动算法设计通常根据当前程序和分数决定下一步，因而会忽略算法在思想引入、暂时退步和修复中形成的来时路。TraceAAD V9.11 沿仍在产生突破的轨迹持续精炼，在停滞时触发结构性探索，并让新方向获得一次由其真实形成历史条件化的着陆精炼。
