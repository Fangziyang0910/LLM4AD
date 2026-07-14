# TraceAAD：基于算法改进轨迹的自动算法设计方法

本文描述当前 `llm4ad.method.traceaad` 实现的 TraceAAD。TraceAAD 将一条算法改进轨迹作为搜索单位，而不是只把单个程序或树节点作为搜索个体。每条轨迹记录从初始程序到当前程序的修改动作、算子、机制标签、适应度变化和结果类型，并据此完成轨迹选择、算子分配、上下文构造、轨迹生存和历史经验蒸馏。

## 1. 问题定义与总体框架

给定一个待设计的算法任务、一个可执行的程序模板、一个大语言模型和一个程序评估器，TraceAAD 的目标是在有限评估预算内搜索适应度最高的有效程序。设候选程序为 $p$，其标量适应度为 $f(p)$；最大化任务直接比较 $f$，最小化任务将比较方向反转。

TraceAAD 维护三类互补记忆：

1. **程序记忆**：保存所有已生成程序及其评估结果和父子关系。
2. **轨迹记忆**：保存当前仍值得投入预算的有界改进路径。
3. **模式记忆**：保存从历史改进步骤中统计得到的机制模式、经验教训和反模式。

搜索过程包含一个进化主回路，以及周期性的蒸馏、反思、岛间迁移和轨迹生存操作：

    轨迹选择
        -> 算子选择
        -> 动作生成与程序生成
        -> 程序评估
        -> 父子边与轨迹更新
        -> 新颖性门控
        -> 算子信用和记忆更新
        -> 周期性 survival / distill / reflect / migration

与只保留当前高分程序的搜索方法相比，TraceAAD 保留了“某个方向如何变好、何时退步以及从哪里重新分叉”等过程信息。

## 2. 搜索状态表示

### 2.1 程序推导图

每个候选程序对应推导图中的一个节点：

$$
p_i = (c_i, z_i, f_i, r_i, \kappa_i, m_i),
$$

其中 $c_i$ 是程序代码，$z_i$ 是程序思想，$f_i$ 是标量适应度，$r_i$ 是运行时间，$\kappa_i$ 是代码复杂度，$m_i$ 是机制标签。节点还可以保存可选的 robustness、confidence 和 per-instance fitness vector。

基于父程序生成子程序时，推导图增加一条有向边：

$$
e_i=(p_u,p_v,a_i,o_i,m_i,\Delta_i,y_i,g_i),
$$

其中 $a_i$ 是自然语言修改动作，$o_i$ 是搜索算子，$m_i$ 是机制标签，$\Delta_i$ 是父子程序之间的有向适应度变化，$y_i$ 是结果类型，$g_i$ 是可选的跨实例迁移信号。

结果类型由有向变化和阈值决定：

$$
y_i \in \{\text{improve},\text{plateau},\text{regress},\text{unknown}\}.
$$

推导图保存搜索过程中发生过的事实；后续的轨迹和模式记忆都从这些节点与边派生。

### 2.2 有界轨迹

轨迹是推导图中的一条有序路径：

$$
\tau=(p_0,e_1,p_1,\ldots,e_L,p_L).
$$

轨迹记录节点序列、边序列、起始节点、终点节点、所属 island、访问次数、状态以及当前价值。轨迹长度有上限，默认最大长度为 8。当从内部前缀重新分支后超过长度上限时，轨迹采用滑动窗口保留最近的有效节点和边。

轨迹有两种基本扩展方式：

- **endpoint extension**：从当前终点生成子程序并追加到轨迹末端；
- **prefix branching**：从轨迹内部的 base node 生成子程序，形成一条新的分支轨迹。

历史轨迹不会因为当前终点较弱而立刻从推导图中删除。轨迹可以被归档，但程序节点和推导边仍保留在程序记忆中。

### 2.3 模式记忆

模式记忆中的条目属于三种类型：

- `mechanism`：某机制在历史图边上的改进统计；
- `lesson`：近期高质量与低质量程序对比得到的经验；
- `anti_pattern`：在特定算子或机制上下文中反复失败的方向。

每条模式记录文本、机制标签、支持节点或边、统计分数、置信度和更新时间。模式记忆还维护 `operator × mechanism` 条件下的尝试次数、成功次数、失败 streak 和 cooldown，用于算子选择与新颖性探索。

## 3. 轨迹价值与选择

### 3.1 终点质量

TraceAAD 只使用当前活跃轨迹的唯一终点计算质量边界，避免归档程序、重复轨迹或异常 outlier 改变当前搜索池的尺度。默认使用 10% 和 90% 分位数裁剪：

$$
Q(\tau)=
\operatorname{clip}\left(
\frac{f(p_L)-f_{\min}}{f_{\max}-f_{\min}},0,1
\right).
$$

对于最小化任务，适应度方向在归一化前反转。当活跃终点只有一个或所有终点适应度相同，质量使用退化边界处理。

### 3.2 路径潜力

令轨迹中归一化后的节点质量为 $q_0,\ldots,q_L$，逐步变化为：

$$
r_i=q_i-q_{i-1}.
$$

越接近当前终点的变化具有越高权重。设折扣系数为 $\gamma$，归一化权重为 $w_i$，则折扣路径回报为：

$$
R(\tau)=
\frac{\sum_i w_i r_i}{\sum_i w_i}
+\lambda_{\mathrm{pos}}\frac{1}{L}\sum_i \mathbb{I}(r_i>0)
-\lambda_{\mathrm{down}}
\frac{\sum_i w_i\max(-r_i,0)}{\sum_i w_i}.
$$

为避免“低质量终点但单次大幅恢复”的路径主导选择，只有终点质量超过 `potential_quality_floor` 时，路径潜力才按质量门控后的值计入：

$$
P(\tau)=
\begin{cases}
0,&Q(\tau)\le q_{\min},\\
R(\tau)\dfrac{Q(\tau)-q_{\min}}{1-q_{\min}},&Q(\tau)>q_{\min}.
\end{cases}
$$

默认 $\gamma=0.8$，正向步骤权重为 $0.25$，回撤惩罚权重为 $0.5$，$q_{\min}=0.5$。

### 3.3 多样性与新颖性

TraceAAD 使用三种不依赖外部 embedding 的相似度：

1. **程序相似度**：去除注释和格式差异后，计算代码 token 集合的 Jaccard 相似度；
2. **机制相似度**：计算轨迹中机制标签集合的 Jaccard 相似度；
3. **轨迹行为相似度**：计算 `(operator, outcome)` 集合的 Jaccard 相似度。

两个轨迹的组合相似度为：

$$
\operatorname{sim}(\tau_a,\tau_b)
=
w_c\operatorname{sim}_{code}
+w_m\operatorname{sim}_{mech}
+w_t\operatorname{sim}_{traj},
$$

默认权重为 $(w_c,w_m,w_t)=(0.4,0.4,0.2)$。对目标轨迹 $\tau$，其边际多样性和新颖性分别为：

$$
D(\tau)=1-\operatorname{mean}_{\tau'\ne\tau}\operatorname{sim}(\tau,\tau'),
$$

$$
N(\tau)=1-\max_{\tau'\ne\tau}\operatorname{sim}(\tau,\tau').
$$

### 3.4 可选的泛化价值

当评估器显式提供 parent/child 的 per-instance fitness vector 或 robustness 证据，并且方法通过 `has_generalization_evidence=True` 启用该路径时，TraceAAD 计算泛化信号。

对每个实例，子程序相对父程序的结果编码为：

$$
g_j=
\begin{cases}
1,&\text{子程序改进},\\
0.5,&\text{结果持平},\\
0,&\text{子程序退步}.
\end{cases}
$$

单步泛化信号是所有实例上的平均值。轨迹泛化价值将所有步骤的平均迁移信号、最后一步信号和终点 robustness 组合：

$$
G(\tau)=0.6\overline{g}_{step}
+0.2g_{last}
+0.2r_{endpoint}.
$$

当前默认标量任务中该维度关闭，默认 `w_generalization=0`。因此不能把普通 scalar fitness 或机制成功率解释为真实跨实例泛化证据。

### 3.5 价值标量化与 Trajectory-UCB

轨迹价值向量为：

$$
V(\tau)=(Q(\tau),P(\tau),D(\tau),N(\tau),G(\tau)).
$$

采样时采用加权标量化：

$$
S(\tau)=
w_qQ+w_pP+w_dD+w_nN+w_gG+U(\tau).
$$

探索项为：

$$
U(\tau)=c_t
\sqrt{\frac{\log(V_{\mathrm{all}}+1)}
{n_\tau+1}},
$$

其中 $n_\tau$ 是轨迹访问次数，$V_{\mathrm{all}}$ 是活跃轨迹总访问次数。搜索阶段推进时 $c_t$ 线性衰减，但保留一个非零 floor；全局 best 长时间不变时，再加入与 stagnation 成正比的探索增量。

默认权重为：

$$
(w_q,w_p,w_d,w_n,w_g)=(0.50,0.20,0.15,0.15,0).
$$

轨迹选择不是在所有轨迹上直接贪心。候选集合包含：

- 标量价值最高的 top-$k$ 条轨迹，默认 $k=12$；
- 每个 island 的局部 top 轨迹；
- 当前全局 best endpoint 对应的 elite 轨迹。

elite 轨迹以默认 0.15 的概率直接采样；其他候选按照温度 softmax 采样。这样可以同时保护全局最优、岛间覆盖和低访问轨迹的探索机会。

## 4. 自适应算子组合

每轮搜索先根据当前轨迹状态筛选可用算子，再由 OperatorPortfolio 分配概率。算子概率由以下因素共同决定：

- 历史归一化收益；
- 有效生成率；
- 新颖候选比例；
- 退步比例；
- LLM 与评估时间成本；
- 产生 global best 或 near-record 的比例；
- 搜索阶段角色偏置。

收益使用 EMA 更新，温度随搜索阶段从高到低变化；每个算子保留最小采样概率，后期对 `novelty_jump` 设置概率上限。每个搜索 iteration 的一批候选只产生一次 portfolio 更新，使用该批次中最优候选的结果作为可比反馈。

### 4.1 Endpoint Refine

Endpoint Refine 从当前轨迹终点继续开发。当轨迹最近一步不是退步，或轨迹还没有历史边时，算子要求模型沿最近的有效方向提出一个针对性修改。若最近一步退步，Endpoint Refine 不可用，为 Backtrack Branch 让出机会。

### 4.2 Backtrack Branch

Backtrack Branch 主动扫描活跃轨迹，寻找终点退步或连续平台、但内部仍存在高价值前缀的轨迹。base node 从以下候选中产生：

1. 当前 endpoint；
2. 退步发生前的父节点；
3. 连续平台之前最近的改进节点；
4. 轨迹内部历史最佳节点。

候选 base 使用节点归一化质量、其前方正向改进和其后方回撤计算 branch score，选择得分最高且不同于 endpoint 的节点。新程序从该 base 分叉，并被要求避开导致原退步或平台的修改方向。

### 4.3 Mechanism Crossover

Mechanism Crossover 从另一个活跃轨迹选择 donor。donor 需要满足：

- 与当前轨迹的机制 profile 具有足够互补性；
- 终点质量至少达到活跃池归一化质量的 0.5；
- donor 机制没有被标记为 anti-pattern。

donor 选择分数结合机制互补性、终点质量和该机制在 crossover 上的历史改进率。生成提示要求只迁移一个主要机制，保留当前 base 程序的其他结构。新节点从 recipient 的 endpoint 分支。

### 4.4 Distill/Simplify

Distill/Simplify 只在当前 endpoint 相对活跃池复杂度较高，且轨迹发生平台或全局 best 已停滞至少 5 个 iteration 时触发。模型被要求删除低贡献代码、保留产生收益的核心机制，并在不降低适应度的情况下降低复杂度。

portfolio 对该算子的非负收益额外加入复杂度下降奖励。

### 4.5 Novelty Jump

Novelty Jump 在 global best 连续停滞至少 12 个 iteration 后触发，并受到 8 个 iteration 的触发 cooldown。它不从已有程序分叉，而是使用 initial-style prompt 生成一个新的完整程序，并创建一条新的初始轨迹。

候选机制族包括 `local_density`、`nn_rank`、`row_normalize`、`edge_contrast`、`sparsified_candidate`、`adaptive_exponent`、`hybrid_distance` 和 `randomization`。机制族根据 Novelty Jump 自身的后验成功率和尝试次数选择；连续失败的机制族进入 operator-conditioned cooldown。

### 4.6 Scale-Transfer

Scale-Transfer 从当前 endpoint 继续扩展，要求减少实例相关硬编码、使用更具尺度不变性的机制。该算子只有在评估器已经提供明确泛化证据时才可触发；普通 scalar task 中默认不会被激活。

## 5. 因果上下文与程序生成

对于 endpoint extension 或 prefix branching，TraceAAD 采用两阶段生成：

1. 先生成自然语言修改动作；
2. 再根据动作生成完整程序代码。

动作 prompt 包含四部分：

1. **任务契约**：task description 和待演化函数签名；
2. **当前轨迹叙事**：最近若干步的 parent、child、operator、mechanism、action、fitness 变化和 outcome；
3. **蒸馏模式**：PatternMemory 中跨图边统计得到的机制、operator-conditioned improve rate、lesson 和 anti-pattern；
4. **对比反馈**：近期 active trajectories 中 best 与 worst endpoint 的机制、想法和相对排名。

算子约束会被直接注入动作 prompt，例如“从高价值前缀分叉”“只迁移一个机制”或“降低复杂度但不降低适应度”。动作生成阶段只允许输出指定数量的修改列表，不输出代码和解释。

随后，代码 prompt 以选定 base node 的代码和单个动作作为输入，要求生成符合目标函数契约的完整程序。Novelty Jump 和初始化不经过动作阶段，而是直接使用 initial-style prompt 生成完整程序。

## 6. 候选注册与新颖性门控

每个成功评估的 refinement candidate 都会：

1. 加入程序推导图；
2. 创建 parent-child ImprovementEdge；
3. 计算有向适应度变化和 outcome；
4. 按算子语义扩展或分支出新轨迹；
5. 更新全局 best、相对排名和机制统计。

Novelty Jump 生成的完整起点没有父边，直接创建新的初始轨迹。

新轨迹进入活跃池前执行 novelty gate。候选会被拒绝的条件包括：

- 与活跃轨迹的组合相似度达到默认阈值 0.92；
- 历史上已经存在相同机制标签且适应度相同的行为重复。

被拒绝的轨迹立即归档，但刷新当前 global best 的候选具有质量覆盖权，可以绕过 gate 保留。新颖性门控只控制轨迹生存，不删除推导图中的程序节点和边。

LLM 解析失败不会消耗 evaluation budget；一旦程序提交给 evaluator，评估调用就计入样本预算，即使评估返回失败。

## 7. 周期性记忆更新

### 7.1 轨迹生存

每个完成的搜索 iteration 都会执行 survival：

1. 归档重复的 path key，只保留访问状态更有代表性的轨迹；
2. 使用当前活跃池重新计算所有轨迹价值；
3. 在每个 island 内按 ValueVec 的 Pareto fronts 排序；
4. 在全局活跃池上再次按 Pareto fronts 排序；
5. 同一 Pareto front 内再按标量价值排序；
6. 始终保护至少一条到达 global best endpoint 的轨迹。

默认每个 island 最多保留 40 条轨迹，全部 island 合计最多保留 160 条轨迹。归档只作用于轨迹，不删除程序推导图。

### 7.2 机制蒸馏

每隔 20 个 completed search iterations，distill 回路扫描推导图中的边，按机制标签和 `operator × mechanism` 统计改进率。统计支持：

- 机制模式：至少有足够支持且出现改进时写入 PatternMemory；
- operator-conditioned anti-pattern：在特定算子下支持充分但改进率低时写入；
- cooldown 恢复：后续证据改善时清除相应 anti-pattern。

机制证据使用真实图边作为 support ID，并通过 idempotent 记录避免同一证据重复计数。

### 7.3 相对反馈与反思

每当全局 best 停滞达到反思周期，并且自上次反思后积累了足够新边时，RankingModel 对 parent-child 比较维护 Elo 风格相对分数。不同连通分量之间不直接比较 Elo 分数，而使用原始 fitness 区分分量。

反思回路从近期 active trajectories 中选择 best 与 worst endpoint，生成结构化 contrast snapshot，并写入：

- best mechanism 的 lesson；
- worst mechanism 的 anti-pattern。

当前实现中的 reflect 是确定性的结构化统计过程，不额外调用 LLM。

### 7.4 Island Migration

当全局 best 处于停滞且达到 migration interval 时，每个 island 选择当前 scalar value 最高的轨迹，并将其移动到下一个 island。迁移保持轨迹 ID、访问次数和价值，不复制新的轨迹身份。

## 8. 完整搜索算法

    输入：LLM、程序模板、task evaluator、最大评估预算 B
    输出：best_node，以及可选的 best_generalization_node

    1. 使用机制多样性提示生成 n_init 个初始程序
    2. 评估有效程序，建立 Program Memory 和初始 Trajectory Memory
    3. while 未达到 B 且搜索未中止：
    4.     从 unique active trajectories 计算 ValueVec 与 trajectory-UCB
    5.     构造 top-k、island top 和 elite 候选池并采样轨迹
    6.     根据 trigger、历史收益和搜索阶段采样一个算子
    7.     算子确定目标轨迹和 base node
    8.     if Novelty Jump:
    9.         生成新的完整程序并创建初始轨迹
    10.    else:
    11.        构造因果轨迹、模式和对比反馈上下文
    12.        先生成动作，再为每个动作生成完整程序
    13.    评估候选并记录节点、边、delta、outcome 和可选泛化信号
    14.    按 endpoint extension、prefix branching 或 fresh start 更新轨迹
    15.    执行 novelty gate，更新 best、排名和机制统计
    16.    使用本轮最佳候选更新 operator portfolio 一次
    17.    每轮执行 survival；按周期执行 distill、reflect 和 migration
    18. 返回标量适应度最高的 best_node

搜索在以下任一条件满足时结束：达到最大评估样本数、搜索被显式中止、没有活跃轨迹，或连续多次尝试没有产生新的评估进度。

## 9. 当前实现配置与边界

TraceAAD 当前实现的主要默认配置如下：

| 配置 | 默认值 |
|---|---:|
| 初始程序数 `n_init` | 4 |
| 每轮动作数 `actions_per_iteration` | 2 |
| 最大轨迹长度 | 8 |
| island 数量 | 4 |
| 每 island 最大轨迹数 | 40 |
| trajectory 选择 | `trajectory_ucb` |
| novelty 阈值 | 0.92 |
| distill 周期 | 20 iterations |
| reflect patience | 20 iterations |
| migration 周期 | 20 iterations |
| reflect 最少新增边 | 8 |
| evaluator worker | 1 |
| 随机种子 | 0 |

需要明确的实现边界：

1. 当前常用 TSP/CVRP 标量 evaluator 只提供 scalar fitness，因此默认不启用真实 per-instance 泛化 credit；`fitness_vector`、robustness 和 Scale-Transfer 是已实现的可选接口。
2. 当前多样性与新颖性使用 token、机制标签和轨迹行为的 Jaccard 相似度，不使用外部 embedding 或 AST embedding。
3. 当前反思回路是 Elo 风格相对排名加结构化 lesson/anti-pattern 生成，不是额外的 reflection LLM。
4. 当前搜索最终返回 `best_node`；`Pareto fronts` 用于 trajectory survival，而不是向调用方返回完整的非支配程序集。
5. `confidence` 等字段保留在数据结构中，但当前默认 scalar 搜索不会用它单独改变轨迹选择。

因此，本文描述的是当前代码已经执行的搜索机制；跨实例泛化、额外 LLM 反思和 embedding 相似度不能被视为默认运行行为。
