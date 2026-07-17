# TraceAAD：基于算法改进轨迹的自动算法设计

本文按当前 `llm4ad.method.traceaad` 实现，说明 TraceAAD 的完整搜索机制：先给设计思想与总体结构，再逐层展开状态表示、价值选择、算子调度、生成评估与记忆更新。文中公式、触发条件与默认超参均对齐代码。

---

## 1. 设计目标与核心思想

### 1.1 问题设定

给定：

- 一个待设计算法任务（如 TSP constructive heuristic）；
- 可执行程序模板（唯一可演化函数签名与输出契约）；
- 大语言模型（LLM）；
- 程序评估器（对训练实例返回标量适应度）。

TraceAAD 在有限评估预算 $B$ 内，搜索适应度最优的有效程序。记候选程序为 $p$，标量适应度为 $f(p)$。最大化任务直接比较 $f$；最小化任务在比较、归一化与有向差分时反转方向。

训练评估器通常对同一 task 的多个实例取平均后返回标量 $f$；最终测试在同 task 的另一数据划分上进行，测试结果不反馈进搜索。

### 1.2 为什么以轨迹为搜索单位

许多 LLM 驱动的算法设计方法以「单个程序」或「树节点」为搜索对象：选一个程序、改一次、评估一次。这样做丢掉了改进过程本身——哪一步改进、哪一步退步、用了什么机制、属于哪条改进路径——而这些过程信息正是后续「该从哪里改、该怎么改」的依据。

TraceAAD 的核心主张是：

> **有界算法改进轨迹**是一等搜索资产。轨迹不仅携带当前程序，还携带导致当前程序的修改历史（动作、算子、机制、有向适应度变化、结果类型）。选择、扩展、归档、上下文构造与经验蒸馏都围绕轨迹展开。

因此：

- 选择时比较的是「路径价值」，而不只是终点分数；
- 扩展时可以沿终点继续，也可以从路径中部回退分叉；
- 生成时把近期因果叙事注入 prompt，而不是只贴当前代码；
- 周期性从边集合蒸馏机制经验，供后续算子与 prompt 使用。

### 1.3 三个配套设计原则

1. **过程信用逐步归因，不回传祖先。** 每条推导边只记录该步自己的 $\Delta$；后代改进不回传到更早节点（刻意避免 MCTS 式 max-backprop 的过度归功）。
2. **价值是多维的，采样与生存用途分离。** 轨迹价值 $V=(Q,P,D,N)$ 在采样时标量化并加 UCB；在活跃池截断时用 Pareto 非支配排序，避免四维塌成单一分数后过早丢掉多样性。
3. **算子是角色化的动作策略，由 portfolio 自适应调度。** 不是固定轮换某一种改法，而是根据触发条件与历史收益，在 exploit / 回退分叉 / 机制重组 / 简化 / 探索跳变之间采样。

---

## 2. 总体架构

### 2.1 三层记忆

| 层       | 实现                 | 存什么                                           | 在搜索中的角色                               |
| -------- | -------------------- | ------------------------------------------------ | -------------------------------------------- |
| 程序记忆 | `DerivationGraph`  | 所有已生成程序节点 + 父子推导边                  | 事实库（ground truth）；不可因轨迹归档而删图 |
| 轨迹记忆 | `TrajectoryMemory` | 当前仍值得投入预算的有界路径                     | 选择与扩展的直接对象                         |
| 模式记忆 | `PatternMemory`    | 机制模式、lesson、anti-pattern，以及算子条件证据 | 注入 prompt；约束 Novelty / Crossover 等算子 |

三层互补：图保存「发生过什么」；轨迹保存「现在还要继续跟哪条路」；模式保存「跨路径可复用的文字与统计经验」。

### 2.2 三回路

| 回路       | 做什么                                                                        | 何时触发                                  |
| ---------- | ----------------------------------------------------------------------------- | ----------------------------------------- |
| 进化主回路 | 选轨迹 → 选算子 → 生成 → 评估 → 写图/轨迹 → 新颖性门控 → 更新 portfolio | 每次 search attempt（有预算且有活跃轨迹） |
| 蒸馏回路   | 扫描全部推导边，更新机制改进率与 anti-pattern                                 | 每完成$20$ 个 search iteration          |
| 反思回路   | 用相对排名对比 best vs worst，写入 lesson / anti-pattern                      | 全局 best 停滞且满足边数门槛（见 §10）   |

此外还有两类辅助维护：

- **轨迹生存（survival）**：每个 completed iteration 都做，用 Pareto 截断活跃池；
- **Island 迁移（migration）**：停滞时周期性把各岛精英轨迹环形移动。

### 2.3 主循环数据流

```text
初始化
  用跨机制多样性提示生成 n_init 个程序
  → 建图与初始轨迹 → 按 island 安置
        │
        ▼
主循环（while 预算未尽且未中止）
  1. 轨迹选择（默认 Trajectory-UCB）
  2. Portfolio 在触发算子中采样
  3. 算子可覆盖选题（Backtrack）并确定 base + 约束
  4. Novelty：initial-style 直接生成；否则两阶段 refinement
  5. 评估 → 写节点/边 → 按算子语义插入轨迹
  6. 新颖性门控；更新 best、Elo、机制证据
  7. 用本批最优观测更新 portfolio 一次；选中轨迹 visit+1
  8. 若完成新的 search iteration：
       更新停滞计数；survival；按需 distill / reflect / migrate
        │
        ▼
末尾再做一次 survival → 返回标量最优 best_node
```

下文按这条数据流，从状态表示讲到局部模块。

---

## 3. 搜索状态表示

本节公式符号尽量与代码字段同名：例如 $\mathrm{idea}$ 对应 `idea`，$\mathrm{mech}$ 对应 `mechanism_tag`，$\mathrm{complexity}$ 对应 `complexity`。单字母只保留约定俗成的 $p$（程序）、$f$（适应度）、$\tau$（轨迹）、$\Delta$（有向差分）。

### 3.1 程序节点

每个候选程序对应推导图中的一个节点：

$$
p_i=(c_i,\mathrm{idea}_i,f_i,\mathrm{complexity}_i,\mathrm{mech}_i),
$$

| 符号 | 含义 | 代码字段 |
| --- | --- | --- |
| $c_i$ | 源代码 | `code` |
| $\mathrm{idea}_i$ | 自然语言设计思想 | `idea` |
| $f_i$ | 标量适应度；评估失败则为空 | `fitness` |
| $\mathrm{complexity}_i$ | 代码复杂度（默认 AST 节点数；评估器也可显式返回） | `complexity` |
| $\mathrm{mech}_i$ | 机制标签（见 §4） | `mechanism_tag` |

另有布尔字段 `is_valid`：无效或评估失败的程序仍可入图，但不参与轨迹价值计算。评估耗时由 profiler 记为 `evaluate_time`，**不进入节点表示，也不参与搜索决策**。

### 3.2 推导边与有向信用

推导图是**单父 DAG**：每个子节点至多一条入边。从父程序生成子程序时增加边：

$$
e=(p_u,p_v,\mathrm{action},\mathrm{op},\mathrm{mech},\Delta,\mathrm{outcome}),
$$

| 符号 | 含义 | 代码字段 |
| --- | --- | --- |
| $\mathrm{action}$ | 自然语言修改动作 | `action` |
| $\mathrm{op}$ | 搜索算子名 | `operator` |
| $\mathrm{mech}$ | 该步机制标签 | `mechanism_tag` |
| $\Delta$ | 有向适应度变化 | `delta` |
| $\mathrm{outcome}$ | 结果类型（improve / regress / plateau / unknown） | `outcome` |
| — | 写入时的搜索 iteration（蒸馏回路使用） | `iteration` |

有向变化：

$$
\Delta=
\begin{cases}
f_{\mathrm{child}}-f_{\mathrm{parent}}, & \text{最大化},\\
f_{\mathrm{parent}}-f_{\mathrm{child}}, & \text{最小化}.
\end{cases}
$$

结果类型 $\mathrm{outcome}$ 由 $\Delta$ 与阈值 $\varepsilon=10^{-6}$ 决定：

$$
\mathrm{outcome}=
\begin{cases}
\mathrm{improve}, & \Delta>\varepsilon,\\
\mathrm{regress}, & \Delta<-\varepsilon,\\
\mathrm{plateau}, & |\Delta|\le\varepsilon,\\
\mathrm{unknown}, & \Delta=\mathrm{null}.
\end{cases}
$$

**逐步归因**：边只承担自身 $\Delta$；不把后代改进回传到祖先。路径潜力（§5.2）是对轨迹上逐步变化的加权统计，不是树搜索式的 backup。

### 3.3 有界轨迹

轨迹是推导图中的一条有序路径：

$$
\tau=(p_0,e_1,p_1,\ldots,e_L,p_L).
$$

每条轨迹记录：

| 字段                         | 含义                             |
| ---------------------------- | -------------------------------- |
| `node_ids` / `edge_ids`  | 路径上的节点与边序列             |
| `base_id`                  | 当前窗口起点                     |
| `endpoint_id`              | 当前终点（始终为序列末节点）     |
| `island_id`                | 所属岛                           |
| `visit_count`              | 被选作扩展目标的次数             |
| `status`                   | `active` / `archived`        |
| `value` / `scalar_value` | 最近一次算出的$V$ 与标量化分数 |

默认最大长度 $L_{\max}=8$。超过上限时**滑动窗口**：丢掉最前端节点与边，保留最近后缀；滑窗后 `base_id` 更新为新序列首节点。

轨迹扩展有三种语义：

| 语义               | 含义                                             | 使用算子                  |
| ------------------ | ------------------------------------------------ | ------------------------- |
| endpoint extension | 从当前终点追加子节点                             | Endpoint Refine、Simplify |
| prefix branching   | 截取内部 base 前缀，再接新子节点，形成新轨迹身份 | Backtrack、Crossover      |
| fresh start        | 无父边，以新节点创建长度 1 的轨迹                | Novelty Jump、初始化      |

关键约定：

- 扩展或分叉**总是新建轨迹对象**（新 ID），不原地改写父轨迹；父轨迹可继续留在活跃池。
- 归档只改轨迹状态，**不删除**图中节点与边。
- 路径键 `path_key=(node_ids, edge_ids)` 用于去重：活跃池中同一路径只保留访问次数更高者（并列时 ID 更小）。

### 3.4 模式记忆

PatternMemory 存三类条目：

| kind             | 含义                                   | 主要来源    |
| ---------------- | -------------------------------------- | ----------- |
| `mechanism`    | 某机制在唯一图边上的改进统计与文字摘要 | 蒸馏回路    |
| `lesson`       | 相对排名中表现强的机制经验             | 反思回路    |
| `anti_pattern` | 全局或算子条件下反复失败的方向         | 蒸馏 / 反思 |

每条模式含：文本、机制标签、support ID 集合、`improve_rate`（该机制在唯一图边上的改进率）、`confidence`。

合并与容量：

- `lesson` / `anti_pattern` 按 `(mechanism_tag, operator_scope)` 合并；
- `mechanism` 按标签增量 upsert；
- 每类容量默认 $50$，超出时按 $\mathrm{confidence}\times\max(\mathrm{improve\_rate},0.1)$ 淘汰。

此外维护幂等证据表：

$$
(\mathrm{operator},\,\mathrm{mechanism},\,\mathrm{support\_id})\mapsto(\mathrm{success},\,\mathrm{iteration}).
$$

用于算子条件下的尝试次数、成功率、连续失败 streak 与 cooldown（Novelty Jump 使用）。

---

## 4. 机制标签推断

机制标签不是人工标注，而是规则从文本推断。系统维护面向 constructive heuristic（尤其 TSP 类）的预设族：

| 机制族                   | 关键词示例                                       |
| ------------------------ | ------------------------------------------------ |
| `local_density`        | density, local density                           |
| `nn_rank`              | nearest neighbor rank, nn rank, ranking          |
| `row_normalize`        | row-wise, row normalization, normalize           |
| `edge_contrast`        | edge contrast, contrast, difference-based        |
| `sparsified_candidate` | sparsif, candidate list, prune candidate         |
| `adaptive_exponent`    | adaptive exponent, power law, weighting exponent |
| `hybrid_distance`      | hybrid distance, distance-statistical            |
| `randomization`        | random, stochastic, noise, probabilistic         |

推断过程：

1. 将动作文本、程序思想与代码拼接后转小写；
2. 按上表顺序做关键词匹配；
3. 若匹配前出现 remove / avoid / without 等否定语境，则不计入该族；
4. 若无正向匹配但存在算子侧 hint（Novelty 指定族、Crossover 的 donor 机制），则采用 hint；
5. 若文本明确在「删除」hint 对应机制，则回退为 `other`。

初始化阶段还会用多样性提示文本辅助推断。标签同时写入节点与边，供相似度、donor 选择、PatternMemory 统计与 Novelty 的 island 分配使用。

换到差异很大的任务时，大量程序可能落入 `other`；这是词表的领域边界，不是搜索逻辑故障。

---

## 5. 轨迹价值与选择

价值计算的对象是**唯一活跃轨迹**（同 `path_key` 只留一个代表）。无效终点对应全零 $V$。

### 5.1 终点质量 Q

先用当前活跃轨迹的唯一有效终点适应度，做 10%/90% 分位数线性插值裁剪，得到边界 $(f_{\min},f_{\max})$。这样可避免归档程序、重复路径或极端失败压缩尺度。

$$
Q(\tau)=\mathrm{clip}\!\left(\frac{f(p_L)-f_{\min}}{f_{\max}-f_{\min}},0,1\right).
$$

最小化任务在归一化前对适应度与边界一并取负。边界缺失时：最大化返回 $1.0$，最小化返回 $0.0$；$|f_{\max}-f_{\min}|\lt 10^{-12}$ 时返回中性值 $0.5$。

### 5.2 路径潜力 P

令轨迹上各节点归一化质量为 $q_0,\ldots,q_L$，逐步质量变化 $\Delta q_i=q_i-q_{i-1}$。近端步权重更高：第 $i$ 步（从近到远倒数）权重为 $\gamma^{n-i-1}$，默认 $\gamma=0.8$。折扣路径回报：

$$
R(\tau)=
\frac{\sum_i w_i \Delta q_i}{\sum_i w_i}
+\lambda_{\mathrm{pos}}\frac{1}{n}\sum_i\mathbb{I}(\Delta q_i>\varepsilon)
-\lambda_{\mathrm{down}}
\frac{\sum_i w_i\max(-\Delta q_i,0)}{\sum_i w_i},
$$

默认 $\lambda_{\mathrm{pos}}=0.25$，$\lambda_{\mathrm{down}}=0.5$。长度不足 $2$ 的轨迹 $R=0$。

为抑制「低质量终点但单次大幅回升」主导选择，潜力经质量门控：

$$
P(\tau)=
\begin{cases}
0, & Q(\tau)\le q_{\min},\\
R(\tau)\dfrac{Q(\tau)-q_{\min}}{1-q_{\min}}, & Q(\tau)>q_{\min},
\end{cases}
\qquad q_{\min}=0.5.
$$

### 5.3 多样性 D 与新颖性 N

不使用外部 embedding。三种 Jaccard 相似度：

1. **程序相似度**：去注释与多余空白后，对标识符 token 集合做 Jaccard；
2. **机制相似度**：轨迹机制 profile（起点机制 $\cup$ 各边机制）的集合 Jaccard；
3. **轨迹行为相似度**：边上 `(operator|outcome)` 指纹集合的 Jaccard。

组合相似度（默认权重 $0.4,0.4,0.2$）：

$$
\mathrm{sim}(\tau_a,\tau_b)=
\frac{w_c\mathrm{sim}_{code}+w_m\mathrm{sim}_{mech}+w_t\mathrm{sim}_{traj}}{w_c+w_m+w_t}.
$$

对目标轨迹，相对其它唯一活跃路径：

$$
D(\tau)=1-\mathrm{mean}_{\tau'\ne\tau}\mathrm{sim}(\tau,\tau'),\qquad
N(\tau)=1-\max_{\tau'\ne\tau}\mathrm{sim}(\tau,\tau').
$$

无其它轨迹时 $D=N=1$。二者都来自同一相似度体系：$D$ 强调平均疏离，$N$ 强调相对最近邻的疏离；生存阶段的 Pareto 截断会同时保留这两维。

### 5.4 标量化与 Trajectory-UCB

价值向量（质量 / 潜力 / 多样性 / 新颖性）：

$$
V(\tau)=(Q,P,D,N)
=(V_{\mathrm{quality}},V_{\mathrm{potential}},V_{\mathrm{diversity}},V_{\mathrm{novelty}}).
$$

采样用加权标量加探索项：

$$
S(\tau)=w_qQ+w_pP+w_dD+w_nN+U(\tau),
$$

默认 $(w_q,w_p,w_d,w_n)=(0.50,0.20,0.15,0.15)$。

$$
U(\tau)=c_t\sqrt{\frac{\log(V_{\mathrm{all}}+1)}{n_\tau+1}},
$$

其中 $n_\tau$ 为该轨迹访问次数，$V_{\mathrm{all}}$ 为唯一活跃轨迹总访问次数。系数

$$
c_t=\max\!\big(c_{\mathrm{floor}},\,c_0(1-t/T)\big)
+\beta\cdot\min(s_{\mathrm{stag}}/T,1),
$$

默认 $c_0=0.4$，$c_{\mathrm{floor}}=0.05$，$\beta=0.20$；$t$ 为当前 search iteration，$T$ 为计划总 iteration，$s_{\mathrm{stag}}$ 为全局 best 连续未更新的停滞计数。随着搜索推进，基础探索衰减，但停滞时会重新抬高探索。

### 5.5 选择流程

默认策略 `trajectory_ucb` 不是全局贪心：

1. 对所有唯一活跃轨迹计算 $V$ 与 $S$，写回轨迹记忆；
2. 取标量最高的 top-k（默认 $k=12$）；
3. 每个 island 再并入局部 top（默认每岛 $1$ 条），避免小岛被全局 top 完全淹没；
4. 并入当前全局 best endpoint 对应的 elite 轨迹（同 endpoint 多条时取访问次数最少者）；
5. 以概率 $0.15$ 直接返回 elite；否则在候选集上按温度 $T_{\mathrm{sel}}=0.8$ 做 softmax 采样。

可选对照策略：`best`（按终点适应度选最优）、`random`（均匀抽唯一活跃轨迹）。实验默认使用 `trajectory_ucb`。

---

## 6. 自适应算子组合

每轮先构造 `OperatorContext`（推导图、轨迹记忆、模式记忆、island 管理器、当前选中轨迹、maximize、停滞计数、算子侧信道 hints），再由 OperatorPortfolio 在**通过 trigger 的算子**中采样。若无一算子触发，回退到 Endpoint Refine。

五个默认算子及其角色：

| 算子                | 名称                    | role         | 意图                             |
| ------------------- | ----------------------- | ------------ | -------------------------------- |
| Endpoint Refine     | `endpoint_refine`     | exploit      | 沿当前终点继续局部强化           |
| Backtrack Branch    | `backtrack_branch`    | path_correct | 从退步/平台前的高价值前缀分叉    |
| Mechanism Crossover | `mechanism_crossover` | recombine    | 把互补机制迁入当前程序           |
| Distill / Simplify  | `distill_simplify`    | simplify     | 在不降适应度前提下降复杂度       |
| Novelty Jump        | `novelty_jump`        | explore      | 停滞时跳到新机制族做 fresh start |

统一协议：

```text
trigger → (可选 select_trajectory 覆盖选题) → select_base → build_constraint
→ 主循环生成与评估 → insert
```

### 6.1 Endpoint Refine（exploit）

**触发**：选中轨迹终点有效；且无历史边，或最近一步不是 regress。若最近一步 regress，则让位给 Backtrack。

**Base**：当前 endpoint。

**约束**：沿最近有效方向继续做一次针对性强化，避开已知退步方向。

**接入**：`extend`（endpoint extension）。

### 6.2 Backtrack Branch（path_correct）

该算子**不依赖** UCB 刚选中的轨迹。原因是高潜力轨迹末端多为 improve，会使「末步 regress 才触发」永久不满足。实现上主动扫描活跃池：

1. 跳过无边轨迹；
2. 要求终点最近一步为 regress 或 plateau；
3. 用四规则 + `branch_score` 选出不同于 endpoint 的内部 base；
4. 在满足条件的轨迹中取 `branch_score` 最高者，替换 `ctx.selected`。

四规则候选 base：

1. 当前 endpoint；
2. 若末步 regress，则退步前的父节点；
3. 若连续两步 plateau，则最近一次 improve 的子节点；
4. 轨迹内部历史最佳节点（若异于 endpoint）。

$$
\mathrm{score}(v)=q(v)+0.3\cdot\mathrm{fwd\_improve}-0.3\cdot\mathrm{bwd\_regress},
$$

其中 $q(v)$ 用**全图**适应度范围归一化；前向正改进与后向回撤分别对前缀/后缀步平均。最终取 score 最大者作为 base。

**约束**：从高价值前缀分叉，提出与导致退步/平台**不同**的修改。

**接入**：`branch_from`（prefix branching）。

### 6.3 Mechanism Crossover（recombine）

**触发**：当前终点有效，且能选到合格 donor。

**Donor 条件**：

- 机制 profile 互补性 $1-\mathrm{sim}_{mech}\ge 0.5$；
- 终点质量 $\ge 0.5$（优先用已缓存 $Q$，否则用活跃终点适应度范围归一化）；
- donor 终点机制在 crossover 算子作用域下不是 anti-pattern。

**Donor 分数**：互补性 $+0.3\times$ 质量 $+0.5\times$ crossover 条件下该机制历史改进率（无证据时用 $0.5$）。

**Base**：recipient 的 endpoint；donor 机制与思想写入 hints，供机制推断与约束文本使用。

**约束**：只迁移一个主要机制，保留 base 程序其余结构。

**接入**：`branch_from`。

### 6.4 Distill / Simplify（simplify）

**触发**需同时满足：

1. 活跃有效终点至少 $2$ 个，且当前复杂度 $\gt 0$；
2. 当前复杂度处于活跃终点复杂度上四分位及以上，且高于中位数；
3. 最近一步为 plateau，**或**全局 best 停滞 $\ge 5$ 个 completed iteration。

**约束**：删除低贡献代码，保留产生收益的核心机制，在不降适应度前提下降低复杂度。

**接入**：`extend`。

### 6.5 Novelty Jump（explore）

**触发**：

1. 全局 best 停滞 $\ge 12$；
2. 存在未处于 anti-pattern / failure-cooldown 的候选机制族；
3. 距上次触发至少 $8$ 个 iteration。

**Base**：无（fresh start）。

**机制族选择**：在 $8$ 个预设族中，按 Novelty Jump 自身的 Beta(1,1) 后验成功率 $(s+1)/(n+2)$ 选最高者；尝试次数多者在并列时靠后，以轮换连败族。连续失败 $\ge 2$ 次后冷却 $24$ 个 iteration。

**生成**：不走动作阶段，直接用 initial-style prompt，将算子约束作为 diversity hint。

**接入**：`create_initial`；新轨迹的 island 由观察到的机制标签 SHA-256 哈希取模分配，促使同机制聚簇。

---

## 7. Operator Portfolio

### 7.1 候选与采样

候选 = 当前 `OperatorContext` 下 `trigger` 为真的算子；若为空，强制 Endpoint Refine。

每个算子维护 EMA 统计：归一化收益、有效率、新颖率、退步率、成本、刷新 global best 比例、near-record 比例。EMA 衰减默认 $0.8$。算子价值：

$$
\begin{aligned}
v(\mathrm{op})=&\alpha\tanh(\widehat{g})
+\beta_v\widehat{v}
+\beta_n\widehat{n}
-\delta_r\widehat{r}
-\delta_c\tanh(\widehat{c}/C)\\
&+b_{\mathrm{gb}}\widehat{\mathrm{gb}}
+b_{\mathrm{nr}}\widehat{\mathrm{nr}}
+\mathrm{role\_bonus}(\mathrm{op},\mathrm{phase}).
\end{aligned}
$$

默认 $\alpha=1$，$\beta_v=0.5$，$\beta_n=0.3$，$\delta_r=0.5$，$\delta_c=0.05$，$C=120$（秒量级成本尺度），$b_{\mathrm{gb}}=0.75$，$b_{\mathrm{nr}}=0.25$。

搜索进度划分为 early（$t/T \lt 1/3$）、mid（$1/3 \le t/T \lt 2/3$）、late（$t/T \ge 2/3$）；各 role 的阶段 bonus：

| role         | early |  mid | late |
| ------------ | ----: | ---: | ---: |
| explore      |  0.20 | 0.10 | 0.05 |
| recombine    |  0.25 | 0.40 | 0.15 |
| path_correct |  0.25 | 0.25 | 0.25 |
| exploit      |  0.20 | 0.35 | 0.50 |
| simplify     |  0.05 | 0.20 | 0.40 |

算子采样温度（记为 $T_{\mathrm{op}}$，与轨迹符号 $\tau$ 区分）调度：

$$
T_{\mathrm{op}}(t)=\max\!\big(T_{\mathrm{floor}},\,
T_{\mathrm{init}}+(T_{\mathrm{end}}-T_{\mathrm{init}})\tfrac{t}{T}\big),
$$

默认 $T_{\mathrm{init}}=1.0$，$T_{\mathrm{end}}=T_{\mathrm{floor}}=0.5$，因此温度从 $1.0$ 线性降到 $0.5$ 后保持。候选概率先对 $v(\mathrm{op})/T_{\mathrm{op}}$ 做 softmax，再施加最小概率地板 $0.05$；晚期阶段将 `novelty_jump` 概率上限截到 $0.2$，多余质量重新分配给其它候选。

### 7.2 批次信用更新

每个 search attempt 只对所选算子做**一次** portfolio 更新；幂等键是 `(operator_name, attempt_id)`，不是 search iteration。反馈取该批候选中最优者（按 maximize/minimize）。

归一化收益：有向 $\Delta$ 除以当前活跃终点适应度尺度，再截断到 $[-1,1]$。尺度定义为唯一活跃终点适应度的 10%–90% 分位差，并与中位数的 5% 取 max，下限 $10^{-3}$；活跃终点不足 $2$ 个时尺度取 $1$。若该批无任何有效观测，记惩罚收益 $-1$。

对 Simplify，若收益非负且父复杂度 $\gt 0$，再混入复杂度下降奖励：

$$
\mathrm{reward}\leftarrow 0.8\cdot\mathrm{reward}+0.2\cdot\frac{\mathrm{complexity}_{\mathrm{parent}}-\mathrm{complexity}_{\mathrm{child}}}{\mathrm{complexity}_{\mathrm{parent}}}.
$$

near-record：相对本轮开始时的 incumbent，有向短差不低于 $-0.10\times$ 适应度尺度。Novelty Jump 的机制成功证据按「是否 near-record」记录；refinement 边按「是否 improve」记录。

---

## 8. 上下文构造与程序生成

### 8.1 初始化

在评估预算内生成 $n_{\mathrm{init}}$（默认 $4$）个初始程序。每个 slot 使用轮换的机制多样性提示（最近邻排序、局部密度、行归一化、稀疏候选表），走 initial-style prompt：任务描述 + 多样性提示 + 目标函数契约，要求输出：

```text
Idea: <自然语言设计思想>
Code:
<一个完整的 Python 函数，放在 markdown 的 python 代码块中>
```

有效程序按 `slot % n_islands` 分配到不同 island，并创建长度为 $1$ 的初始轨迹。解析失败不消耗评估预算；一旦提交评估即计入样本预算。连续生成停滞达到 `max_stalled_iterations` 会提前结束初始化。

### 8.2 两阶段 Refinement

除 Novelty Jump 与初始化外，均采用两阶段生成。

**阶段 A：动作生成。** `build_action_prompt` 拼接：

1. 任务描述与适应度方向说明；
2. 当前轨迹最近至多 $5$ 步因果叙事（parent→child、算子、机制、动作、fitness 变化、outcome）；
3. PatternMemory 蒸馏块（top-4 机制及其改进率；当前算子条件下的 improve rate；top-3 lesson/anti-pattern）；
4. RankingModel 的 best vs worst 对比反馈；
5. 算子名、role 与约束文本；
6. base 节点代码、思想与选择原因；
7. 目标函数契约；
8. 指令：提出恰好 `actions_per_iteration`（默认 $2$）条编号修改，每条只改一个主机制，禁止输出代码与解释。

**阶段 B：代码生成。** 对每个解析出的动作，用 base 节点代码 + 该动作构造 code prompt，要求返回新的 Idea 与完整函数实现，保持签名与输出契约不变。

动作解析容忍编号、项目符号、`Action k:` 前缀，截断到期望条数。程序解析优先提取 Idea 行与首个代码块，再映射到模板中的唯一可演化函数；失败则尝试 trimmer。LLM 异常或解析失败不计评估预算。

### 8.3 每轮开销

典型 refinement 一轮：

- $1$ 次动作 LLM 调用；
- 至多 `actions_per_iteration` 次代码 LLM 调用；
- 至多同等次数的评估（每次评估 $+1$ 样本预算）。

Novelty Jump 一轮：至多 `actions_per_iteration` 次 initial-style LLM + 评估，无动作阶段。

一个 completed search iteration，大致对应相对搜索起点前进了 `actions_per_iteration` 个评估样本（实现上用样本计数整除判定）。

---

## 9. 候选注册、新颖性门控与相对排名

### 9.1 注册流程

每个成功评估的 refinement 候选：

1. 以推断的机制标签写入新节点；
2. 写入父→子边（动作、算子、机制、$\Delta$、outcome、iteration）；
3. 按算子 `insert` 语义扩展 / 分叉 / 新建轨迹；
4. 用父子适应度更新 Elo 风格相对排名；
5. 更新全局 `best_node`；
6. 向 PatternMemory 记录该边的幂等机制结果；
7. 执行 novelty gate。

Novelty Jump 候选无父边，直接建初始轨迹；机制成功按 near-record 记账。

### 9.2 新颖性门控

新轨迹进入活跃池前检查：

1. **结构相似**：与其它活跃轨迹的组合最大相似度是否 $\ge 0.92$；
2. **行为重复**：历史任意轨迹中是否已存在「相同机制标签且适应度数值相等」。

若命中且该候选**未**刷新当前 live global best，则立即归档该轨迹。若刷新了 global best，则质量覆盖，强制保留。门控只影响轨迹生存，不删除图节点 / 边。通过门控后为该轨迹计算并写入 $V$。

### 9.3 相对排名模型

`RankingModel` 对每次父子比较做 Elo 更新（$K=16$），并维护比较图的连通分量。反思用的 best/worst 对比：

1. 取最近 window（默认 $20$）条活跃轨迹的有效终点；
2. 先按 raw fitness 选出最优 / 最差分量；
3. 分量内再按 Elo 分数（并列时参考 raw fitness）取 best / worst。

未连通分量之间不直接比较 Elo。该模型服务于 prompt 对比块与反思回路，**不**进入 `OperatorContext`，也不直接决定轨迹 UCB 选择。

---

## 10. 周期性记忆更新

每个**完成的** search iteration 执行 `_periodic_hooks`。停滞计数在 completed iteration 上更新：best 未变则 $+1$，否则清零。

### 10.1 轨迹生存（每轮）

1. 归档重复 `path_key`，只留代表轨迹；
2. 重算全部活跃轨迹的 $V$ 与标量价值；
3. 保护至少一条到达当前 global best endpoint 的轨迹（多条时取 Pareto 序中的第一条）；
4. 各 island 内按 ValueVec 的非支配层排序，同层内按标量价值降序，截断到 `max_per_island=40`；
5. 全局再截断到 `max_active_trajectories`（默认 $n_{\mathrm{islands}}\times\max_{\mathrm{per\_island}}=160$）。

Pareto 支配：四维价值均 $\ge$ 且至少一维严格 $\gt$。归档不删图。对外返回仍是单一 `best_node`，Pareto 只服务于活跃池管理。

### 10.2 机制蒸馏（每 $20$ 个 completed iteration）

扫描全部推导边，幂等写入 `operator × mechanism` 证据，并按机制聚合：

- 若某机制支持 $\ge 2$ 且改进率 $\ge 0.4$，清除其全局 anti-pattern；
- 若支持 $\ge 2$ 且至少一次改进，upsert `mechanism` 模式（文本报告 improve 比例，分数为改进率）；
- 若某 `operator × mechanism` 支持 $\ge 5$ 且改进率 $\lt 0.2$，写入 operator-scoped anti-pattern；若随后改进率回升则清除。

### 10.3 反思（停滞触发）

当停滞计数 $\gt 0$ 且为 `patience_reflect=20` 的倍数，且自上次反思以来新增边数 $\ge 8$ 时触发。用 RankingModel 对比生成 lesson（best 机制）与 anti-pattern（worst 机制，且不同于 best）；机制标签为 `other` 时跳过。该过程为确定性模板文本，**不额外调用 LLM**。

### 10.4 Island 迁移（停滞触发）

当停滞 $\gt 0$ 且 completed iteration 为 `migration_interval=20` 的倍数时，每个 island 将其标量价值最高的 $1$ 条轨迹**移动**到下一 island（环形）。迁移保持轨迹 ID、访问次数与价值，不复制新身份。

Island 分配规则：

- 初始化：`slot % n_islands`；
- Novelty Jump：`SHA-256(mechanism_tag) % n_islands`。

---

## 11. 完整搜索算法

**输入**：LLM、程序模板、task evaluator、最大评估预算 $B$。

**输出**：标量最优 `best_node`。

```text
1.  用机制多样性提示生成并评估至多 n_init 个初始程序
2.  建立推导图与初始轨迹，按 island 轮转安置
3.  while 未耗尽预算且未中止：
4.      if 无活跃轨迹: 停止
5.      按 trajectory_ucb / best / random 选择轨迹
6.      构造 OperatorContext，由 portfolio 采样算子
7.      若算子 override 选题（Backtrack），替换目标轨迹
8.      确定 base node 与算子约束
9.      if Novelty Jump:
10.         initial-style 生成至多 A 个新程序并建初始轨迹
11.     else:
12.         构造因果/模式/对比上下文，生成 A 条动作
13.         对每条动作生成完整程序并评估
14.         写节点、边、Δ、outcome，按算子语义更新轨迹
15.     novelty gate；更新 best、Elo、机制证据
16.     用本批最优候选更新 portfolio 一次；记录选中轨迹访问 +1
17.     若完成新的 search iteration:
18.         更新 stagnation；执行 survival
19.         按周期执行 distill；按停滞条件执行 reflect / migration
20. if 仍有活跃轨迹: 再执行一次 survival
21. 返回 best_node
```

终止条件：达到预算、显式中止、无活跃轨迹，或连续 `max_stalled_iterations=20` 次 attempt 未产生新的评估进度。

---

## 12. 默认超参一览

### 12.1 搜索主配置

| 配置                      |             默认值 |
| ------------------------- | -----------------: |
| `n_init`                |                  4 |
| `actions_per_iteration` |                  2 |
| `max_trajectory_length` |                  8 |
| `n_islands`             |                  4 |
| `max_per_island`        |                 40 |
| 全局活跃轨迹上限          |                160 |
| 采样策略                  | `trajectory_ucb` |
| novelty 阈值              |               0.92 |
| distill 周期              |      20 iterations |
| reflect patience          |      20 iterations |
| migration 周期            |      20 iterations |
| reflect 最少新增边        |                  8 |
| evaluator worker          |                  1 |
| 随机种子                  |                  0 |
| 连续停滞停止              |        20 attempts |

### 12.2 价值与选择

| 配置                                                 |                  默认值 |
| ---------------------------------------------------- | ----------------------: |
| $(w_q,w_p,w_d,w_n)$                                | $(0.5,0.2,0.15,0.15)$ |
| 相似度权重$(w_c,w_m,w_t)$                          |       $(0.4,0.4,0.2)$ |
| 路径折扣$\gamma$                                   |                     0.8 |
| $(\lambda_{\mathrm{pos}},\lambda_{\mathrm{down}})$ |          $(0.25,0.5)$ |
| 潜力质量门控$q_{\min}$                             |                     0.5 |
| 适应度裁剪分位                                       |                    0.10 |
| UCB$(c_0,c_{\mathrm{floor}},\beta)$                |     $(0.4,0.05,0.20)$ |
| top-k / island top                                   |                  12 / 1 |
| elite 直采概率                                       |                    0.15 |
| 选择 softmax 温度 $T_{\mathrm{sel}}$ | 0.8 |

### 12.3 Portfolio 与算子内置阈值

| 配置                                                                 |                   默认值 |
| -------------------------------------------------------------------- | -----------------------: |
| EMA 衰减                                                             |                      0.8 |
| $(\alpha,\beta_v,\beta_n,\delta_r,\delta_c)$                       | $(1,0.5,0.3,0.5,0.05)$ |
| 成本尺度$C$                                                        |                      120 |
| $(b_{\mathrm{gb}},b_{\mathrm{nr}})$                                |          $(0.75,0.25)$ |
| near-record 容差                                                     |                     0.10 |
| $(T_{\mathrm{init}},T_{\mathrm{end}},T_{\mathrm{floor}})$ | $(1.0,0.5,0.5)$ |
| 最小算子概率 / late novelty cap                                      |               0.05 / 0.2 |
| Simplify 停滞门槛                                                    |                        5 |
| Novelty 停滞 / 触发冷却 / 族失败冷却                                 |              12 / 8 / 24 |
| Crossover 互补性 / 质量门槛                                          |                0.5 / 0.5 |
| Elo$K$ / reflect window                                            |                  16 / 20 |
| Pattern 每类容量                                                     |                       50 |
| distill`min_support`                                               |                        2 |

---

本文描述的是当前代码实际执行的搜索机制，可作为后续改动的对照基线。
