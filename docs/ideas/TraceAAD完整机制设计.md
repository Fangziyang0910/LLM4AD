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

许多 LLM 驱动的算法设计方法以「单个程序」或「树节点」为搜索对象：选一个程序、改一次、评估一次。这样做丢掉了改进过程本身——哪一步改进、哪一步退步、用了什么动作、属于哪条改进路径——而这些过程信息正是后续「该从哪里改、该怎么改」的依据。

TraceAAD 的核心主张是：

> **有界算法改进轨迹**是一等搜索资产。轨迹不仅携带当前程序，还携带导致当前程序的修改历史（动作、算子、有向适应度变化、结果类型）。选择、扩展、归档、上下文构造，以及边级经验检索，都围绕轨迹展开。

因此：

- 选择时比较的是「路径价值」，而不只是终点分数；
- 扩展时可以沿终点继续，也可以从路径中部回退分叉；
- 生成时把近期因果叙事注入 prompt，而不是只贴当前代码；
- 从当前 run 的真实推导边中检索少量成功/失败 action，作为无标签经验写回后续 prompt。

### 1.3 三个配套设计原则

1. **过程信用逐步归因，不回传祖先。** 每条推导边只记录该步自己的 $\Delta$；后代改进不回传到更早节点（刻意避免 MCTS 式 max-backprop 的过度归功）。
2. **价值是多维的，采样与生存用途分离。** 轨迹价值 $V=(Q,P,D,N,C,R)$ 在采样时标量化并加 UCB；在活跃池截断时用 Pareto 非支配排序，避免六维塌成单一分数后过早丢掉多样性与效率折中。
3. **算子是角色化的动作策略，由 portfolio 自适应调度。** 不是固定轮换某一种改法，而是在可行性候选上按历史收益与阶段先验做概率采样；trigger 只排除结构性不可行，不把探索/利用策略写成硬 if-else。

---

## 2. 总体架构

### 2.1 三层记忆

| 层       | 实现                 | 存什么                                      | 在搜索中的角色                               |
| -------- | -------------------- | ------------------------------------------- | -------------------------------------------- |
| 程序记忆 | `DerivationGraph`  | 所有已生成程序节点 + 父子推导边             | 事实库（ground truth）；不可因轨迹归档而删图 |
| 轨迹记忆 | `TrajectoryMemory` | 当前仍值得投入预算的有界路径                | 选择与扩展的直接对象                         |
| 经验记忆 | `ExperienceMemory` | 对图边的只读视图：有界成功/失败 action 检索 | 指导 refinement 动作生成                     |

三层互补：图保存「发生过什么」；轨迹保存「现在还要继续跟哪条路」；经验记忆是对图边的有界查询，不复制第二份事实，也不依赖任务相关机制词表。

### 2.2 主回路与维护

| 回路        | 做什么                                                                        | 何时触发                                  |
| ----------- | ----------------------------------------------------------------------------- | ----------------------------------------- |
| 进化主回路  | 选轨迹 → 选算子 → 生成 → 评估 → 写图/轨迹 → 新颖性门控 → 更新 portfolio | 每次 search attempt（有预算且有活跃轨迹） |
| 轨迹生存    | Pareto 截断活跃池                                                             | 每个 completed iteration                  |
| Island 迁移 | 各岛精英轨迹环形移动                                                          | 停滞时周期性触发                          |

不再有周期机制聚合或对比反思回路；RankingModel 仍在每次 refinement 前即时生成 best/worst 对比。

### 2.3 主循环数据流

```text
初始化
  用 run-local idea 去重提示生成 n_init 个程序
  → 建图与初始轨迹 → 按 slot % n_islands 安置
        │
        ▼
主循环（while 预算未尽且未中止）
  1. 轨迹选择（默认 Trajectory-UCB）
  2. Portfolio 在触发算子中采样
  3. 算子可覆盖选题（Backtrack）并确定 base + 约束
  4. Novelty：initial-style 直接生成；否则两阶段 refinement
  5. 评估 → 写节点/边 → 按算子语义插入轨迹
  6. 新颖性门控；更新 best、Elo
  7. 用本批最优观测更新 portfolio 一次；选中轨迹 visit+1
  8. 若完成新的 search iteration：
       更新停滞计数；survival；按需 migrate
        │
        ▼
末尾再做一次 survival → 返回标量最优 best_node
```

下文按这条数据流，从状态表示讲到局部模块。

---

## 3. 搜索状态表示

本节公式符号尽量与代码字段同名：例如 $\mathrm{idea}$ 对应 `idea`，$\mathrm{complexity}$ 对应 `complexity`。单字母只保留约定俗成的 $p$（程序）、$f$（适应度）、$\tau$（轨迹）、$\Delta$（有向差分）。

### 3.1 程序节点

每个候选程序对应推导图中的一个节点：

$$
p_i=(c_i,\mathrm{idea}_i,f_i,\mathrm{complexity}_i,\mathrm{runtime}_i),
$$

| 符号                      | 含义                                                                         | 代码字段       |
| ------------------------- | ---------------------------------------------------------------------------- | -------------- |
| $c_i$                   | 源代码                                                                       | `code`       |
| $\mathrm{idea}_i$       | 自然语言设计思想                                                             | `idea`       |
| $f_i$                   | 标量适应度；评估失败则为空                                                   | `fitness`    |
| $\mathrm{complexity}_i$ | 代码复杂度合成分（Shinka 风格：CC+Halstead+LOC+nesting；评估器也可显式返回） | `complexity` |
| $\mathrm{runtime}_i$    | 评估耗时（秒）                                                               | `runtime`    |

另存分项 `complexity_metrics`（cc / Halstead volume / loc / nesting 等）供上下文展示。评估失败程序**不入图、不建轨迹**。节点表示程序状态，不再强制压缩为单一机制类别。

### 3.2 推导边与有向信用

推导图是**单父 DAG**：每个子节点至多一条入边。从父程序生成子程序时增加边：

$$
e=(p_u,p_v,\mathrm{action},\mathrm{op},\Delta,\mathrm{outcome}),
$$

| 符号                 | 含义                                                 | 代码字段      |
| -------------------- | ---------------------------------------------------- | ------------- |
| $\mathrm{action}$  | 自然语言修改动作                                     | `action`    |
| $\mathrm{op}$      | 搜索算子名                                           | `operator`  |
| $\Delta$           | 有向适应度变化                                       | `delta`     |
| $\mathrm{outcome}$ | 结果类型（improve / regress / plateau / unknown）    | `outcome`   |
| —                   | 写入时的搜索 iteration（经验排序并列时优先较新记录） | `iteration` |

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

**逐步归因**：边只承担自身 $\Delta$；不把后代改进回传到祖先。路径潜力（§5.2）是对轨迹上逐步变化的加权统计，不是树搜索式的 backup。Novelty Jump 没有父边，不进入 action 经验检索；其成败仍由 OperatorPortfolio 按整个算子更新。

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

### 3.4 经验记忆

`ExperienceMemory` 是 `DerivationGraph` 上的只读经验视图。它不复制图边，也不维护第二份事实；外部界面只有一个查询：

```python
examples(*, operator: str, positive_k: int = 2, negative_k: int = 2) -> ExperienceBatch
```

查询规则：

1. 仅检索 action 非空的 refinement 边；
2. 成功例来自 `outcome=improve`，失败例来自 `outcome=regress`；不注入 plateau；
3. 优先返回当前 operator 的记录；不足时用其它 operator 的全局记录补齐；
4. 成功例按有向 $\Delta$ 降序，失败例按 $\Delta$ 升序（最强退步优先）；并列时优先较新 iteration；
5. 对规范化后完全相同的 action 文本全局去重；若同一 action 同时有成功和失败记录，优先保留当前 operator 的记录，再取绝对 $\Delta$ 最大者，并列时取较新记录；不做语义聚类；
6. 返回结构化 `ExperienceExample`，prompt 格式化由 `context.py` 负责。

默认至多注入 $2$ 条成功与 $2$ 条失败 action，每条 action 最多保留 $300$ 个字符；action 调用日志记录经验块字符数。这些示例是当前 run 内的任务内经验，但其产生和检索不依赖任务词表。

短期轨迹最近 $5$ 步因果叙事仍是 refinement 的主上下文；边级经验块是有界补充，不引入额外 LLM 调用或 embedding 模型。

---

## 5. 轨迹价值与选择

价值计算的对象是**唯一活跃轨迹**（同 `path_key` 只留一个代表）。图中节点均由成功评估产生；若防御性遇到 `fitness is None`，对应全零 $V$。

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

不使用外部 embedding。两种 Jaccard 相似度：

1. **程序相似度**：去注释与多余空白后，对标识符 token 集合做 Jaccard；
2. **轨迹行为相似度**：边上 `(operator|outcome)` 指纹集合的 Jaccard。

组合相似度（默认权重 $0.7,0.3$）：

$$
\mathrm{sim}(\tau_a,\tau_b)=
\frac{w_c\mathrm{sim}_{code}+w_t\mathrm{sim}_{traj}}{w_c+w_t}.
$$

对目标轨迹，相对其它唯一活跃路径：

$$
D(\tau)=1-\mathrm{mean}_{\tau'\ne\tau}\mathrm{sim}(\tau,\tau'),\qquad
N(\tau)=1-\max_{\tau'\ne\tau}\mathrm{sim}(\tau,\tau').
$$

无其它轨迹时 $D=N=1$。二者都来自同一相似度体系：$D$ 强调平均疏离，$N$ 强调相对最近邻的疏离；生存阶段的 Pareto 截断会同时保留这两维。

### 5.4 紧凑性 C 与速度 R

这里的复杂度是**候选程序的静态结构复杂度**，不是算法的渐进时间复杂度。默认用 Radon 与 AST 从完整候选代码中提取四项指标：

- $CC$：所有代码块的圈复杂度之和，反映分支与独立控制路径；
- $H$：Halstead volume，反映运算符与操作数构成的计算信息量；
- $LOC$：物理代码行数，反映程序规模；
- $N$：`FunctionDef/If/For/While/With/Try` 等结构的 AST 最大嵌套深度。

先对量纲做固定尺度归一化：

$$
\widehat{CC}=\frac{CC}{10},\qquad
\widehat{H}=\frac{\log_2(H+1)}{10},\qquad
\widehat{LOC}=\frac{\log_2(LOC+1)}{10},\qquad
\widehat{N}=\frac{N}{5}.
$$

再得到节点上保存的原始复杂度合成分：

$$
\mathrm{complexity}(p)=
\min\!\left(
0.4\widehat{CC}+0.4\widehat{H}+0.1\widehat{LOC}+0.1\widehat{N},
1
\right).
$$

$CC$ 与 Halstead 各占 $40\%$，使指标主要惩罚过多控制路径和计算结构；$LOC$ 与嵌套深度各占 $10\%$，作为规模与可读性约束，避免把“代码较长”简单等同于“机制较差”。对 $H$ 和 $LOC$ 使用对数压缩，防止长代码中的极端值主导合成分；最后截断到 $1$，保持节点间可比。评估器显式返回正复杂度时优先使用该值；若 Radon 分析异常，则用 AST 节点数与行数构造可比的退化分数。

原始合成分表达一个程序自身的结构，但搜索更关心它相对当前候选池是否紧凑。因此再对唯一活跃终点的 raw $\mathrm{complexity}$ / $\mathrm{runtime}$ 使用与 $Q$ 相同的 10%/90% 分位裁剪，得到 $(c_{\min},c_{\max})$、$(r_{\min},r_{\max})$，再翻转为「越大越好」：

$$
C(\tau)=1-\mathrm{clip}\!\left(\frac{\mathrm{complexity}(p_L)-c_{\min}}{c_{\max}-c_{\min}},0,1\right),\qquad
R(\tau)=1-\mathrm{clip}\!\left(\frac{\mathrm{runtime}(p_L)-r_{\min}}{r_{\max}-r_{\min}},0,1\right).
$$

缺失、非正值或池内无可用差异时取中性 $0.5$。这种“固定公式合成 raw complexity，再按活跃池转成 compactness”的两层设计，既保留结构度量的稳定含义，又让搜索压力适应当前候选集。runtime 取评估 wall-clock（MEoH 效率副目标）；它与结构复杂度正交，因为代码结构复杂不必等于实际执行较慢。

### 5.5 标量化与 Trajectory-UCB

价值向量（质量 / 潜力 / 多样性 / 新颖性 / 紧凑性 / 速度）：

$$
V(\tau)=(Q,P,D,N,C,R)
=(V_{\mathrm{quality}},V_{\mathrm{potential}},V_{\mathrm{diversity}},V_{\mathrm{novelty}},V_{\mathrm{compactness}},V_{\mathrm{speed}}).
$$

采样用加权标量加探索项：

$$
S(\tau)=w_qQ+w_pP+w_dD+w_nN+w_cC+w_rR+U(\tau),
$$

默认 $(w_q,w_p,w_d,w_n,w_c,w_r)=(0.42,0.18,0.12,0.12,0.08,0.08)$。

$$
U(\tau)=c_t\sqrt{\frac{\log(V_{\mathrm{all}}+1)}{n_\tau+1}},
$$

其中 $n_\tau$ 为该轨迹访问次数，$V_{\mathrm{all}}$ 为唯一活跃轨迹总访问次数。系数

$$
c_t=\max\!\big(c_{\mathrm{floor}},\,c_0(1-t/T)\big)
+\beta\cdot\min(s_{\mathrm{stag}}/T,1),
$$

默认 $c_0=0.4$，$c_{\mathrm{floor}}=0.05$，$\beta=0.20$；$t$ 为当前 search iteration，$T$ 为计划总 iteration，$s_{\mathrm{stag}}$ 为全局 best 连续未更新的停滞计数。随着搜索推进，基础探索衰减，但停滞时会重新抬高探索。

### 5.6 选择流程

默认策略 `trajectory_ucb` 不是全局贪心：

1. 对所有唯一活跃轨迹计算 $V$ 与 $S$，写回轨迹记忆；
2. 取标量最高的 top-k（默认 $k=12$）；
3. 每个 island 再并入局部 top（默认每岛 $1$ 条），避免小岛被全局 top 完全淹没；
4. 并入当前全局 best endpoint 对应的 elite 轨迹（同 endpoint 多条时取访问次数最少者）；
5. 以概率 $0.15$ 直接返回 elite；否则在候选集上按温度 $T_{\mathrm{sel}}=0.8$ 做 softmax 采样。

可选对照策略：`best`（按终点适应度选最优）、`random`（均匀抽唯一活跃轨迹）。实验默认使用 `trajectory_ucb`。

---

## 6. 自适应算子组合

每轮先构造 `OperatorContext`（推导图、轨迹记忆、经验记忆、island 管理器、当前选中轨迹、maximize、停滞计数、算子侧信道 hints），再由 OperatorPortfolio 在**通过可行性 trigger 的算子**中采样。若无一算子触发，回退到 Endpoint Refine。

五个默认算子及其角色：

| 算子                | 名称                    | role         | 意图                          |
| ------------------- | ----------------------- | ------------ | ----------------------------- |
| Endpoint Refine     | `endpoint_refine`     | exploit      | 沿当前终点继续局部强化        |
| Backtrack Branch    | `backtrack_branch`    | path_correct | 从高价值内部前缀分叉          |
| Mechanism Crossover | `mechanism_crossover` | recombine    | 把互补算法思路迁入当前程序    |
| Simplify            | `simplify`            | simplify     | 在不降适应度前提下降复杂度    |
| Novelty Jump        | `novelty_jump`        | explore      | 开放式 fresh start            |

统一协议：

```text
trigger → (可选 select_trajectory 覆盖选题) → select_base → build_constraint
→ 主循环生成与评估 → insert
```

### 6.1 Endpoint Refine（exploit）

**触发**：始终可行（图中节点均已评估合法）。

**Base**：当前 endpoint。

**约束**：沿最近有效方向继续做一次针对性强化，避开已知退步方向。

**接入**：`extend`（endpoint extension）。

### 6.2 Backtrack Branch（path_correct）

该算子**不依赖** UCB 刚选中的轨迹。实现上主动扫描活跃池，找存在内部前缀的多步轨迹：

1. 跳过无边（长度 $1$）轨迹；
2. 用四规则 + `branch_score` 选出不同于 endpoint 的内部 base；
3. 在满足条件的轨迹中取 `branch_score` 最高者，替换 `ctx.selected`。

**触发（唯一硬门槛）**：活跃池中至少存在一条长度 $\ge 2$、且能选出 $\mathrm{base}\ne\mathrm{endpoint}$ 的轨迹。

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

保留算子名 `mechanism_crossover` 以兼容日志口径；donor 选择不再读取机制统计或反模式。

**触发**：始终可行。初始化不做 crossover；初始化完成后活跃池必有其他轨迹可供重组。

**Donor 选择**：对其他活跃轨迹按互补性与质量软排序，取最高分者（不因门槛硬禁用算子）：

$$
\begin{aligned}
\mathrm{comp}&=1-(0.7\,\mathrm{sim}_{code}+0.3\,\mathrm{sim}_{traj}),\\
\mathrm{donor\_score}&=\mathrm{comp}+0.3Q.
\end{aligned}
$$

$Q$ 优先用已缓存质量，否则用活跃终点适应度范围归一化。

**Base**：recipient 的 endpoint；donor idea 写入 hints，供约束文本使用。

**约束**：只移植一个明确的算法思路，保留 base 程序其余结构；不再提供伪机制族名称。

**接入**：`branch_from`。

### 6.4 Simplify（simplify）

**触发（可行性）**需同时满足：

1. 活跃有效终点至少 $2$ 个，且当前复杂度 $\gt 0$；
2. 当前复杂度处于活跃终点复杂度上四分位及以上，且高于中位数。

不再要求末步 plateau 或全局停滞；调用频率由 portfolio 历史收益与阶段 bonus（late 偏好 simplify）调节。

**约束**：删除低贡献代码，保留产生收益的核心思路，在不降适应度前提下降低复杂度。

**接入**：`extend`。

### 6.5 Novelty Jump（explore）

**触发**：始终可行。不再使用全局停滞门槛或算子冷却硬门控；探索频率由 portfolio EMA、阶段 bonus 与 late novelty 概率上限调节。

不再依赖预设机制族、Beta 后验选择、family 冷却或 anti-pattern 过滤；因此不存在「所有 family 被禁用」的死路。

**Base**：无（fresh start）。

**约束**：要求一个与当前活跃精英 idea 明显不同的完整方案，并可列出最多 $4$ 个已有 idea 作为避免重复的参考。

**生成**：不走动作阶段，直接用 initial-style prompt，将算子约束作为 diversity hint。

**接入**：`create_initial`；新轨迹分配到当前活跃轨迹最少的 island，并列时选编号最小者。生成后仍由程序/轨迹相似度 novelty gate 决定是否保留。

---

## 7. Operator Portfolio

### 7.1 候选与采样

候选 = 当前 `OperatorContext` 下可行性 `trigger` 为真的算子；若为空，强制 Endpoint Refine。Endpoint / Crossover / Novelty 默认始终在候选中；Backtrack 与 Simplify 仅在结构性可行时加入。

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

near-record：相对本轮开始时的 incumbent，有向短差不低于 $-0.10\times$ 适应度尺度。

---

## 8. 上下文构造与程序生成

### 8.1 初始化

在评估预算内生成 $n_{\mathrm{init}}$（默认 $4$）个初始程序。第一个候选要求一个简单、完整的有效方案；后续候选在 prompt 中列出已生成的简短 idea，要求使用明显不同的算法思路。这只是 run-local 去重，不预设任务机制。走 initial-style prompt：任务描述 + 多样性提示 + 目标函数契约，要求输出：

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
2. 当前轨迹最近至多 $5$ 步因果叙事（parent→child、算子、动作、fitness/complexity/runtime 变化、outcome）；
3. `[Past Action Evidence]`：至多 $2$ 条成功与 $2$ 条失败 action（operator、截断后的 action、有向 $\Delta$）；空记忆时给简短空状态；
4. RankingModel 的 best vs worst 对比反馈（仅 idea 与 fitness）；
5. 算子名、role 与约束文本；
6. base 节点的 idea、结构/runtime 摘要、代码与选择原因；
7. 目标函数契约；
8. 指令：提出恰好 `actions_per_iteration`（默认 $2$）条编号修改，每条只改一个主算法思路，并避免无必要的复杂度/runtime 膨胀；禁止输出代码与解释。

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

1. 写入新节点（代码、idea、fitness、complexity、runtime）；
2. 写入父→子边（动作、算子、$\Delta$、outcome、iteration）；
3. 按算子 `insert` 语义扩展 / 分叉 / 新建轨迹；
4. 用父子适应度更新 Elo 风格相对排名；
5. 更新全局 `best_node`；
6. 执行 novelty gate。

Novelty Jump 候选无父边，直接建初始轨迹，因此不进入 action 经验检索。

### 9.2 新颖性门控

新轨迹进入活跃池前检查：与其它活跃轨迹的组合最大相似度是否 $\ge 0.92$。若命中且该候选**未**刷新当前 live global best，则立即归档该轨迹。若刷新了 global best，则质量覆盖，强制保留。门控只影响轨迹生存，不删除图节点 / 边。通过门控后为该轨迹计算并写入 $V$。

### 9.3 相对排名模型

`RankingModel` 对每次父子比较做 Elo 更新（$K=16$），并维护比较图的连通分量。best/worst 对比：

1. 取最近 window（默认 $20$）条活跃轨迹的有效终点；
2. 先按 raw fitness 选出最优 / 最差分量；
3. 分量内再按 Elo 分数（并列时参考 raw fitness）取 best / worst。

未连通分量之间不直接比较 Elo。该模型服务于 refinement prompt 对比块，**不**进入 `OperatorContext`，也不直接决定轨迹 UCB 选择。

---

## 10. 周期性维护

每个**完成的** search iteration 执行 `_periodic_hooks`。停滞计数在 completed iteration 上更新：best 未变则 $+1$，否则清零。

### 10.1 轨迹生存（每轮）

1. 归档重复 `path_key`，只留代表轨迹；
2. 重算全部活跃轨迹的 $V$ 与标量价值；
3. 保护至少一条到达当前 global best endpoint 的轨迹（多条时取 Pareto 序中的第一条）；
4. 各 island 内按 ValueVec 的非支配层排序，同层内按标量价值降序，截断到 `max_per_island=40`；
5. 全局再截断到 `max_active_trajectories`（默认 $n_{\mathrm{islands}}\times\max_{\mathrm{per\_island}}=160$）。

Pareto 支配：六维价值均 $\ge$ 且至少一维严格 $\gt$。归档不删图。对外返回仍是单一 `best_node`，Pareto 只服务于活跃池管理。

### 10.2 Island 迁移（停滞触发）

当停滞 $\gt 0$ 且 completed iteration 为 `migration_interval=20` 的倍数时，每个 island 将其标量价值最高的 $1$ 条轨迹**移动**到下一 island（环形）。迁移保持轨迹 ID、访问次数与价值，不复制新身份。

Island 分配规则：

- 初始化：`slot % n_islands`；
- Novelty Jump：当前活跃轨迹最少的 island，并列时选编号最小者。

已删除周期机制聚合与对比反思回路；经验检索始终是对当前图边的即时只读查询。

---

## 11. 完整搜索算法

**输入**：LLM、程序模板、task evaluator、最大评估预算 $B$。

**输出**：标量最优 `best_node`。

```text
1.  用 run-local idea 去重提示生成并评估至多 n_init 个初始程序
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
12.         构造因果/边级经验/对比上下文，生成 A 条动作
13.         对每条动作生成完整程序并评估
14.         写节点、边、Δ、outcome，按算子语义更新轨迹
15.     novelty gate；更新 best、Elo
16.     用本批最优候选更新 portfolio 一次；记录选中轨迹访问 +1
17.     若完成新的 search iteration:
18.         更新 stagnation；执行 survival
19.         按停滞条件执行 migration
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
| migration 周期            |      20 iterations |
| evaluator worker          |                  1 |
| 随机种子                  |                  0 |
| 连续停滞停止              |        20 attempts |

### 12.2 价值与选择

| 配置                                                 |                              默认值 |
| ---------------------------------------------------- | ----------------------------------: |
| $(w_q,w_p,w_d,w_n,w_c,w_r)$                        | $(0.42,0.18,0.12,0.12,0.08,0.08)$ |
| 相似度权重$(w_c,w_t)$                              |                       $(0.7,0.3)$ |
| 路径折扣$\gamma$                                   |                                 0.8 |
| $(\lambda_{\mathrm{pos}},\lambda_{\mathrm{down}})$ |                      $(0.25,0.5)$ |
| 潜力质量门控$q_{\min}$                             |                                 0.5 |
| 适应度裁剪分位                                       |                                0.10 |
| UCB$(c_0,c_{\mathrm{floor}},\beta)$                |                 $(0.4,0.05,0.20)$ |
| top-k / island top                                   |                              12 / 1 |
| elite 直采概率                                       |                                0.15 |
| 选择 softmax 温度$T_{\mathrm{sel}}$                |                                 0.8 |

### 12.3 Portfolio 与算子内置阈值

| 配置                                                        |                   默认值 |
| ----------------------------------------------------------- | -----------------------: |
| EMA 衰减                                                    |                      0.8 |
| $(\alpha,\beta_v,\beta_n,\delta_r,\delta_c)$              | $(1,0.5,0.3,0.5,0.05)$ |
| 成本尺度$C$                                               |                      120 |
| $(b_{\mathrm{gb}},b_{\mathrm{nr}})$                       |          $(0.75,0.25)$ |
| near-record 容差                                            |                     0.10 |
| $(T_{\mathrm{init}},T_{\mathrm{end}},T_{\mathrm{floor}})$ |        $(1.0,0.5,0.5)$ |
| 最小算子概率 / late novelty cap                             |               0.05 / 0.2 |
| Simplify 相对复杂度门槛                                     |     上四分位且高于中位数 |
| Crossover donor 排序                                        |        $\mathrm{comp}+0.3Q$ |
| 经验块成功/失败条数                                         |                    2 / 2 |
| 经验 action 截断字符数                                      |                      300 |
| Elo$K$ / contrast window                                  |                  16 / 20 |

---

本文描述的是当前代码实际执行的搜索机制，可作为后续改动的对照基线。
