# TraceAAD：基于算法改进轨迹的自动算法设计

> 权威机制说明。对齐当前 `llm4ad.method.traceaad` 实现（含精英课程、经验记忆与机会感知 portfolio）。公式、触发条件与默认超参以代码为准。

---

## 0. 导读：先给出结论

**TraceAAD 要解决的问题**：在有限评估预算下，用 LLM 搜索更好的算法程序时，不要只盯着「当前最好的那份代码」，而要把「这次改动从哪来、改了什么、变好还是变坏」当作一等搜索资产。

**核心主张（一句话）**：

> 有界算法改进轨迹是搜索与生成的共同枢纽；图保存事实，轨迹决定预算投向，经验与课程只指导 LLM 如何改，不伪造搜索状态。

**方法由四层决策组成（金字塔从上到下）**：

1. **投向哪条路**（轨迹选择）：在活跃轨迹上比较多维价值，用 Trajectory-UCB 分配下一次扩展机会。
2. **用什么改法**（算子 + portfolio）：在结构性可行算子上按历史收益与上下文 soft bonus 采样，硬门控只排除不可行，不做探索策略硬编码。
3. **给 LLM 看什么**（上下文）：短期因果叙事 + 边级成败经验 + 精英课程（事实 / 推断分层）+ best–worst 对比。
4. **改完如何记账**（注册与维护）：写图与边、新颖性门控、课程反馈、portfolio 更新；周期性做 Pareto 生存与岛迁移。

读完本文应能回答：TraceAAD 与「选一个程序改一次」有何本质不同；一次 search attempt 逐步发生了什么；事实、推断、指令如何隔离；默认超参落在何处。实现细节（续训、日志文件）见第 11 节。

---

## 1. 问题设定与设计动机

### 1.1 问题设定

给定：

- 待设计算法任务（如 TSP constructive heuristic、CVRP ACO prior、OP constructive）；
- 可执行程序模板（唯一可演化函数签名与输出契约）；
- 大语言模型（LLM）；
- 程序评估器（对训练集实例返回标量适应度 $f$）。

TraceAAD 在评估预算 $B$ 内搜索适应度最优的有效程序。最大化任务直接比较 $f$；最小化任务在比较、归一化与有向差分时反转方向。训练评估通常对多实例取平均；held-out 测试不反馈进搜索。

### 1.2 为什么现有范式不够

许多 LLM 驱动的算法设计方法以**单个程序**或**树节点**为搜索对象：选出一个程序、改一次、评估一次。这样做丢掉了过程信息——哪一步改进、哪一步退步、用了什么算子、属于哪条改进路径。而这些信息恰恰回答两件不同的事：

| 问题 | 需要的信息 |
| --- | --- |
| 下一步预算投向哪里？ | 哪条**真实路径**仍值得继续扩展 |
| 下一步该怎么改代码？ | 哪些**修改结构**值得模仿、组合或避免 |

把「路径形状」直接当成「未来潜力」，或把稀缺成功事件与大量 plateau/regress 混在同一价值里，都会产生噪声。TraceAAD 因此把轨迹升格为一等对象，并把**搜索价值**与**教学证据**分开处理。

### 1.3 与本仓库其它方法的概念对照

| | 典型程序级进化 / EoH 式 | MCTS-AHD（本仓库） | PathWise（本仓库） | **TraceAAD** |
| --- | --- | --- | --- | --- |
| 搜索对象 | 种群中的个体程序 | 树节点 + 选择策略 | 路径/种群混合（实现依赖） | **有界改进轨迹** |
| 信用 | 个体适应度 | 常含树备份 / 节点统计 | 路径相关 | **边级逐步 $\Delta$，不回传祖先** |
| 生成上下文 | 精英代码 / 简单对比 | 节点局部信息 | 路径信息 | **因果叙事 + 经验边 + 精英课程** |
| 探索机制 | 变异/交叉比例 | UCT 等 | 方法内建 | **角色化算子 + portfolio 概率** |

对照目的是澄清定位，不是宣称数值优势。数值比较见 `docs/results/`。

### 1.4 四条设计原则

1. **过程信用逐步归因，不回传祖先。** 每条推导边只记录该步自己的有向 $\Delta$；后代改进不回传到更早节点（刻意避免 MCTS 式 max-backprop 的过度归功）。
2. **价值多维，采样与生存用途分离。** 轨迹价值 $V=(Q,P,D,N,C,R)$ 在采样时标量化并加 UCB；在活跃池截断时用 Pareto 非支配排序，避免六维塌成单一分数后过早丢掉多样性与效率折中。
3. **算子角色化，由 portfolio 自适应调度。** Trigger 只排除结构性不可行；探索/利用频率由历史收益、上下文 bonus 与 $\epsilon$-mixture 决定，不写成大段硬 if-else。
4. **事实、推断、指令分层。** DAG 节点与边是 `fact`；Champion / improve-chain / repair / recombine 是 `inference`；LLM 新生成的修改是 `instruction`。课程可以影响生成，但不能冒充真实父子关系，也不进入 active pool。

---

## 2. 方法总览

### 2.1 四层记忆

| 层 | 实现 | 存什么 | 角色 |
| --- | --- | --- | --- |
| 程序记忆 | `DerivationGraph` | 已评估程序节点 + 单父推导边 | 事实库；轨迹归档不删图 |
| 轨迹记忆 | `TrajectoryMemory` | 仍值得投入预算的有界路径 | 选择与扩展的直接对象 |
| 经验记忆 | `ExperienceMemory` | 图边上的只读成功/失败 action 视图 | 指导 refinement 动作 |
| 课程记忆 | `EliteCurriculum` | 从 best 事件与 DAG 构造的有界课程 | 生成参照；不进活跃池 |

互补关系：**图**回答「发生过什么」；**轨迹**回答「现在跟哪条路」；**经验**提供单步 action 证据；**课程**把稀缺成功结构组织成带证据等级的生成材料。

### 2.2 两条时间尺度上的回路

| 回路 | 做什么 | 何时 |
| --- | --- | --- |
| 进化主回路 | 选轨迹 → 选算子 → 组上下文 → 生成 → 评估 → 写图/轨迹 → novelty → 更新 portfolio / 课程 | 每次有预算的 search attempt |
| 周期性维护 | Pareto 生存；停滞时岛迁移；写 checkpoint | 每个 completed search iteration |

RankingModel（Elo）在每次 refinement 的父子比较后更新，服务于 prompt 中的 best–worst 对比，**不**直接进入轨迹 UCB。

### 2.3 主循环数据流

```text
初始化
  run-local idea 去重提示 → 生成并评估至多 n_init 个程序
  → 建图与长度-1 轨迹 → slot % n_islands 安置
        │
        ▼
主循环（预算未尽且未中止）
  1. 轨迹选择（默认 Trajectory-UCB）
  2. Portfolio 在可行算子中采样（含 OperatorPreview）
  3. 算子可覆盖选题（Backtrack）并确定 base + 约束
  4. 非 fresh-start：组装 CurriculumPacket
  5. Novelty：initial-style；否则两阶段 refinement
  6. 评估 → 写节点/边 → 按算子语义插入轨迹
  7. Novelty gate；更新 best、Elo；课程 packet outcome
  8. 本批最优观测更新 portfolio 一次；选中轨迹 visit+1
  9. 若完成新的 search iteration：
       更新停滞；survival；按需 migrate；checkpoint
        │
        ▼
末尾再 survival → 返回标量最优 best_node
```

下文按「状态 → 选择 → 改法 → 生成 → 记账」展开。

---

## 3. 搜索状态表示

符号尽量与代码字段同名。单字母约定：$p$ 程序、$f$ 适应度、$\tau$ 轨迹、$\Delta$ 有向差分。

### 3.1 程序节点

$$
p_i=(c_i,\mathrm{idea}_i,f_i,\mathrm{complexity}_i,\mathrm{runtime}_i).
$$

| 符号 | 含义 | 字段 |
| --- | --- | --- |
| $c_i$ | 源代码 | `code` |
| $\mathrm{idea}_i$ | 自然语言设计思想 | `idea` |
| $f_i$ | 标量适应度 | `fitness` |
| $\mathrm{complexity}_i$ | 结构复杂度合成分（§3.1.1） | `complexity` |
| $\mathrm{runtime}_i$ | 评估耗时（秒） | `runtime` |

评估失败的程序**不入图、不建轨迹**。另存 `complexity_metrics` 供展示。

#### 3.1.1 复杂度合成分

默认用 Radon + AST 提取：$CC$（圈复杂度之和）、$H$（Halstead volume）、$LOC$、$N$（结构最大嵌套）。归一化后：

$$
\mathrm{complexity}(p)=\min\!\big(
0.4\widehat{CC}+0.4\widehat{H}+0.1\widehat{LOC}+0.1\widehat{N},\,1\big),
$$

其中 $\widehat{CC}=CC/10$，$\widehat{H}=\log_2(H+1)/10$，$\widehat{LOC}=\log_2(LOC+1)/10$，$\widehat{N}=N/5$。评估器显式返回正复杂度时优先使用；分析失败则退化为 AST/行数代理。这是**静态结构**度量，不是渐进时间复杂度。

### 3.2 推导边与有向信用

推导图是**单父 DAG**。边：

$$
e=(p_u,p_v,\mathrm{action},\mathrm{op},\Delta,\mathrm{outcome}).
$$

$$
\Delta=
\begin{cases}
f_{\mathrm{child}}-f_{\mathrm{parent}}, & \text{最大化},\\
f_{\mathrm{parent}}-f_{\mathrm{child}}, & \text{最小化}.
\end{cases}
\qquad
\mathrm{outcome}=
\begin{cases}
\mathrm{improve}, & \Delta>\varepsilon,\\
\mathrm{regress}, & \Delta<-\varepsilon,\\
\mathrm{plateau}, & |\Delta|\le\varepsilon,\\
\mathrm{unknown}, & \Delta=\mathrm{null},
\end{cases}
$$

$\varepsilon=10^{-6}$。**逐步归因**：边只承担自身 $\Delta$；路径潜力（§4.2）是轨迹上逐步变化的加权统计，不是树备份。Novelty Jump 无父边，不进入 action 经验检索；其算子级成败仍由 portfolio 更新。

### 3.3 有界轨迹

$$
\tau=(p_0,e_1,p_1,\ldots,e_L,p_L).
$$

关键字段：`node_ids` / `edge_ids`、`base_id`、`endpoint_id`、`island_id`、`visit_count`、`status`（`active`/`archived`）、`value` / `scalar_value`。

默认 $L_{\max}=8$。超限则**滑动窗口**保留最近后缀。扩展语义：

| 语义 | 含义 | 算子 |
| --- | --- | --- |
| endpoint extension | 从终点追加 | Endpoint、Simplify |
| prefix branching | 截内部前缀再接新子节点（新轨迹 ID） | Backtrack、Crossover |
| fresh start | 无父边，长度 1 | Novelty、初始化 |

约定：扩展/分叉**总是新建轨迹对象**；归档不删图；`path_key=(node_ids,edge_ids)` 用于活跃池去重（同路径保留 `visit_count` 更高者，并列保留 id 更大者）。

### 3.4 经验记忆

`ExperienceMemory` 是图上的只读视图，界面为：

```text
examples(operator, positive_k=2, negative_k=2) → ExperienceBatch
```

规则：

1. 只取 action 非空、有 $\Delta$、且 `outcome\in\{\mathrm{improve},\mathrm{regress}\}$ 的边（不注入 plateau）；
2. 优先同 operator 记录，不足再用其它算子补齐；
3. 成功按 $\Delta$ 降序，失败按 $\Delta$ 升序（最强退步优先）；并列时优先较新 `iteration`；
4. 对规范化空白后的 action 文本**全局去重**：同一文本只留一条代表边。并列优先级为：当前 operator 匹配 > $|\Delta|$ 更大 > `iteration` 更新 > `edge.id` 更大；若同一 action 既有成功又有失败记录，仍按该代表规则择一，**不做语义聚类**；
5. 返回结构化 `ExperienceExample`，prompt 格式化由 `context.py` 负责；每条 action 默认截断 300 字符。

短期轨迹最近 5 步因果叙事仍是 refinement 的主上下文；边级经验块是有界补充，**无额外 LLM / embedding**。

### 3.5 精英课程记忆

`EliteCurriculum` 从事实构造生成参照，**不创建**节点、边或 active trajectory。维护：

1. `ChampionEvent`：每次 global best 刷新时追加（保留最近 `max_champion_events=4` 用于组装）；
2. packet 级 `_usage` / `_reward`：课程被注入后按 offspring 结果更新，**不回写**边上的 $\Delta$/`outcome`。

每次 `build()` 时从 DAG **即时派生**课程 traces（无单独的离线 refresh）：

| 类型 | 构造规则（代码） | `causal_status` | 默认 `(confidence, causal_coherence)` |
| --- | --- | --- | --- |
| `champion` | 最近至多 4 个 `ChampionEvent` 逐步标注 | `jump` | $(0.9, 0.0)$ |
| `improve_chain` | 从每条 `improve` 边向上回溯连续 improve 父边，链长上限 3 | `direct` | $(0.85, 1.0)$ |
| `prefix_repair` | 由 `regress` 边生成；`terminal_node_id` 指向**父节点（高价值前缀）**，不是退步 endpoint | `prefix` | $(0.75, 0.8)$（子无 fitness 时 confidence $0.5$） |
| `contrastive` | 每条 regress 配一条 improve 链（优先同 operator，否则按 `_trace_score` 全局最优） | `composed` | $(0.45, 0.4)$ |
| `elite_recombine` | `_trace_score` 最高的两条 improve 链，优先末步 operator 不同，各取末步拼成 composed | `composed` | $(0.4, 0.0)$ |

接口：`record_best_event` / `build(...)` → `CurriculumPacket` / `record_outcome`。`V_search` 只服务真实轨迹；教学效用不写入 `Trajectory.value`。按算子组装见 §5.3。

---

## 4. 轨迹价值与选择：预算投向哪里

价值对象是**唯一活跃轨迹**（同 `path_key` 一个代表）。防御性若遇 `fitness is None`，则 $V$ 全零。

### 4.1 终点质量 $Q$

用活跃终点适应度的 10%/90% 分位裁剪得 $(f_{\min},f_{\max})$，再线性归一：

$$
Q(\tau)=\mathrm{clip}\!\left(\frac{f(p_L)-f_{\min}}{f_{\max}-f_{\min}},0,1\right).
$$

最小化任务在归一化前对适应度与边界取负。边界退化时返回约定中性值（见代码）。

### 4.2 路径潜力 $P$

令逐步归一化质量变化为 $\Delta q_i$，近端权重 $\gamma^{n-i-1}$（默认 $\gamma=0.8$）：

$$
R(\tau)=
\frac{\sum_i w_i\Delta q_i}{\sum_i w_i}
+\lambda_{\mathrm{pos}}\frac{1}{n}\sum_i\mathbb{I}(\Delta q_i>\varepsilon)
-\lambda_{\mathrm{down}}\frac{\sum_i w_i\max(-\Delta q_i,0)}{\sum_i w_i},
$$

默认 $\lambda_{\mathrm{pos}}=0.25$，$\lambda_{\mathrm{down}}=0.5$。再用质量门控：

$$
P(\tau)=
\begin{cases}
0, & Q\le q_{\min},\\
R\cdot(Q-q_{\min})/(1-q_{\min}), & Q>q_{\min},
\end{cases}
\quad q_{\min}=0.5.
$$

### 4.3 多样性 $D$ 与新颖性 $N$

无外部 embedding。两种 Jaccard：

1. **程序**：去注释与多余空白后，标识符 token 集合 Jaccard；
2. **轨迹行为**：边上 `(operator|outcome)` 指纹集合 Jaccard。

$$
\mathrm{sim}= \frac{w_c\mathrm{sim}_{code}+w_t\mathrm{sim}_{traj}}{w_c+w_t},
\quad
D=1-\mathrm{mean}_{\tau'\ne\tau}\mathrm{sim},\quad
N=1-\max_{\tau'\ne\tau}\mathrm{sim}.
$$

默认 $(w_c,w_t)=(0.7,0.3)$。无其它轨迹时 $D=N=1$。

### 4.4 紧凑性 $C$ 与速度 $R$

对活跃终点 raw complexity / runtime 做与 $Q$ 相同的分位裁剪，再翻转为「越大越好」：

$$
C=1-\mathrm{clip}(\mathrm{norm}(\mathrm{complexity})),\qquad
R=1-\mathrm{clip}(\mathrm{norm}(\mathrm{runtime})).
$$

缺失或无差异时取 $0.5$。runtime 是评估 wall-clock，与结构复杂度正交。

### 4.5 标量化与 Trajectory-UCB

$$
V=(Q,P,D,N,C,R),\qquad
S=w^\top V+U(\tau),
$$

默认权重 $(0.42,0.18,0.12,0.12,0.08,0.08)$。

$$
U=c_t\sqrt{\frac{\log(V_{\mathrm{all}}+1)}{n_\tau+1}},\quad
c_t=\max\!\big(c_{\mathrm{floor}},\,c_0(1-t/T)\big)+\beta\cdot\min(s_{\mathrm{stag}}/T,1).
$$

默认 $c_0=0.4$，$c_{\mathrm{floor}}=0.05$，$\beta=0.20$。推进时基础探索衰减，停滞时抬高。

### 4.6 选择流程（`trajectory_ucb`）

1. 对唯一活跃轨迹计算 $V$ 与 $S$；
2. 取标量 top-$k$（默认 $12$）；
3. 每岛并入局部 top（默认 $1$）；
4. 并入全局 best endpoint 对应 elite 轨迹；
5. 以概率 $0.15$ 直采 elite；否则温度 $T_{\mathrm{sel}}=0.8$ 的 softmax。

对照策略：`best`、`random`。实验默认 `trajectory_ucb`。

### 4.7 搜索价值与教学效用分离

$V(\tau)$ 是 `V_search`，只服务预算与生存。课程检索使用独立的教学打分（实现为 `_trace_score`），**不**写入 `Trajectory.value`，也**不**改写边上的 $\Delta$/`outcome`：

$$
\begin{aligned}
s(e)=&\,G+0.25\,C+0.15\,\kappa+0.1\,N+0.05\,r-0.02\,u\\
&+[0.15\text{ if }e\text{ 与当前 base 相关}],
\end{aligned}
$$

其中 $G=$`quality_gain`，$C=$`causal_coherence`，$\kappa=$`confidence`，$N$ 为课程新颖性项，$r$/$u$ 为历史 reward/usage。排序时先取「任一步 operator 匹配当前算子」的子集，不足再 fallback 全局。

`record_outcome` 按 offspring 给 packet 内 traces 记 reward：`global_best=1.0`，`near_record=0.4`，`improve=0.2`，`plateau=-0.1`，否则（含 regress）`-0.4`。`primary_trace_id`（=`positive_traces[0]`）拿全额，其余 trace 仅拿 $0.25\times$ reward。

`near_record` 判定：相对 incumbent 的有向差 $\ge -\mathrm{tol}\times s_t$，默认 $\mathrm{tol}=0.10$，$s_t$ 为活跃池适应度的稳健尺度（10–90% 分位差与 median 相关）；同时供 portfolio 的 near-record EMA 使用。

---

## 5. 算子组合与 Portfolio：用什么改法

### 5.1 统一协议

构造 `OperatorContext` → Portfolio 在 `trigger` 为真的算子中采样 →（可选）覆盖选题 → `select_base` → `build_constraint` → 生成评估 → `insert`。若无一算子可行，强制 `endpoint_refine`。

| 算子 | 名称 | role | 意图 |
| --- | --- | --- | --- |
| Endpoint Refine | `endpoint_refine` | exploit | 强化当前终点 |
| Backtrack Branch | `backtrack_branch` | path_correct | 内部前缀分叉修复 |
| Mechanism Crossover | `mechanism_crossover` | recombine | 迁入一个互补动作 |
| Simplify | `simplify` | simplify | 降复杂度且不牺牲适应度 |
| Novelty Jump | `novelty_jump` | explore | fresh start |

### 5.2 各算子触发与语义

| 算子 | `trigger`（硬门槛） | Base / 接入 |
| --- | --- | --- |
| Endpoint | 恒真 | endpoint；`extend` |
| Backtrack | 存在长度≥2 且能选出 $\mathrm{base}\ne\mathrm{endpoint}$ 的活跃轨迹 | 扫描池内最高 `branch_score` 轨迹并覆盖选题；`branch_from` |
| Crossover | 恒真 | recipient endpoint；donor 软排序；`branch_from` |
| Simplify | 活跃有效复杂度样本≥2，当前复杂度>0，且 ≥上四分位并 >中位数 | endpoint；`extend` |
| Novelty | 恒真 | 无父；`create_initial` 到最少负载岛 |

**Backtrack base 候选**：endpoint；末步 regress 则退步父节点；连续两步 plateau 则最近 improve 子节点；内部历史最佳（若异于 endpoint）。打分：

$$
\mathrm{score}(v)=q(v)+0.3\cdot\mathrm{fwd\_improve}-0.3\cdot\mathrm{bwd\_regress}.
$$

**Crossover 的两路「donor」勿混淆**：

1. **Live donor**（算子侧）：从当前活跃池按
   $\mathrm{comp}=1-(0.7\mathrm{sim}_{code}+0.3\mathrm{sim}_{traj})$，
   $\mathrm{donor\_score}=\mathrm{comp}+0.3Q$
   选轨迹，其 idea 写入 `ctx.hints["donor_idea"]` 进入约束文本；
2. **课程 `donor_trace`**：来自历史 improve 链的 `elite_recombine` 组装，只作为 prompt 证据，**不是**上面的 live donor。

**Novelty**：initial-style 生成；不调用 `EliteCurriculum.build`。约束中的「避免重复」来自活跃终点按 `scalar_value`（缺省用 fitness）降序去重后至多 **4** 条 idea（`avoid_ideas`）。

### 5.3 课程包按算子组装（`build`）

仅当存在 `base_node_id`（refinement）时调用。默认至多 4 个 champion 事件、packet 内至多 2 条 positive traces。Novelty 不调用 `build`。

| 算子 | positive | 附加 |
| --- | --- | --- |
| endpoint / simplify | champion + improve_chain（至多 2） | contrast；**仅当** `stagnation>0` 时注入 repair |
| backtrack | champion + improve_chain（至多 1） | **必选** repair（若有）；contrast |
| crossover | champion + improve_chain（至多 1） | **donor_trace**；contrast |
| novelty | — | — |

`repair_trace` 注入条件精确为：`operator==backtrack_branch` **或** `stagnation>0`。`build()` 时对 packet 内 `trace_ids` 递增 `_usage`。

算子专属 `packet.instructions`（进入 `[Elite Curriculum]`）：backtrack 强调前缀修复、勿重复 regress action；crossover 强调只移植一个 donor idea；novelty（若出现）强调防重复；其余声明 elite trace 是证据而非保证规则。

### 5.4 OperatorPreview（上下文 soft bonus）

采样前为每个可行算子预计算 preview（真实 target/base 与有界 bonus，默认 $|\beta|\le 0.2$）。要点：

- Backtrack：末步 regress/plateau、深度差、`branch_score`；
- Endpoint：improve 连胜加分，regress 减分；
- Novelty：停滞且近期 novelty downside 不高、池多样性低时小幅加分；
- Crossover：有/无 donor；
- Simplify：复杂度分位满足 trigger 时加分。

全局停滞主要抬高 $\epsilon$，而不是无条件抬高 Novelty。

### 5.5 Portfolio 采样与信用

实现为机会感知：有界 utility、折扣 UCB、上下文 bonus、全局 $\epsilon$-mixture；晚期仍将 `novelty_jump` 概率上限截到 $0.2$（待消融确认是否保留）。

$$
\begin{aligned}
z_i&=\alpha\mu_i-\lambda_d d_i-\lambda_c c_i
+b_{\mathrm{gb}}\widehat{\mathrm{gb}}+b_{\mathrm{nr}}\widehat{\mathrm{nr}}
+\beta_i+U_i,\\
p_i&=(1-\epsilon_t)\operatorname{softmax}(z_i/T_t)+\epsilon_t/|C_t|.
\end{aligned}
$$

$T_t:1.0\to0.5$，$\epsilon_t:0.15\to0.05$。每 attempt 对所选算子更新一次（幂等键 `(operator, attempt_id)`）。候选相对统一 anchor 的有向差经活跃尺度归一后取 $\tanh$，batch 聚合 $0.5\max+0.5\mathrm{mean}$。Simplify 在 fitness utility 非负时可混入复杂度下降项。日志：`operator_selection`、`operator_batch`。

---

## 6. 上下文构造与程序生成：给 LLM 看什么

### 6.1 初始化

生成至多 $n_{\mathrm{init}}=4$ 个初始程序。slot 0 要求简单完整方案；slot $>0$ 列出图中已有 idea（**最近至多 4 条，各截断 80 字符**），要求明显不同思路（run-local 去重，无任务词表）。输出约定：

```text
Idea: <自然语言设计思想>
Code:
```python
<完整函数>
```
```

有效程序按 `slot % n_islands` 建长度-1 轨迹。解析失败不耗评估预算；提交评估即计入样本。连续生成停滞达阈值可提前结束初始化。

### 6.2 两阶段 Refinement

除 Novelty / 初始化外：

**阶段 A（动作）** `build_action_prompt` 顺序拼接：

1. 任务与适应度方向；
2. 最近至多 5 步因果叙事；
3. `[Past Action Evidence]`：至多 2 成 / 2 败；
4. `[Elite Curriculum]`：按算子组装的 traces。实现上 **不**另开独立 Failure Boundary 块；repair / contrast / donor 作为同块内子标题（`Prefix repair:` / `Contrastive boundary:` / `Donor trace:`）。每步标注 `evidence_type`、`causal_status`、operator 与截断后的 action；
5. RankingModel best vs worst（idea + fitness）；
6. 算子名、role、约束；
7. base 的 idea、结构/runtime 摘要、代码与选择原因；
8. 函数契约；
9. 指令：恰好 `actions_per_iteration`（默认 2）条编号修改；禁止输出代码。

**阶段 B（代码）**：对每个动作，用 base 代码 + 动作生成完整 Idea+函数。

解析容忍常见编号格式；失败不计评估预算。

### 6.3 每轮开销

- Refinement：1 次动作 LLM + 至多 $A$ 次代码 LLM + 至多 $A$ 次评估；
- Novelty：至多 $A$ 次 initial-style LLM + 评估，无动作阶段。

Completed search iteration 大致对应相对搜索起点前进 $A$ 个评估样本（实现用样本计数整除判定）。**当前主循环内评估串行**：即使 `num_evaluators>1`，每次 `_evaluate` 只提交一个 future 并阻塞；并行度主要来自任务侧实例级 workers（如 CVRP `n_workers`）。

---

## 7. 评估后更新：如何记账与维护

### 7.1 候选注册

成功评估的 refinement 候选：

1. 写节点（code/idea/fitness/complexity/runtime）；
2. 写父→子边（action/operator/$\Delta$/outcome/iteration）；
3. 按算子 `extend` / `branch_from` / `create_initial` 更新轨迹；
4. 父子适应度更新 Elo（$K=16$）；
5. 更新 `best_node`；若刷新则 `record_best_event`；
6. Novelty gate；
7. `CurriculumPacket` 的 packet 级 outcome 反馈。

### 7.2 新颖性门控

对新轨迹相对其它 **active** 轨迹取组合相似度最大值。若 $\ge 0.92$ 且**未**刷新 live global best → 立即 archive（不删图）。若刷新了 global best → quality override，强制保留。通过后门控后写入 $V$。

### 7.3 相对排名（Elo）

期望分 $E_a=1/(1+10^{(R_b-R_a)/400})$，按胜/平/负更新。Union-find 维护连通分量；跨分量不直接比 Elo。Contrast：取最近 window（默认 20）条活跃终点，先按 raw fitness 定 best/worst 分量，分量内再按 Elo 取端点。

### 7.4 轨迹生存（每 completed iteration）

1. 同 `path_key` 去重归档；
2. 重算活跃池 $V$；
3. 保护全局 best 对应的 Pareto 优先轨迹；
4. 每岛截断到 `max_per_island=40`；
5. 全局截断到 `max_active_trajectories=160`。

截断顺序：保护集优先，其余按六维 Pareto 前沿，同前沿按 scalar 与 id。

### 7.5 Island 迁移

当 `stagnation>0`、`iteration>0` 且 `iteration % 20 == 0`：每岛按 scalar 取 top-1，环形移到下一岛（改 `island_id`，不新建身份）。

### 7.6 终止条件

| 条件 | 含义 |
| --- | --- |
| 样本数 ≥ $B$ | 正常耗尽预算 |
| 无活跃轨迹 | 停止 |
| 连续 `max_stalled_iterations=20` 次 attempt 无新评估进度 | 生成停滞 |
| 连续采样失败达阈值 | `search_aborted` |
| 显式中止 | 外部信号 |

---

## 8. 完整搜索算法

**输入**：LLM、程序模板、evaluator、预算 $B$。  
**输出**：标量最优 `best_node`。

```text
1.  生成并评估至多 n_init 个初始程序；建图与初始轨迹；按岛安置
2.  while 有预算且未中止：
3.      if 无活跃轨迹: break
4.      Trajectory-UCB / best / random 选题
5.      Portfolio 采样算子（含 preview）
6.      Backtrack 可覆盖选题；确定 base 与约束
7.      若非 fresh-start：EliteCurriculum.build → packet
8.      if Novelty:
9.          initial-style 生成并 create_initial
10.     else:
11.         两阶段生成至多 A 个程序并评估；写图；按算子 insert
12.     novelty gate；Elo；best event；packet outcome
13.     portfolio 更新一次；选中轨迹 visit+1
14.     if 新 completed iteration:
15.         更新 stagnation；survival；按需 migrate；checkpoint
16. 最终 survival；返回 best_node
```

---

## 9. 端到端例子（一次 attempt 的逻辑走读）

假设最大化任务，活跃池中已有多条轨迹，当前 global best 适应度为 $f^\star$。

1. **选题**：Trajectory-UCB 在 top-k ∪ 岛精英 ∪ elite 上采样，选中轨迹 $\tau$（终点 $p_L$，近期一步为 improve）。
2. **选算子**：Endpoint / Crossover / Novelty 均在候选中；Backtrack 因存在可分叉前缀也在候选。Preview 给 Endpoint 小幅 improve-streak bonus。Portfolio 采样得到 `endpoint_refine`。
3. **Base**：即 $p_L$。组装 CurriculumPacket：champion + 一条 improve_chain + contrast。
4. **动作 LLM**：看到 5 步因果、2 成 2 败经验、课程 traces、best–worst idea、以及「继续强化当前终点」约束，产出 2 条编号动作。
5. **代码 LLM ×2**：各生成完整函数并**依次**评估得 $f_1,f_2$；写节点与边；对每个成功子节点 `extend` 出新轨迹；portfolio 对本批观测做一次聚合更新。
6. **Gate**：若某子轨迹与池过相似且未破 $f^\star$，archive；若破纪录则保留并 `record_best_event`。
7. **记账**：Elo 更新；packet outcome；portfolio 用本批最优 utility 更新 Endpoint；$\tau$.visit += 1。
8. **若本 iteration 完成**：若 best 未动则 stagnation+1；Pareto 截断；每 20 个停滞 iteration 迁移岛精英；写 checkpoint。

该例子用于建立因果顺序；真实 run 中算子分布由 portfolio 动态决定。

---

## 10. 默认超参一览

### 10.1 搜索主配置

| 配置 | 默认 |
| --- | ---: |
| `n_init` | 4 |
| `actions_per_iteration` | 2 |
| `max_trajectory_length` | 8 |
| `n_islands` | 4 |
| `max_per_island` | 40 |
| 全局活跃上限 | 160 |
| 采样策略 | `trajectory_ucb` |
| novelty 阈值 | 0.92 |
| migration 周期 | 20 iterations |
| 连续停滞停止 | 20 attempts |
| 精英事件窗口 | 4 |
| improve-chain 最大步 | 3 |
| 课程 positive traces | ≤2 |
| 课程/经验 action 截断 | 300 字符 |

### 10.2 价值与选择

| 配置 | 默认 |
| --- | ---: |
| $(w_q,w_p,w_d,w_n,w_c,w_r)$ | $(0.42,0.18,0.12,0.12,0.08,0.08)$ |
| $(w_{\mathrm{code}},w_{\mathrm{traj}})$ | $(0.7,0.3)$ |
| $\gamma$ | 0.8 |
| $(\lambda_{\mathrm{pos}},\lambda_{\mathrm{down}})$ | $(0.25,0.5)$ |
| $q_{\min}$ | 0.5 |
| 适应度裁剪分位 | 0.10 |
| UCB $(c_0,c_{\mathrm{floor}},\beta)$ | $(0.4,0.05,0.20)$ |
| top-k / island top | 12 / 1 |
| elite 直采概率 | 0.15 |
| $T_{\mathrm{sel}}$ | 0.8 |

### 10.3 Portfolio

| 配置 | 默认 |
| --- | ---: |
| EMA 衰减 | 0.8 |
| $(\alpha,\lambda_d,\lambda_c)$ | $(1,0.5,0.05)$ |
| 成本尺度 | 120 |
| UCB $(c,n_0)$ | $(0.5,1)$ |
| $(b_{\mathrm{gb}},b_{\mathrm{nr}})$ | $(0.5,0.25)$ |
| near-record 容差 | 0.10 |
| $(T_{\mathrm{init}},T_{\mathrm{end}})$ | $(1.0,0.5)$ |
| $(\epsilon_{\mathrm{init}},\epsilon_{\mathrm{end}})$ | $(0.15,0.05)$ |
| prior $(\mathrm{pseudo},\mathrm{mean})$ | $(2,0.05)$ |
| context bound / late novelty cap | 0.2 / 0.2 |
| batch 聚合 | $0.5\max+0.5\mathrm{mean}$ |
| Elo $K$ / contrast window | 16 / 20 |

---

## 11. 实现边界、日志与续训

### 11.1 模块地图

| 模块 | 职责 |
| --- | --- |
| `traceaad.py` | 主循环、评估、gate、survival、hooks |
| `derivation_graph.py` / `trajectory_memory.py` | 事实图与轨迹池 |
| `value.py` / `credit.py` / `similarity.py` | $V$、潜力、相似度 |
| `portfolio.py` / `operator_signals.py` / `operators/` | 调度与五算子 |
| `experience_memory.py` / `curriculum.py` / `context.py` / `prompt.py` | 生成证据与 prompt |
| `feedback.py` / `islands.py` / `complexity.py` | Elo、迁移、复杂度 |
| `checkpoint.py` / `resume.py` | 断点续训 |

### 11.2 主要日志事件

`program_evaluated`、`operator_selection`、`operator_batch`、`child_accepted`（可带 `curriculum_ids`）、`novelty_gate`、`best_updated`、`trajectory_created`、`migrate`、`checkpoint_saved`，以及各类 `*_error` / `search_stopped`。`method_state.jsonl` 的 `iteration_start` 可含 `curriculum_ids` 与 `curriculum_snapshot`。`llm_calls.jsonl` 记录各段字符数（含 `experience_chars` / `curriculum_chars`）。

### 11.3 Checkpoint

每个 completed iteration 与 run 结束写入 `logs/checkpoints/ckpt_{sample}.json` 与 `latest.json`（图、轨迹、portfolio、curriculum、Elo、rng、样本计数、best 等）。`RESUME_FROM=<run_dir>` 加载后续跑；ExperienceMemory 由图谱重建。要求 checkpoint 中至少一条 active 轨迹。

### 11.4 文档地图

| 文档 | 读什么 |
| --- | --- |
| **本文** | TraceAAD 完整搜索机制与默认超参（唯一权威对照） |
| `docs/results/` | 各 task 实验结果，不是机制定义 |
| `docs/worklog/` | 研究过程记录 |

---

## 12. 小结

TraceAAD 的逻辑链条是：

**稀缺的有效改进发生在轨迹上** → 用多维 $V$ 与 UCB **决定跟哪条路** → 用角色化算子与 portfolio **决定怎么改** → 用分层证据 **决定 LLM 看见什么** → 用边级事实与门控/生存 **决定如何更新记忆**，同时严格禁止课程伪造搜索状态。

本文描述的是当前代码实际执行的机制，可作为实现对照与后续消融的基线。Portfolio 的 late novelty cap 等细节仍可能随实验调整；以代码与本节超参表的同步更新为准。
