# TraceAAD 构造轨迹与精英课程机制

## 1. 机制命题

TraceAAD 当前把 trajectory 同时当作两种东西：

1. **搜索资源**：决定下一次把评估预算投向哪里；
2. **生成证据**：告诉 LLM 哪些修改曾经发生过、结果如何。

这两种用途需要的价值定义并不相同。搜索资源需要回答：

> 哪条真实路径值得继续投入预算？

生成证据需要回答：

> 哪些修改结构值得被 LLM 学习、组合或避免？

在算法自动设计中，真正有效的改进是稀缺事件。大多数边是 plateau、regress 或低幅度波动，因此把普通轨迹的形状直接当成“未来潜力”会产生大量噪声。当前审计也显示，分支机制能够产生真实改进，但现有 path potential 与后续收益的相关性很弱。

本机制的核心命题是：

> **搜索历史中的高质量结构应被重新组织成可读、可组合、带证据等级的课程材料；它们作为 LLM 的生成指导和精英算子输入，而不是伪造的 active trajectory。**

这不是给搜索池添加更多轨迹，而是给 TraceAAD 增加一个从事实图到生成策略的中间层。

---

## 2. 第一性原理

### 2.1 搜索的真实对象

给定程序状态 $p$，LLM 根据上下文 $x$ 产生动作 $a$，执行后得到程序状态 $p'$，评估器返回 $f(p')$：

$$
a \sim \pi_{\mathrm{LLM}}(a \mid p, x), \qquad
p' = \mathrm{Apply}(p, a), \qquad
y = \mathrm{Evaluate}(p').
$$

TraceAAD 可以改变三件事：

1. 从哪个状态 $p$ 开始生成；
2. 给 LLM 哪些历史证据 $x$；
3. 如何根据 $y$ 更新下一轮的搜索和课程。

构造轨迹主要作用于第二件事，并通过精英算子间接影响第一件事；它不应伪装成新的程序状态或新的评估结果。

### 2.2 三种事实等级

系统中必须显式区分：

| 等级 | 内容 | 可信度 | 允许的用途 |
| --- | --- | --- | --- |
| `fact` | DAG 中真实存在的节点、父子边、action、fitness、outcome | 最高 | credit、Backtrack、日志、课程 |
| `inference` | 从多条事实边构造出的 champion、前缀、组合链 | 中等 | prompt、算子先验 |
| `instruction` | LLM 根据课程提出的下一步修改 | 待评估 | 生成新子代 |

最重要的约束是：

> **推断关系可以影响生成，但不能冒充事实边。**

因此，跨分支拼接的 champion trace 不能用于真实路径长度、父子 credit、trajectory visit 或 prefix branching。

### 2.3 轨迹价值必须拆成两个空间

定义两套互不混淆的价值：

$$
V_{\mathrm{search}}(\tau)
=
\text{真实路径作为预算投放对象的价值},
$$

$$
V_{\mathrm{teach}}(e)
=
\text{历史事件/课程片段作为生成证据的价值}.
$$

`V_search` 服务 `Trajectory-UCB`、Pareto survival 和 active pool；`V_teach` 服务课程检索、精英算子和 prompt 组装。

第一版不把 `V_teach` 直接加进 `V_search`。否则一条“很会讲故事”的构造链可能抢走预算，却没有可扩展的真实 endpoint。

---

## 3. 新的记忆架构

TraceAAD 从“三层记忆”扩展为四层：

```text
Program Memory
  DerivationGraph
  所有真实程序节点和父子边

Search Memory
  TrajectoryMemory
  当前仍参与预算分配的真实路径

Experience Memory
  ExperienceMemory
  边级成功/失败 action 的事实查询

Curriculum Memory
  EliteCurriculum
  从事实图构造的精英、修复、对照和组合课程
```

### 3.1 Program Memory：事实源

`DerivationGraph` 是唯一事实源，保存：

```text
ProgramNode:
  id, code, idea, fitness, complexity, runtime

ImprovementEdge:
  id, parent_id, child_id, action, operator,
  delta, outcome, iteration
```

课程层可以引用这些对象，但不复制成另一份“程序事实”。引用失效或节点归档时，课程仍然可以保留为历史记录，但必须标记其来源状态。

### 3.2 Search Memory：真实搜索单位

`TrajectoryMemory` 继续只保存真实父子边构成的路径。它负责：

- trajectory selection；
- endpoint extension；
- Backtrack branch；
- visit count；
- novelty gate；
- Pareto survival；
- active island 管理。

构造轨迹不得直接进入这里。

### 3.3 Experience Memory：单步经验

`ExperienceMemory` 继续回答：

> 对当前 operator，有哪些单步 action 曾经成功或失败？

它适合短、局部、低成本的证据，不负责构造跨步叙事。

### 3.4 Curriculum Memory：结构化课程

`EliteCurriculum` 回答：

> 当前搜索阶段和 operator 最需要看到哪一种成功结构、失败边界或互补动作？

它不存新的程序状态，只存带来源引用的 `CurriculumTrace`。其职责包括：

1. 记录 global-best 更替；
2. 从 DAG 提取真实 improve chain；
3. 识别高价值前缀和破坏性后缀；
4. 从不同精英支系构造组合课程；
5. 按 operator、搜索阶段和停滞状态组装 prompt；
6. 记录课程被使用后产生的真实后代结果。

---

## 4. 统一数据模型

### 4.1 `TraceStep`

每个课程步骤都必须说明来源和证据类型：

```text
TraceStep:
  source_node_id
  source_edge_id | null
  parent_node_id | null
  operator
  action
  fitness_before
  fitness_after
  delta_to_parent
  delta_to_incumbent
  outcome
  evidence_type
  causal_status
```

`causal_status` 取值：

| 值 | 含义 |
| --- | --- |
| `direct` | 真实父子边，action 和 delta 可直接解释 |
| `prefix` | 真实路径中的高价值节点，但后续步骤可能失败 |
| `jump` | 两个节点都是真实节点，但中间不存在连续父子关系 |
| `composed` | 从多个真实分支抽取并排列的动作序列 |

### 4.2 `CurriculumTrace`

```text
CurriculumTrace:
  id
  kind
  steps
  source_node_ids
  source_edge_ids
  terminal_node_id | null
  quality_gain
  causal_coherence
  novelty
  reuse_count
  last_used_iteration
  freshness
  confidence
```

`kind` 取值：

```text
champion
improve_chain
prefix_repair
contrastive
elite_recombine
```

其中：

- `quality_gain`：真实适应度收益；
- `causal_coherence`：步骤是否来自连续真实边；
- `novelty`：与当前已使用课程的差异；
- `reuse_count`：被注入 prompt 或精英算子使用的次数；
- `freshness`：距离最近一次使用或产生的时间；
- `confidence`：由证据强度和构造类型决定。

### 4.3 `CurriculumPacket`

每次生成前，课程层只向调用方暴露一个小接口：

```python
packet = curriculum.build(
    *,
    operator=operator_name,
    base_node_id=base_node_id,
    selected_trajectory_id=trajectory_id,
    iteration=iteration,
    stagnation=stagnation,
)
```

返回：

```text
CurriculumPacket:
  positive_traces
  repair_trace
  contrast_trace
  donor_trace
  instructions
  evidence_summary
```

调用方不需要知道 champion 如何抽取、课程如何排序或 token 如何裁剪。复杂度集中在 `EliteCurriculum` 的实现内，外部接口保持小而稳定。

---

## 5. 课程构造器

课程构造器不是一次性把整张 DAG 线性化，而是维护多个有明确语义的视图。

### 5.1 Champion Trace：全局精英记录链

每次 global best 被刷新，追加一个 `ChampionEvent`：

```text
previous_best_node
new_best_node
source_parent_node
source_edge
operator
delta_to_previous_best
delta_to_parent
sample / iteration
```

得到：

```text
best_0 -> best_1 -> ... -> best_k
```

这条链的特点：

- 质量单调改善；
- 相邻节点可能跨分支；
- 只说明“这些状态依次成为纪录”，不说明 `best_i` 如何生成 `best_{i+1}`；
- 适合展示全局进步方向，不适合逐步 credit。

构造规则：

1. 初始化阶段的第一个有效候选作为 `champion_root`；
2. 只记录刷新 incumbent 的节点；
3. 对极小的数值噪声使用 directed-delta 阈值；
4. 同一 operator 连续产生的重复微改进合并为一个窗口；
5. 保留最近记录、最大幅度记录和不同 operator 的代表记录；
6. 课程展示时最多取最近 $K$ 个事件，不把整条历史塞进 prompt。

### 5.2 Causal Improve Chain：真实连续成功链

在 DAG 上寻找连续 `outcome=improve` 的短路径。与 champion trace 不同，它要求：

```text
p_i -> p_{i+1} -> p_{i+2}
```

每一步都有真实 edge。候选链优先级由以下因素共同决定：

$$
S_{\mathrm{chain}}
=
w_q Q_{\mathrm{terminal}}
+ w_g G_{\mathrm{sum}}
+ w_c C_{\mathrm{causal}}
+ w_n N_{\mathrm{chain}}
+ w_f F_{\mathrm{fresh}}.
$$

其中：

- $Q_{\mathrm{terminal}}$：终点质量；
- $G_{\mathrm{sum}}$：有向累计改进；
- $C_{\mathrm{causal}}$：连续真实边比例；
- $N_{\mathrm{chain}}$：与已有课程的差异；
- $F_{\mathrm{fresh}}$：新近程度。

不把长度本身当成价值。长链可能只是重复小改，短链可能包含一次关键突破。

### 5.3 Prefix Repair：高价值前缀与失败边

若轨迹满足：

```text
strong_prefix -> regressed_endpoint
```

则构造：

```text
[成功前缀]
  为什么这个节点质量高
[失败动作]
  哪个动作导致退步
[修复目标]
  从前缀重新提出不同的下一步修改
```

这个课程专门服务 `backtrack_branch` 和停滞阶段。它把失败看成边界信息，而不是只把失败程序当作负样本。

### 5.4 Contrastive Trace：成败并置

从相同或相近 base 邻域中抽取：

```text
同一前缀 -> 成功 action -> 改善
同一前缀 -> 失败 action -> 退步
```

若没有相同 base，则退化为同 operator、相近质量和相近结构的对照。

Contrastive trace 只能作为“方向边界”：

- 哪类修改曾经导致退步；
- 哪些复杂度增长没有换来收益；
- 哪些动作只是重复已有结构。

它不参与 active trajectory 的选择，也不作为“反向奖励”。

### 5.5 Elite Recombine：精英动作组合链

从两个不同支系抽取：

```text
recipient elite:  当前质量高、结构稳定
donor elite:      包含互补成功动作
```

构造的不是伪造程序链，而是一个组合课程：

```text
Recipient 的保留约束
Donor 的单个成功动作
Donor 动作的历史结果
两者之间的结构差异
```

LLM 只负责提出一个新的真实 action。真正的父子边在新程序评估成功后才写入 DAG。

---

## 6. 课程价值：从“路径潜力”转为“教学效用”

### 6.1 教学效用的组成

对课程 $e$ 定义：

$$
V_{\mathrm{teach}}(e)
=
\alpha G
+\beta C
+\gamma R
+\delta N
+\eta T
-\lambda U.
$$

各项含义：

- $G$（gain）：真实质量提升；
- $C$（causality）：因果连续性；
- $R$（reusability）：该结构被复用后产生有效后代的比例；
- $N$（novelty）：与当前课程集合的差异；
- $T$（transfer）：在不同 base、island 或 operator 下仍有效的程度；
- $U$（overuse penalty）：过度重复使用带来的衰减。

这不是一个单纯的“越高分越好”排序。课程必须同时满足：

1. 有真实收益证据；
2. 能说明发生了什么；
3. 不只是当前单个 endpoint 的偶然细节；
4. 不被同一条成功故事垄断。

### 6.2 证据强度

不同课程的先验置信度：

```text
causal improve chain  >  prefix repair  >  champion jump  >  composed trace
```

但 champion jump 的质量信息可能高于一条普通 causal chain。因此：

- `causal_status` 控制“能否声称因果”；
- `quality_gain` 控制“是否值得展示”；
- `confidence` 控制“在 prompt 中的措辞强度”。

不得把这三个维度压成一个无解释的分数。

### 6.3 课程衰减与多样性

课程有三种衰减：

1. **时间衰减**：旧课程逐渐降低；
2. **使用衰减**：同一课程重复出现会降低；
3. **结果衰减**：课程被复用后连续产生 plateau/regress 时降低。

课程池使用分层保留：

```text
global champion layer
operator-specific layer
island-specific layer
failure-boundary layer
```

每层都保留少量名额，避免一个全局冠军覆盖所有生成上下文。

---

## 7. 精英进化算子

现有 `endpoint_refine`、`backtrack_branch` 和 `mechanism_crossover` 保留。新增的精英机制不是替代它们，而是给它们提供更明确的动作策略。

### 7.1 Elite Refine：精英定向强化

**Base**：当前 global best 或 selected trajectory endpoint。

**课程**：Champion Trace 最近有效事件 + 一个相关 causal improve chain。

**约束**：

```text
保留当前程序的核心结构；
延续一个已出现过有效证据的方向；
只改变一个主要算法思想；
不要机械复制历史 action。
```

它对应精英变异中的 exploitation，但 mutation 必须由 LLM 重新解释，而不是字符串复用。

### 7.2 Elite Repair：精英前缀修复

**Base**：高价值前缀节点，而不是退步 endpoint。

**课程**：Prefix Repair。

**约束**：

```text
恢复前缀的有效性质；
明确避开导致退步的 action；
提出一个与失败动作不同的替代修改。
```

它把现有 Backtrack 从“选择历史 base”提升为“选择历史 base + 失败边界 + 修复目标”。

### 7.3 Elite Recombine：精英交叉

**Recipient**：当前高质量程序。

**Donor**：互补精英链中的一个成功 action 或 idea。

**约束**：

```text
只移植一个 donor 方向；
保留 recipient 的输入输出契约和主体结构；
解释 donor 方向为何可能与 recipient 互补；
不复制 donor 的完整程序。
```

这是真正的“精英交叉”：交叉的是经过评估的修改结构，而不是两个伪造程序节点。

### 7.4 Elite Contrast：精英边界学习

**Base**：当前 endpoint 或高质量前缀。

**课程**：一条成功精英动作 + 一条相近失败动作。

**约束**：

```text
保留成功动作的有效目标；
避免失败动作暴露出的复杂度、重复或过度修改；
提出一个局部、可评估的折中变体。
```

### 7.5 何时启用精英算子

精英算子不是固定比例轮换，而是由 portfolio 选择。但 portfolio 的奖励必须区分：

- 是否刷新 global best；
- 是否接近 record；
- 是否只重复了已有 elite；
- 是否带来新的高质量支系；
- 是否在停滞阶段恢复了有效改进。

精英算子失败时，衰减的是该算子或该课程组合的先验，不删除底层事实，也不禁用所有精英课程。

---

## 8. 动态课程组装

课程不是固定 prompt 模板，而是一个根据搜索状态组装的 `CurriculumPacket`。

### 8.1 正常利用阶段

上下文顺序：

```text
当前真实轨迹最近步骤
当前 operator 的成功 action
Champion Trace 最近 2～4 个事件
一个 causal improve chain
一个短失败边界
```

目标是延续有效方向，同时避免重复历史失败。

### 8.2 停滞阶段

上下文顺序：

```text
当前高价值前缀
造成退步或 plateau 的动作
另一条支系的互补精英动作
当前 champion 的最后一次有效变化
```

目标从“继续强化”切换为“修复或重组”，而不是不断重复当前 endpoint refine。

### 8.3 探索阶段

`novelty_jump` 不应被 Champion Trace 过度约束。它只接收：

- 已有 elite idea 的摘要，用于避免直接重复；
- 少量跨岛、低相似度课程；
- 不包含具体实现路径的开放任务约束。

探索算子可以读取课程的“禁重复”信息，但不应读取完整成功链作为强先验。

### 8.4 Operator 过滤

课程按 operator 角色筛选：

| Operator | 主课程 | 辅助课程 |
| --- | --- | --- |
| `endpoint_refine` | champion、improve_chain | contrastive |
| `backtrack_branch` | prefix_repair | causal、contrastive |
| `mechanism_crossover` | elite_recombine | champion |
| `simplify` | 成功复杂度下降链 | 复杂度失败边界 |
| `novelty_jump` | 低相似课程摘要 | 不使用具体 champion 步骤 |

同一课程可以被多个 operator 使用，但每次使用都记录上下文角色，避免把一个成功 action 的收益错误归给所有 operator。

---

## 9. 搜索主循环中的新数据流

```text
初始化真实程序
  -> DerivationGraph / TrajectoryMemory
  -> 记录 champion event

每次 search attempt
  1. 从 TrajectoryMemory 选择真实 trajectory
  2. Portfolio 选择 operator
  3. EliteCurriculum 根据 operator/base/stagnation 组装 packet
  4. 选择真实 base node
  5. LLM 生成一个或多个 action
  6. LLM 将 action 实现为完整程序
  7. 评估程序
  8. 成功评估后写入真实 node/edge/trajectory
  9. 更新 best、ExperienceMemory、RankingModel、Portfolio
 10. 若刷新 best，追加 champion event
 11. 以新结果反向更新课程的 reuse/reward/confidence
 12. survival / migration
```

关键顺序：

> 先生成真实 offspring，再更新课程；课程不能提前把尚未评估的 LLM 预测当作成功经验。

### 9.1 课程使用记录

每次生成需要记录：

```text
curriculum_id
curriculum_kind
source_edge_ids
operator
base_node_id
prompt_role
```

候选评估后，再记录：

```text
delta_to_base
delta_to_incumbent
outcome
accepted
global_best
near_record
```

这样可以把“某条课程被使用过”与“某条课程产生了有效 offspring”区分开。

### 9.2 课程反馈不是边级 credit 的替代

一条课程可能包含多个 action，最终 offspring 只被评估一次，因此不能把整条课程的收益平均回传给所有步骤。

默认采用分层反馈：

1. 对整个 `CurriculumPacket` 记录 outcome；
2. 对其中唯一被指定为 primary action 的步骤给予主要 credit；
3. 对 donor、contrast 和背景步骤只记录 exposure；
4. 只有真实连续链中的边才保留原有 stepwise credit。

这避免了把“上下文中出现过”误认为“真正导致了改进”。

---

## 10. 精英保护与群体多样性

精英机制不能退化为单一 global best 的复制器。需要同时维护：

### 10.1 Global elite

保存当前最高质量的 champion chain，保证全局最好方向不会完全从上下文消失。

### 10.2 Lineage elite

每个 island 或结构支系保留一条局部精英链。它们可以质量略低，但必须提供不同的代码/行为模式。

### 10.3 Repair elite

保存“高质量前缀 + 失败后缀”的边界课程。它们不一定是最终最优，但对 Backtrack 和停滞恢复有价值。

### 10.4 Diversity quota

课程组装时不能让同一条 champion chain 占满所有位置。至少满足：

```text
一个全局精英来源
一个非冠军支系来源
一个成功/失败边界来源
```

若课程池中只有冠军链，则减少精英提示强度，而不是把冠军链重复填充。

---

## 11. 失败模式与硬约束

### 11.1 不允许伪造 DAG 因果

Champion jump、elite recombine 和 contrastive trace 都必须标注 `jump` 或 `composed`。它们不能：

- 增加图边；
- 改变 node parent；
- 增加 trajectory length；
- 参与 stepwise delta；
- 伪造 visit 或 survival 价值。

### 11.2 不允许把成功故事当成充分条件

课程中的指令必须使用证据措辞：

```text
This modification previously improved fitness in the recorded context.
It is an example to consider, not a guaranteed solution.
```

不要在 prompt 中写成“该机制一定有效”。

### 11.3 不允许 champion 垄断

使用次数、时间和连续失败都要降低课程权重；跨 island 的非冠军精英必须有保留名额。

### 11.4 不允许 prompt 无限增长

课程层设置独立的 token budget：

```text
champion events: 2～4
causal chain:    1 条
repair/contrast: 1 条
donor action:    1 条
```

当前真实 trajectory 叙事仍是主上下文，课程是有界补充。

### 11.5 不允许把课程反馈泄漏进历史事实

课程的 confidence、reuse 和 reward 是课程层状态，不能回写修改原始 `ImprovementEdge` 的 delta/outcome。

---

## 12. 模块接口与实现归属

### 12.1 新增 `EliteCurriculum`

建议新增：

```text
llm4ad/method/traceaad/curriculum.py
```

外部只暴露：

```python
record_best_event(...)
refresh_from_graph(...)
build(...)
record_outcome(...)
snapshot(...)
```

内部实现负责：

- champion event 维护；
- causal chain 提取；
- prefix repair 构造；
- contrastive 配对；
- elite recombine 组装；
- 课程评分、衰减、配额；
- prompt packet 截断。

### 12.2 `context.py`

`context.py` 只负责把 `CurriculumPacket` 格式化为 prompt，不负责决定哪些课程有价值。

建议 prompt 分块：

```text
[Current Causal Trajectory]
[Past Action Evidence]
[Elite Curriculum]
[Failure Boundary]
[Operator]
[Base Program]
```

### 12.3 `traceaad.py`

主循环只负责：

1. 在 best 更新时通知 `EliteCurriculum`；
2. 生成前请求 packet；
3. 评估后提交 packet outcome；
4. 在日志中记录课程来源。

不把课程构造逻辑塞进 `_update_best`、`_run_refine` 或具体 operator，避免调用方承担过多状态。

### 12.4 operators

operator 只声明：

```text
required_curriculum_kind
primary_step_role
base_selection_policy
```

具体课程检索仍由 `EliteCurriculum` 完成。这样 `Backtrack` 不需要理解 champion 的全局构造，`Crossover` 也不需要复制课程排序逻辑。

---

## 13. 实施计划

### 阶段 1：建立课程事实层

1. 增加 `ChampionEvent`、`TraceStep`、`CurriculumTrace`、`CurriculumPacket`；
2. 在 best 更新处记录来源 node、edge、operator、delta；
3. 增加 `EliteCurriculum`，先实现 champion trace；
4. 保持 active trajectory 和现有 `V_search` 完全不变；
5. 日志记录课程来源和证据类型。

### 阶段 2：加入 DAG 结构课程

1. 实现连续 improve chain；
2. 实现 prefix repair；
3. 实现 contrastive pair；
4. 实现课程质量、因果性、重复使用和新鲜度状态；
5. 让 `CurriculumPacket` 按 operator 返回有界内容。

### 阶段 3：精英算子化

1. 将 `endpoint_refine` 接入 Elite Refine；
2. 将 `backtrack_branch` 接入 Elite Repair；
3. 将 `mechanism_crossover` 接入 Elite Recombine；
4. 为 `simplify` 增加复杂度成功链和失败边界；
5. 对 `novelty_jump` 只提供低约束、多样性课程。

### 阶段 4：闭环反馈

1. 记录课程 exposure；
2. 记录 packet 级 outcome；
3. 对 primary step 更新教学效用；
4. 增加使用衰减、连续失败衰减和跨 island 配额；
5. 把课程快照写入 `method_state.jsonl`，保证运行过程可审计。

### 阶段 5：机制文档与参数收口

1. 将 `TraceAAD完整机制设计.md` 的三层记忆更新为四层记忆；
2. 明确 `V_search` 与 `V_teach` 的分工；
3. 把课程预算、衰减、配额和 operator 映射写入正式配置；
4. 删除“构造轨迹直接进入 active pool”这类歧义表述；
5. 在 worklog 中记录实现后的机制判断。

---

## 14. 机制不变量

实现完成后必须始终满足：

1. 课程只能引用真实 node/edge，不能制造程序事实；
2. active trajectory 只由真实父子边组成；
3. `V_search` 不直接等于 `V_teach`；
4. Champion Trace 可以非因果，但必须标记为 `jump`；
5. Causal Improve Chain 的每一步必须有真实 edge；
6. 课程反馈不能改写原始 edge 的 delta/outcome；
7. LLM 每次只生成当前真实 base 的下一步修改；
8. 精英课程不能让 novelty、island 和探索逻辑失效；
9. 课程使用和课程收益必须分开记录；
10. 任何课程都不能因为曾经成功而永久获得最高优先级。

---

## 15. 最终机制图景

TraceAAD 的搜索闭环变为：

```text
真实程序/边
    |
    v
DerivationGraph -----> Search Memory
    |                       |
    |                       +--> UCB / survival / Backtrack
    |
    v
EliteCurriculum
    |
    +--> Champion Trace
    +--> Causal Improve Chain
    +--> Prefix Repair
    +--> Contrastive Trace
    +--> Elite Recombine
    |
    v
CurriculumPacket
    |
    v
LLM 生成下一步 action
    |
    v
真实 offspring 评估
    |
    +--> 更新 DAG / trajectory / portfolio
    +--> 更新 champion
    +--> 更新课程教学效用
```

因此，TraceAAD 不再只是在“轨迹池里选择一条路径”，而是同时维护：

1. **哪些真实状态值得继续搜索**；
2. **哪些历史结构值得教给 LLM**；
3. **哪些精英结构值得重组、变异或修复**；
4. **哪些失败边界应该阻止重复犯错**。

人工构造轨迹的真正作用不是扩大搜索空间，而是把稀缺的成功事件压缩成可复用的生成策略，同时严格保留事实、推断和新生成结果之间的语义边界。
