# TraceAAD V7：可执行状态唯一的轨迹进化

> 本文定义 V7 的拟议机制；实现已完成，但性能实验尚未运行。
> V7 的设计来源、证据和待验证假设见
> [V5 机制全面分析与 V7 设计](../research/TraceAAD-V5机制全面分析与V7设计.md)。
> V4/V5/V6 继续由各自文档和独立实现维护。

## 科学动机与可证伪设计假设

### 1. 科学问题

算法改进历史只有在保留适用条件时，才能帮助下一步生成。与此同时，搜索预算
实际作用于当前可执行程序；相同程序即使来自不同历史，也不应重复占用有限繁衍
席位。

V7 的核心设计假设是：

> 完整保存每次算法改进的事实路径，用与当前程序严格匹配的历史指导生成；
> 同时让 active frontier 表示不同的可执行程序状态，并让探索压力随剩余预算
> 下降，使搜索后期集中到已经显示价值的路线。

因此 V7 明确分开：

- **事实层**：所有有效程序节点、修改边和轨迹身份；
- **生成层**：当前主轨迹、唯一结构锚点和可选参考轨迹；
- **繁衍层**：按精确代码状态去重的 active endpoint；
- **预算层**：质量、趋势、访问数和剩余预算。

这些假设是待检验的机制关系，不是已获得的性能结果。若它们成立，实验应检验：历史条件是否能够更准确地指导当前可执行程序的生成；精确 endpoint 去重是否使 active frontier 表示不同的可执行程序状态；剩余预算衰减是否使后期选择更多依赖已显示的路线价值。最终算法质量、重复运行稳定性和 held-out 表现仍是主要判据。

## 形式化状态、轨迹、算子与选择

### 2. 搜索状态

程序节点保存代码、实现思想、fitness、非空 LOC 和 `code_hash`。修改边保存唯一
结构父代、Action、算子、主/参考轨迹、三类 fitness 变化、代码变化、结果标签
和批次信息。每个有效子程序只有一条入边；参考程序只提供知识，不形成第二结构
父代。

一条 active 轨迹保存最多 8 个节点和 7 条边：

\[
\tau_i=(n_0,e_1,n_1,\ldots,e_k,n_k),\quad k\le 7.
\]

轨迹记录 endpoint、当前保留路径的 compact best、访问次数、状态、质量 $Q$、
趋势 $P$ 和调度值 $V$。路径超过上限时从最早端截断，完整派生图不删除。

### 3. 可执行状态唯一性

active frontier 满足不变量：

\[
\forall \tau_i\ne\tau_j\in\mathcal A,
\operatorname{hash}(endpoint_i)\ne\operatorname{hash}(endpoint_j).
\]

这里的 hash 是完整规范化程序文本的精确 `code_hash`，不使用 token 相似度、
Idea 相似度或路径相似度。

初始化和每个有效子代写入后立即检查该不变量：

1. 新程序节点和修改边始终保留；
2. 新轨迹始终写入事实 memory；
3. 若 active 中已有相同 endpoint，只保留一条代表路线 active；
4. 其余重复路线标记为 archived，不再被选择或作为参考；
5. 优先保留当前 global-best route，否则确定性保留较早 route id。

这是一条状态卫生规则，不判断两段不同代码是否功能相似，也不删除不同历史。
checkpoint 恢复时若发现两个 active endpoint 具有相同 `code_hash`，直接拒绝恢复。

初始化目标是 30 个唯一 active endpoint。重复有效程序仍消耗 evaluator 预算并
写入事实层，但不增加 active 数；连续尝试没有增加唯一 active 状态时按停滞上限
停止，达到预算上限时即使不足 30 个唯一状态也停止。

### 4. 历史上下文

主轨迹始终对应当前唯一结构锚点。选择 endpoint 时展示它的形成历史；选择内部
compact best 时，历史分为：

- `[How This Program Was Reached]`：锚点形成过程；
- `[Later Attempts From This Program]`：锚点之后已测试的尝试。

每条边按时间顺序展示 Planned Action、Implemented Idea、父子 fitness、
improve/plateau/regress、代码变化比例和 LOC 变化。主锚点完整展示代码，历史边
不重复完整代码。

双轨迹算子另外展示参考轨迹及其 compact-best 程序。参考轨迹只说明另一条路线
有哪些具体思想和试错结果；子程序仍写入主轨迹。

Action 和 Code 阶段接收同一份主历史；双轨迹时也接收同一份参考历史和参考
程序，避免决策与实现使用不同事实条件。

### 5. 四个语义算子

| 算子 | 约束 |
| --- | --- |
| `trace_ideate` | 根据保留历史提出真正新的算法方向，把 regress/plateau 当作已测试边界 |
| `trace_refine` | 对已显示价值的机制或历史暴露的弱点做一次聚焦修改 |
| `trace_synthesize` | 从主/参考轨迹各找一个受支持原则，使两者在主程序内形成功能交互，不拼接完整实现 |
| `trace_transfer` | 保持主程序核心结构，只适配参考轨迹中的一个受支持思想 |

有至少两条 active 路线时，四算子等概率选择。只有一条 active 路线或双轨迹
上下文不可用时，只在 `ideate/refine` 中等概率选择。算子不维护在线 reward、
Elo 或成熟度状态。

双轨迹参考排除主轨迹后按路线质量 $Q$-softmax 选择，使用参考路线 compact
best。参考选择不设置相似度门槛，也不增加参考路线访问数。

### 6. Action 与 Code

Action 阶段输出最多两条编号、单行、自包含的自然语言修改。每条 Action 说明
具体改动，并遵守当前算子语义。Code 阶段从唯一主锚点实现 Action，返回完整
可执行候选。

默认 Action 输出上限 1024 token，Code 输出上限 8192 token。每一条有效
Action 单独生成和评价一个候选；解析失败不进入图，也不消耗 evaluator 预算。

### 7. 轨迹评分

#### 7.1 成果质量

endpoint 和 compact best 分别在当前 active 集合中做并列感知百分位排名：

\[
Q_i=0.7Q_{endpoint,i}+0.3Q_{best,i}.
\]

程序比较首先看 evaluator fitness；仅在 fitness 完全相同时，非空 LOC 更少者
优先。最大化和最小化任务先统一为有向 fitness。

#### 7.2 近期趋势

对保留路径的每条父子边定义：

\[
s_j=\begin{cases}
+1,&\Delta_j>10^{-12}\\
0,&|\Delta_j|\le10^{-12}\\
-1,&\Delta_j<-10^{-12}.
\end{cases}
\]

用折扣 $d=0.8$ 让近期边权重更高：

\[
P_i=\frac{1}{2}\left(
1+\frac{\sum_{j=1}^{k}d^{k-j}s_j}
{\sum_{j=1}^{k}d^{k-j}}
\right).
\]

无历史边时 $P_i=0.5$。调度基础值为：

\[
V_i=0.8Q_i+0.2P_i.
\]

#### 7.3 剩余预算衰减 UCB

评估预算为 $B$，已经评价 $b$ 个程序，路线访问数为 $n_i$，全局批次选择计数
为 $N$。$N$ 是从搜索开始发起过的主轨迹 sibling 批次数，不是当前 active
路线访问数之和，因此归档路线不会让探索时钟回退。剩余预算比例：

\[
r_b=\operatorname{clip}\left(\frac{B-b}{B},0,1\right).
\]

父代调整值：

\[
A_i=V_i+0.25r_b
\sqrt{\frac{\log(1+N)}{1+n_i}}.
\]

再按温度 0.2 的 softmax 选择主轨迹。早期 $r_b$ 较大，低访问路线得到覆盖；
预算接近耗尽时 $r_b\to0$，选择逐步回到 $V_i$。无限预算配置下 $r_b=1$。

访问单位是“主轨迹被选择并发起一个 sibling 批次”。即使后续生成或解析失败，
访问仍增加，因为路线已经获得一次 proposal 机会。

### 8. 锚点与批次语义

主轨迹选定后，若 endpoint 与 compact best 不同，则二者等概率成为唯一结构
锚点；相同时直接选择该节点。

一个批次的 active 状态、global best、主轨迹、锚点、参考和 prompt context
在生成 sibling 前冻结。两个 sibling 都相对同一快照计算 `delta_global_best`。
批次结束后在全部有效 sibling 中确定唯一 global-best winner；完全相同质量时
使用确定性代码/Action 顺序消除生成顺序影响。

### 9. 有效候选与失败边界

只有通过代码解析并获得有限数值 fitness 的程序才能形成节点、边和轨迹。

- LLM transport 失败：记录错误，不消耗 evaluator 预算；
- Action/Code parse 失败：记录解析失败，不进入图；
- evaluator runtime/timeout/invalid result：消耗 evaluator 预算，记录失败类型，
  不进入图；
- NaN/Inf：按 `invalid_result` 拒绝，不进入图。

有效子代记录相对父代、路线 compact best 和批前 global best 的三种有向变化，
以及 operator、Action、参考、代码变化和 LOC 变化。

### 10. 种群管理

目标 active 数为 30。达到 60 个唯一 active endpoint 时收缩到 30：

1. 保留包含 global best 或相同 global-best executable state 的路线；
2. 从其余路线中按 $Q$ 保留 3 个质量精英；
3. 剩余席位按 $V=0.8Q+0.2P$ softmax 无放回抽样；
4. 其它路线归档，事实节点和边不删除。

V7 不维护 semantic diversity reserve。可执行状态差异由精确 hash 不变量保证；
不同代码是否具有功能互补性留给 LLM proposal 和 evaluator 判断。

### 11. 复杂度规则

复杂度只使用非空 LOC，并且只作为 fitness 完全相同的 tie-break。严格更优的长
程序仍优于较短程序。LOC 不以连续惩罚项或独立加权项进入 Q/P/UCB；当两个程序
fitness 完全相同时，它会打破 Q 百分位、compact best、锚点和 global best 的
并列。复杂度不作为候选拒绝条件，也不触发独立 simplify 算子。

## 实现不变量与可复现性

### 12. 上下文协议

实验必须显式提供正的 context input limit，默认 runner 使用 24576 token。
构造 prompt 时从最长 8 步历史开始，逐步缩短历史直到符合上限。

双轨迹 Action context 若超限，由调用方重新选择单轨迹算子并构造一次单轨迹
context；双轨迹 Action 虽然可放入、但对应 Code prompt 超限时，整批重新选择
单轨迹算子并重新生成 Action。任何实际发送给 LLM 的 prompt 都不得超过配置的
input limit。

### 13. Checkpoint 与身份校验

V7 协议 ID 为 `traceaad-v7-v1`，checkpoint schema 为 9。checkpoint 保存：

- 完整节点、边、轨迹和 active/archive 状态；
- Q/P/V、访问数、global best 及样本顺序；
- evaluator 预算、批次、失败和停滞状态；
- RNG 状态和 profiler 累计信息；
- 完整搜索配置；
- 任务描述、模板、evaluator 类型/关键设置、LLM 类型/模型/端点的非密钥身份。

恢复时要求协议、搜索配置和 runtime identity 完全一致，并验证图拓扑、code hash、
LOC、有限 fitness、轨迹边对齐、best route 以及 active endpoint 唯一性。旧版本
checkpoint 不自动迁移。

### 14. 完整流程

```text
1. 在预算内生成并评价初始程序：
   - 每个有效程序写入事实层；
   - 相同 code_hash 只保留一条 active 路线；
   - 直到得到 30 个唯一 active endpoint 或预算耗尽。
2. 重复直到预算耗尽或安全停止：
   a. 冻结当前 active 和 global-best 快照；
   b. 计算 Q、P、V 和预算衰减 UCB，选择主轨迹；
   c. 从 endpoint / compact best 选择唯一主锚点；
   d. 等概率选择可用语义算子；
   e. 双轨迹时按 Q-softmax 选择参考路线；
   f. 在 context hard limit 内构造共享历史；
   g. 生成两条 Action，再分别生成完整代码；
   h. 真实评价候选，失败不入图；
   i. 有效候选写入单父节点、边和新轨迹；
   j. 在批次快照上确定唯一 global-best winner；
   k. 精确 endpoint 去重，重复路线只归档繁衍状态；
   l. 重新评分；60 个唯一 active 时收缩到 30；
   m. 周期保存 checkpoint。
3. 返回 fitness 最优、完全同分时 LOC 最少的程序。
```

### 15. 默认配置

| 配置 | V7 默认值 |
| --- | ---: |
| 评估预算 | 1000 |
| 初始/目标唯一 active endpoint | 30 |
| 管理阈值 | 60 |
| 质量精英 | 3 |
| 每轮 sibling Action | 2 |
| 轨迹节点上限 | 8 |
| (Q_{endpoint}/Q_{best}) | 0.7 / 0.3 |
| (Q/P) | 0.8 / 0.2 |
| 趋势折扣 | 0.8 |
| 趋势阈值 | (10^{-12}) |
| 初始 UCB 系数 | 0.25 |
| UCB 时间因子 | 剩余预算比例 |
| softmax 温度 | 0.2 |
| 四算子 | 参考可用时等概率 |
| 锚点 | endpoint / compact best 等概率 |
| 参考 | 排除主轨迹后按 Q-softmax |
| active 去重 | 精确 `code_hash` |
| 生存 | global best + 3 个 Q 精英 + Q/P softmax |
| semantic diversity reserve | 无 |
| 复杂度 | fitness 完全同分时择短 |
| Action / Code token 上限 | 1024 / 8192 |
| runner context input limit | 24576 |
| 在线全局经验 | 无 |
| checkpoint schema | 9 |
| 默认实验目录 | `traceaad_v7/version7/` |

## 预期结果与证据边界

### 16. 实验边界

V7 当前已经完成独立实现、runner 接入和机制测试，尚无正式搜索或 held-out
结果。V7 的 endpoint 去重、预算衰减 UCB 和生存简化是待验证假设，不能根据
代码正确性宣称已经超过 V5 或 MCTS-AHD。预期证据必须分别覆盖机制是否按定义运行、
搜索过程中的状态与选择变化，以及最终搜索质量和 held-out 表现；这些证据完成前，
不能将预期结果表述为已观察结果。

正式实验前按研究笔记中的顺序完成四任务冒烟、单变量消融、三重复 1000 预算
搜索和全部 held-out 测试。结果完成前不更新最终结果页。
