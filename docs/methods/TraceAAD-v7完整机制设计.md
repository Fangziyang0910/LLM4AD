# TraceAAD V7：可执行状态唯一的轨迹进化

> 本文定义正式 TraceAAD V7 机制；正式实验批次为 `v7_20260804_001931`
> （协议 `traceaad-v7` / checkpoint schema 11），结果见
> [实验总汇](../results/实验总汇.md)。
> V7 的设计来源、证据和待验证假设见
> [V5 机制全面分析与 V7 设计](../research/TraceAAD-V5机制全面分析与V7设计.md)。
> V4/V5/V6 继续由各自文档和独立实现维护。V7 的正式训练与 held-out 已完成；本文不把
> 结果页数字重复写入机制定义。

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

$$
\tau_i=(n_0,e_1,n_1,\ldots,e_k,n_k),\quad k\le 7.
$$

轨迹记录 endpoint、当前保留路径的 compact best、访问次数、状态、质量 $Q$、
趋势 $P$ 和调度值 $V$。路径超过上限时从最早端截断，完整派生图不删除。若从内部
compact-best 锚点分支，锚点之后的旧尝试以最多 8 条 `evidence_edge_ids` 携带到
新路线；这些边是知识证据，不是第二结构父代，也不改变路径长度。

### 3. 可执行状态唯一性

active frontier 满足不变量：

$$
\forall \tau_i\ne\tau_j\in\mathcal A,
\operatorname{hash}(endpoint_i)\ne\operatorname{hash}(endpoint_j).
$$

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

初始化目标是 30 个唯一 active endpoint。重复有效程序仍消耗候选 evaluator 预算并
写入事实层，但若已有相同 `code_hash`，直接复用已知 finite fitness，不重复调用
evaluator；它仍不增加 active 数。连续尝试没有增加唯一 active 状态时按停滞上限
停止，达到预算上限时即使不足 30 个唯一状态也停止。

### 4. 历史上下文

主轨迹始终对应当前唯一结构锚点。选择 endpoint 时展示它的形成历史；选择内部
compact best 时，历史分为：

- `[How This Program Was Reached]`：锚点形成过程；
- `[Later Attempts From This Program]`：锚点之后已测试的尝试。
- `[Carried Route Evidence]`：此前从内部锚点分支时携带的旧尝试，避免新路线重复相同
  的失败边界。

每条边按时间顺序只展示 Requested change、父子 fitness、相对父代结果、路线推进
量、路线 best 更新原因和是否产生 global breakthrough。模型自述的 Implemented Idea、代码变化比例和
LOC 只保留在事实日志中，不进入生成上下文。主锚点完整展示代码，历史边不重复
完整代码。

双轨迹算子另外展示参考轨迹及其 compact-best 程序。参考轨迹只说明另一条路线
有哪些具体思想和试错结果；子程序仍写入主轨迹。

Action 阶段接收轨迹事实和可选参考程序，用于决定下一步修改。Code 阶段只接收
当前主程序、已选 Requested Modification 和必要的参考程序代码，负责忠实执行
该修改，不重新解释完整搜索历史。

### 5. 四个语义算子

| 算子 | 约束 |
| --- | --- |
| `trace_ideate` | 提出历史中尚未尝试的具体方向；失败只约束同一实现，不否定整个思想 |
| `trace_refine` | 只修改一个已有机制，并指向路线推进证据或一个明确暴露的弱点 |
| `trace_synthesize` | 从两条不同程序路线各取一个有结果支持的原则，在一个具体接口交互 |
| `trace_transfer` | 保持主程序结构，只迁移参考路线中主程序尚未拥有的一个有结果支持的原则 |

有至少两条 active 路线时，四算子等概率选择。只有一条 active 路线或双轨迹
上下文不可用时，只在 `ideate/refine` 中等概率选择。算子不维护在线 reward、
Elo 或成熟度状态。

双轨迹参考排除主轨迹以及与主锚点具有相同 `code_hash` 的路线后，按路线质量
$Q$-softmax 选择，使用参考路线 compact-best。若没有不同的可执行参考程序，
退化为单轨迹算子。参考选择不增加参考路线访问数。

### 6. Action 与 Code

Action 阶段通过 OpenAI-compatible `response_format=json_schema` 请求结构化输出。
schema 只包含一个 `actions` 字符串数组，数组长度为 1–2；每条 Action 是单行、
自包含的自然语言修改，说明具体改动并遵守当前算子语义。Code 阶段从唯一主
锚点实现 Action，返回完整可执行 Python 程序，并由程序 parser 和 evaluator 校验。

默认 Action 输出上限 1024 token，每条 Action 另有 600 字符上限；Code 输出上限
16384 token。每一条有效
Action 单独生成和评价一个候选；JSON 解析或 schema 后校验失败不进入图，也不消耗
evaluator 预算。V7 不接受编号文本或其他非 JSON 兜底；若服务端未执行 schema，
该次 Action 明确记为协议失败。

### 7. 轨迹评分

#### 7.1 成果质量

endpoint 和 compact best 分别在当前 active 集合中做并列感知百分位排名：

$$
Q_i=0.7Q_{\mathrm{endpoint},i}+0.3Q_{\mathrm{best},i}.
$$

程序比较首先看 evaluator fitness；仅在 fitness 完全相同时，非空 LOC 更少者
优先。最大化和最小化任务先统一为有向 fitness。

#### 7.2 近期趋势

对保留路径的每条父子边，使用子程序相对该路线在该边之前的 compact-best 的有向
fitness 变化 $Delta_j=Delta_{mathrm{route_best}}$：

$$
s_j=\begin{cases}
+1,&\Delta_j>10^{-12}\\
0,&|\Delta_j|\le10^{-12}\\
-1,&\Delta_j<-10^{-12}.
\end{cases}
$$

用折扣 $d=0.8$ 让近期边权重更高：

$$
P_i=\frac{1}{2}\left(
1+\frac{\sum_{j=1}^{k}d^{k-j}s_j}
{\sum_{j=1}^{k}d^{k-j}}
\right).
$$

无历史边时 $P_i=0.5$。调度基础值为：

$$
V_i=0.8Q_i+0.2P_i.
$$

#### 7.3 剩余预算衰减 UCB

评估预算为 $B$，已经评价 $b$ 个程序，路线访问数为 $n_i$，全局批次选择计数
为 $N$。$N$ 是从搜索开始发起过的主轨迹 sibling 批次数，不是当前 active
路线访问数之和，因此归档路线不会让探索时钟回退。剩余预算比例：

$$
r_b=\operatorname{clip}\left(\frac{B-b}{B},0,1\right).
$$

父代调整值：

$$
A_i=V_i+0.25r_b
\sqrt{\frac{\log(1+N)}{1+n_i}}.
$$

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
- exact `code_hash` cache hit：消耗候选预算并进入事实层，但复用已知 fitness，
  不再次执行 evaluator；该优化要求 evaluator 对相同代码和固定数据/seed 是确定的，
  当前正式 runner 的任务配置满足这一条件。
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

复杂度只使用非空 LOC，并且只作为 fitness 完全相同的 tie-break。该 tie-shorter
比较结果同时写入 parent outcome、route-best update 和 global-best update，确保
评分、趋势和上下文使用同一比较器。严格更优的长程序仍优于较短程序。LOC 不以连续惩罚项或独立加权项进入 Q/P/UCB；当两个程序
fitness 完全相同时，它会打破 Q 百分位、compact best、锚点和 global best 的
并列。复杂度不作为候选拒绝条件，也不触发独立 simplify 算子。

## 实现不变量与可复现性

### 12. 上下文协议

实验必须显式提供正的 context input limit，默认 runner 使用 24576 token。
构造 Action prompt 时从最长 8 步历史开始，优先保留主轨迹，再缩短参考轨迹；
内部锚点同时保留锚点形成证据和锚点之后的最近尝试，直到符合上限。

双轨迹 Action context 若超限，由调用方重新选择单轨迹算子并构造一次单轨迹
context。Code prompt 不重复完整历史，因此只在当前程序、Action 和参考程序本身
超限时回退。任何实际发送给 LLM 的 prompt 都不得超过配置的 input limit。

### 13. Checkpoint 与身份校验

V7 协议 ID 为 `traceaad-v7`，checkpoint schema 为 11。checkpoint 保存：

- 完整节点、边、轨迹和 active/archive 状态；
- Q/P/V、访问数、global best 及样本顺序；
- evaluator 预算、批次、失败和停滞状态；
- RNG 状态；

运行工件与其他 TraceAAD 版本共用 `TraceAADArtifacts` 三分开契约（`logs/` 监控、`artifacts/` 原始分析、`checkpoints/` 续训）。checkpoint 不保存 profiler/事件计数。
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
| $Q_{\mathrm{endpoint}}/Q_{\mathrm{best}}$ | 0.7 / 0.3 |
| $Q/P$ | 0.8 / 0.2 |
| 趋势折扣 | 0.8 |
| 趋势阈值 | $10^{-12}$ |
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
| Action / Code token 上限 | 1024 / 16384 |
| Action 输出协议 | strict JSON Schema：`{"actions": [...]}` |
| runner context input limit | 24576 |
| 在线全局经验 | 无 |
| checkpoint schema | 11 |
| 默认实验目录 | `traceaad_v7/version7/` |

## 预期结果与证据边界

### 16. 实验边界

V7 已完成独立实现、runner 接入、机制测试、四任务三重复正式搜索和完整 held-out。
endpoint 去重、预算衰减 UCB 和生存简化仍是机制假设；正式结果只能说明它们在当前
任务、模型、预算和重复协议下的联合行为，不能据此宣称已普遍超过 V5 或 MCTS-AHD。
实现事实、搜索过程和 held-out 结果仍需分开报告。

正式批次已按四任务、三重复、1000 evaluator 和完整 held-out 协议完成。后续若开展
单变量消融，应建立独立批次，不覆盖 `v7_20260804_001931` 的权威工件。
