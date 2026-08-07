# TraceAAD V9-Core：以真实匹配的改进历史指导下一步搜索

## 1. 设计动机

算法设计是逐步改进的过程。一个程序从某个初始方案出发，经过思想引入、实现、评价、
延续、修复、回退和重组，才形成当前状态。每个中间程序都留下两类可用于下一步决策的
事实：它是怎样形成的，以及从它出发已经尝试过什么。

V9-Core 的做法是：把与当前程序严格匹配的真实改进历史交给 LLM，让它据此决定下一步
算法修改。搜索位置和生成预算由树调度稳定给出，历史是唯一变化的信息来源。

这条链是：

```text
程序改进产生真实历史
    -> 历史暴露有效思想、失败尝试和当前路线边界
    -> LLM 据此选择并实现下一步修改
    -> 新程序获得真实 evaluator 结果
    -> 结果进入树并成为后续历史
```

V9-Core 的设计只围绕这条链。树搜索负责在有限预算下决定从哪个程序继续，
四个算子负责限定怎样使用历史；它们都是支撑机制，不替代“利用算法改进历史指导
下一步”这一中心。

## 2. 为什么先做 Core

TraceAAD 的多个历史版本同时改变过种群、生存、记忆、轨迹信用、算子、生成调用、树宽
和树深。联合版本的最终结果只能评价整套系统，无法说明某个机制是否有效。V8.3 的正式
结果和后续修正还表明，更多信用公式与结构约束可能改变搜索形态，却不保证最终质量。

V9-Core 因此采用奥卡姆剃刀：只保留表达核心思想所必需的机制。运行可靠性不属于可删减的部分；
checkpoint、配置身份校验、失败分类、原始工件和完整 provenance 全部保留。

各版本提供的主要经验如下：

| 来源 | V9-Core 吸收的经验 | V9-Core 的处理 |
| --- | --- | --- |
| V5 | 匹配当前程序的轨迹、四个语义算子和双轨迹参考具有清楚语义 | 保留四算子和真实路径，取消 active population |
| V8 | 完整单父树能保存所有有效中间状态 | 保留完整树和唯一结构父代 |
| V8.2 | 10 根、深层开发、自适应 `new_child` 与单次 `Idea + Code` 构成强而简洁的搜索骨架 | 作为 V9-Core 的搜索基线 |
| 原始 V8.3 | 固定深度和高回报 `new_child` 会造成根锁定或 sibling 堆积 | 不采用固定深度和 V8.3 信用公式 |
| V8.3-credit | 更复杂的信用公式改变搜索形态，但不改变调度的基本职责 | 不采用趋势、路线信用和概率控制器 |

V9-Core 在算法行为上有意接近 V8.2。独立的包、协议和 checkpoint 使历史输入成为唯一
可变量。

## 3. 机制边界

V9-Core 保留：

- 完整单父代程序树和固定虚拟根；
- 10 个评价有效的初始根程序；
- subtree-best max-backup；
- 递归 UCT 和自适应 `new_child`；
- `trace_ideate`、`trace_refine`、`trace_synthesize`、`trace_transfer`；
- 当前程序、最近形成历史和已测试直接分支；
- 双轨迹算子的其他根分支参考程序及其形成历史；
- 每个候选一次 LLM 调用直接输出 `Idea + Code`；
- 每次扩展最多两个 sibling；
- fitness 优先、完全同分时非空 LOC 择短；
- 完整 checkpoint、配置身份校验和原始运行工件。

V9-Core 不引入：

- active/archive population 或生存淘汰；
- 全局自由文本经验、反思或课程记忆；
- 独立 Action/description LLM 调用；
- `trace_simplify` 或新的算子集合；
- 在线代码去重、语义去重或 fitness cache；
- 趋势、路线信用、算子信用、Elo 或多维价值公式；
- progressive widening、根扩展或固定树深/树宽上限；
- 按 LOC 连续惩罚 fitness。

## 4. 搜索状态：完整单父树

### 5.1 虚拟根

虚拟根 $r$ 不包含代码和 fitness，只保存固定初始根节点、访问数、全树最好值及其节点。
初始化完成后，虚拟根只在已有根分支之间选择，不再生成新根。

### 5.2 程序节点

每个评价有效程序形成一个 `ProgramNode`：

```text
ProgramNode = {
    id, code, idea,
    fitness, directed_fitness,
    program_loc, code_hash,
    parent_id, incoming_edge_id, child_ids, depth,
    visit_count, expansion_count,
    subtree_value, subtree_best_node_id,
    creation_order, batch_id, operator,
}
```

最小化任务使用 $y(n)=-fitness(n)$，最大化任务使用 $y(n)=fitness(n)$，统一为越大越好
的有向 fitness。原始 fitness 始终保留并用于任务报告。

每个非根程序只有一个结构父代。参考程序只提供生成信息，不构成第二父代。因此，任意
节点都唯一确定一条形成路径：

$$
\tau(n)=(n_0,e_1,n_1,\ldots,e_k,n_k=n).
$$

### 5.3 改进边

父节点 $p$ 到子节点 $c$ 的 `ImprovementEdge` 保存：

- 实际实现的 `Idea` 和所用算子；
- 参考节点及参考根分支；
- 相对父节点、批前 global best 的有向 fitness 变化；
- `improve / plateau / regress` 结果；
- LOC 变化、代码变化比例；
- 是否刷新 global best 及刷新原因；
- iteration、batch、sibling 和 evaluator sample 顺序。

边记录可追溯事实，不把局部改善解释为思想的长期因果信用。

## 5. 真实匹配历史

### 6.1 “匹配”的定义

当前扩展节点为 $n$ 时，V9-Core 只提供由 $n$ 的真实结构位置确定的历史：

1. 当前程序的完整代码和原始 fitness；
2. 从所属初始根到 $n$ 的最近形成边；
3. 已经从 $n$ 直接测试过的代表性分支；
4. 双轨迹算子使用的参考程序及其真实形成路径。

历史不会从其他节点随机借用，不由 LLM 自行总结，不包含尚未发生的推断，也不把某条
历史配给不对应的代码。这就是本版唯一的 `matched_history` 协议。

### 6.2 当前节点的形成历史

`[How This Program Was Reached]` 默认展示最近 8 条祖先边。每步包含：

- 已实现的思想；
- 父子 fitness 和结果分类；
- 是否刷新当时的 global best；
- LOC 与代码变化比例。

历史不重复祖先完整代码，避免上下文随树深线性增长。当前节点完整代码始终展示。

### 6.3 已测试直接分支

`[Previously Tested From This Program]` 默认最多展示 8 个直接子分支：

1. 先选 subtree-best 最高的 4 个分支；
2. 剩余席位选择最近创建且未入选的分支；
3. 最后按创建顺序展示。

每个摘要包含已实现思想、直接结果、该分支实际达到的 subtree-best fitness，以及从直接
子节点到该最好节点的深度。它帮助模型避免重复失败尝试，也允许识别“直接退步但后来
发展出价值”的真实路线。subtree best 只表示到达事实，不表示思想已被因果证明。

### 6.4 上下文边界

运行必须显式提供 `context_token_limit`。若完整上下文超限：

1. 双轨迹时先逐步缩短参考形成历史；
2. 再逐步减少当前节点的直接分支摘要；
3. 双轨迹仍无法容纳时退化为本轮可用的单轨迹算子；
4. 必要任务契约、当前完整程序和当前形成历史仍无法容纳时，本轮记录
   `context_overflow`，不发送超限请求。

V9-Core 不通过删除当前程序或伪造历史来满足上下文限制。

## 6. 初始化

初始化目标是获得 10 个评价有效程序，而不是只发起 10 次请求。每个初始候选由任务、
目标函数和多样性提示直接生成 `Idea + Code`，经 evaluator 返回有限 fitness 后成为虚拟
根的直接子节点。

第一个候选要求简单、完整、有效；后续候选提示避开最近最多 6 个有效初始思想。初始化
中的 LLM transport/parse 失败不消耗 evaluator 预算；evaluator 一旦启动就消耗预算，
失败程序不入树。

初始化在以下任一条件满足时结束：

- 得到 10 个有效根程序；
- evaluator 总预算耗尽；
- 连续 20 次生成尝试未得到可解析程序；
- 搜索被观测层安全中止。

若没有任何有效根程序，搜索明确以 `empty_tree` 停止。

## 7. 四个轨迹语义算子

每轮从当前可用算子中 seeded 等概率选择：

| 算子 | 下一步决策语义 | 额外参考 |
| --- | --- | --- |
| `trace_ideate` | 根据形成历史和已测试边界提出尚未尝试的新方向 | 无 |
| `trace_refine` | 聚焦修正一个已显示价值或暴露弱点的机制 | 无 |
| `trace_synthesize` | 让当前分支与另一根分支各自有支持的原则产生功能交互 | 另一根分支 |
| `trace_transfer` | 保持当前程序核心结构，迁移另一根分支的一个有支持思想 | 另一根分支 |

算子只限定主要改进意图，不是代码静态分类器。若树中不存在合格的其他根分支，双轨迹
算子不可用，本轮只在 `trace_ideate` 和 `trace_refine` 中选择。V9-Core 不根据短期
命中率在线调整算子概率。

## 8. 双轨迹参考选择

设当前节点所属根分支为 $root(n)$。参考候选按以下步骤构造：

1. 排除 $root(n)$；
2. 对每个其他根分支取其当前 subtree-best 程序；
3. 排除与当前程序 `code_hash` 完全相同的代表；
4. 使用该根分支 subtree value 的全树中秩百分位做 softmax；
5. 温度默认 $\tau=0.2$，抽取一个参考根分支及其代表程序。

若候选 $i$ 的归一化质量为 $Z_i$，则：

$$
P(i)=\frac{\exp((Z_i-\max_j Z_j)/\tau)}
{\sum_j\exp((Z_j-\max_k Z_k)/\tau)}.
$$

参考程序及其最近 8 条真实形成边进入 prompt。参考选择不增加其访问数，不改变其子树
价值，也不接收主路径的 backup。

## 9. 单次 `Idea + Code` 生成协议

每个 child slot 只进行一次 LLM 调用，输出：

````text
Idea: <一句话说明实际实现的算法修改>
Code:
```python
<完整函数实现>
```
````

prompt 由任务契约、fitness 方向、真实匹配历史、当前完整程序、算子约束、目标函数签名
以及可选参考历史和代码组成。`Idea` 必须在第一个代码围栏之前显式出现。初始化候选以
原始模板为基础；直接子代以当前节点的完整程序为解析模板，因此当前节点已经使用的
imports、顶层 helper 和其他函数在模型省略时仍会保留。生成目标函数的参数列表和返回类型
必须与模板契约完全一致，签名漂移视为 parse 失败；解析器同时保留模板/当前程序的
preface，并允许小型顶层 helper。缺失 Idea、缺失代码围栏或无法恢复目标函数均视为
parse 失败。

一次 expansion batch 默认包含两个 sibling。二者共享选中节点、算子、参考节点和批前
global best，只通过候选序号提示生成不同实现。V9-Core 不增加 Action 规划调用，也不在
代码生成后调用 LLM 撰写 description。

## 10. 子树价值与 global best

节点的子树价值使用乐观 max-backup：

$$
G(n)=\max\left(y(n),\max_{c\in Children(n)}G(c)\right).
$$

每个有效新节点写入后，从父节点沿祖先路径重算 $G$ 和 subtree-best pointer，虚拟根同步
保存全树最好节点。

global best 的确定规则为：

1. 先比较原始任务 fitness 的正确方向；
2. fitness 完全相同时选择非空 LOC 更少的程序；
3. fitness 与 LOC 都相同时选择 node id 更小的程序。

一个 batch 中可能有多个候选优于批前 best，但只有按上述规则最好的一个记录为本 batch
的 global-best winner。复杂度不进入连续搜索分数，也不压过任何严格 fitness 改善。

## 11. 全树中秩归一化

选择只使用当前全树有效节点的有向 fitness 作为参照。对数值 $x$，设严格小于它的数量
为 $L$，等于它的数量为 $E$，全树节点数为 $M$：

$$
Z(x)=\frac{L+(E-1)/2}{M-1},\qquad M>1.
$$

当只有一个值或全树全部同分时，定义 $Z(x)=0.5$。中秩百分位保留排序、正确处理并列，
且不易被单个离群值压缩。它只用于搜索调度和参考抽样，不修改原始 fitness。

## 12. 递归 UCT 选择

总 evaluator 预算为 $T$、已启动评价数为 $t$ 时：

$$
r_t=\operatorname{clip}\left(\frac{T-t}{T},0,1\right).
$$

无限预算时定义 $r_t=1$。从父节点 $p$ 进入已有子节点 $c$ 的分数为：

$$
S_{down}(c\mid p)=Z(G(c))+\lambda_0r_t
\sqrt{\frac{\log(1+N(p))}{N(c)}}.
$$

默认 $\lambda_0=0.1$。每轮先在固定根分支间按该分数选择，然后在程序节点内部递归比较
`new_child` 与全部 `descend`。最高分完全相同时使用本次运行的 seeded RNG 打破并列。

## 13. 自适应 `new_child`

LLM 修改空间不可枚举。V9-Core 为每个程序节点设置一个内部调度选项 `new_child`，用于
判断是从当前程序再开一批直接分支，还是沿已有子树继续深入。它不产生额外 LLM 调用。

### 14.1 Expansion batch 回报

设节点 $n$ 已启动 $B(n)$ 个 batch，第 $b$ 个 batch 的有效子节点集合为 $C_b$：

$$
R_b(n)=\max_{c\in C_b}Z(G(c)).
$$

若 batch 没有有效子节点，回报为 0。某个子节点以后发展出更好后代时，其所在 batch 的
回报随 $G(c)$ 更新；这表示该 batch 实际开出了一条后来有价值的路线，不等价于思想因果
信用。

### 14.2 新分支分数

尚无扩展经验时，当前程序自身质量是局部修改的先验。默认先验权重 $\beta=1$：

$$
Q_{new}(n)=
\frac{\beta Z(y(n))+\sum_{b=1}^{B(n)}R_b(n)}
{\beta+B(n)}.
$$

$$
S_{new}(n)=Q_{new}(n)+\lambda_0r_t
\sqrt{\frac{\log(1+N(n))}{1+B(n)}}.
$$

在节点 $n$，`S_new` 与所有 `S_down` 直接竞争。`new_child` 获胜则在 $n$ 生成一个最多
两个 sibling 的 batch；某个子节点获胜则进入该节点继续递归。叶节点没有已有子节点，
因此首次到达必然扩展。

该机制不设置 child 上限或深度上限。树形由实际结果、访问统计、剩余预算和价值竞争
共同形成。V9-Core 明确保留这一 V8.2 基线，以免在历史消融前同时改变搜索骨架。

## 14. 访问、扩展和失败记账

一次可执行 batch 在发送生成请求前，对虚拟根和选中路径各增加一次访问，并令选中节点
的 `expansion_count` 增加 1。两个 sibling 共享这一次 batch visit，不按候选数重复增加。

失败语义如下：

| 阶段 | evaluator 预算 | 建节点 | batch 回报 |
| --- | ---: | ---: | ---: |
| 上下文无法容纳 | 不消耗 | 否 | 不启动 batch |
| LLM transport 失败 | 不消耗 | 否 | 0 |
| Idea/Code parse 失败 | 不消耗 | 否 | 0 |
| evaluator timeout/runtime/invalid | 消耗 | 否 | 0 |
| NaN/Inf 或非数值 fitness | 消耗 | 否 | 0 |
| 有限数值 fitness | 消耗 | 是 | 按该 batch 最佳子树质量 |

连续 20 个搜索 iteration 未增加有效节点时以 `stalled_generation` 停止；连续 20 次 LLM
transport 失败时由观测层安全中止。已启动 evaluator 即计预算，运行不会超出配置的
`max_sample_nums`。summary 的终态严格区分：预算耗尽为 `finished`，空树或生成停滞为
`stalled`，观测层安全中止为 `aborted`，未捕获异常为 `error`；具体触发原因写入
`stop_reason`。transport 失败虽然不消耗 evaluator 预算，但每次请求都会以
`status=transport` 写入 `artifacts/llm_calls.jsonl`。

## 15. 完整搜索流程

```text
Input: evaluator budget T, initial roots K=10, siblings m=2

1. 创建空虚拟根。
2. 在预算内生成并评价程序，直到得到 K 个有效根或触发停止条件。
3. 重复直到 evaluator 预算耗尽或安全停止：
   a. 用全树有向 fitness 建立中秩参照分布，计算剩余预算比例；
   b. 从虚拟根开始，递归比较已有子节点 UCT 与当前节点 new_child；
   c. 在 new_child 获胜的程序节点停止下降；
   d. 选择可用的四类轨迹算子；
   e. 双轨迹算子从其他根分支抽取 subtree-best 参考；
   f. 构造当前程序、最近形成边、已测试直接分支和可选参考历史；
   g. 若上下文有效，记录一次路径访问和 expansion batch；
   h. 调用 LLM 生成最多 m 个 Idea + Code sibling；
   i. 对每个可解析程序进行真实评价；
   j. 每个有限 fitness 候选写入一个节点和一条唯一父边；
   k. 沿祖先路径 max-backup，并更新 batch 回报和 global best；
   l. 写入候选、边、LLM 调用、选择决策和 checkpoint。
4. 返回 fitness 最优、完全同分时 LOC 最少的程序。
```

## 16. Checkpoint 与运行工件

### 17.1 Checkpoint

V9 checkpoint schema 1 至少保存：

- 协议标识和完整搜索配置；
- 任务、evaluator、模型和非密钥运行身份摘要；
- 虚拟根、全部节点、全部边及创建顺序；
- 每个节点的访问、扩展、subtree value 和 best pointer；
- global best、evaluator 计数、batch/attempt 计数和停滞状态；
- seeded RNG 完整状态。

恢复时校验树连通、无环、单父代、深度、fitness、LOC/hash、边 provenance、batch 一致性、
subtree backup、global best、搜索配置和运行身份。V8/V8.2 checkpoint 即使字段结构相似，
也因协议和 schema 身份不同而禁止直接恢复。

### 17.2 原始工件

```text
<run>/
  run_config.json
  logs/progress.log
  logs/errors.jsonl
  logs/summary.json
  artifacts/candidates.jsonl
  artifacts/edges.jsonl
  artifacts/llm_calls.jsonl
  artifacts/decisions.jsonl
  checkpoints/latest.json
```

`decisions.jsonl` 保留完整选中路径、每层 `descend/expand` 选项、质量、原始 subtree value、
访问/扩展次数、最终分数、算子、参考 provenance、送入 prompt 的边 id 和 backup 变化。
运行时只落盘事实与在线决策；长期贡献、因果信用和消融结论由离线分析计算。
从 checkpoint 续跑时，profiler 会从已有 JSONL 重建 evaluator、候选、边、决策、错误和
LLM 调用的累计计数，并沿用原始 summary 的 `started_at`（若存在），不会只报告续跑片段。

## 17. 实现不变量

- 虚拟根不包含程序或 fitness，初始化完成后不再扩展；
- 每个有效程序只有一个结构父代，所有有效程序永久保留；
- 当前节点唯一确定当前完整程序和它的形成路径；
- 当前历史只来自当前节点祖先边和当前节点真实直接子分支；
- 双轨迹参考必须来自不同根分支，且不形成结构父代或 backup；
- 每个 child slot 只有一次 `Idea + Code` LLM 调用；
- 一次 batch 的两个 sibling 共享选择上下文和批前 global best；
- $G$ 只由真实有限 fitness 的自身和后代决定；
- 路径访问按 batch 记一次，失败 batch 仍增加扩展次数；
- `new_child` 与已有子节点统一竞争，不使用 progressive widening；
- 根数量、节点 child 数和树深以外不设隐藏结构门槛；
- fitness 严格优先，LOC 只处理完全同分；
- 不使用在线去重、全局经验、路线趋势、算子信用或额外反思调用；
- checkpoint 与日志只保证可靠性，不改变搜索行为。

## 18. 默认参数

| 参数 | V9-Core 默认值 | 作用 |
| --- | ---: | --- |
| `max_sample_nums` | 1000 | evaluator 总预算，含初始化；直接构造器和正式 runner 均默认此值 |
| `n_init` | 10 | 评价有效的固定根程序数 |
| `offspring_per_iteration` | 2 | 每个 expansion batch 的最大 sibling 数 |
| `ancestor_history_limit` | 8 | 当前和参考形成历史的最近边数 |
| `direct_child_limit` | 8 | 当前节点直接分支摘要上限 |
| `direct_child_top_count` | 4 | 直接分支中优先按子树质量选择的数量 |
| `reference_temperature` | 0.2 | 其他根分支参考 softmax 温度 |
| `exploration_constant` | 0.1 | UCT 初始探索系数 $\lambda_0$ |
| `expansion_prior_weight` | 1.0 | `new_child` 当前节点质量先验 $\beta$ |
| `code_max_tokens` | 8192 | 单次 Idea + Code 输出上限 |
| `context_token_limit` | runner 默认 24576，必须显式为正 | 输入上下文硬边界 |
| `max_consecutive_sample_failures` | 20 | LLM transport 连续失败上限 |
| `max_stalled_iterations` | 20 | 连续未增加树节点的 iteration 上限 |
| `checkpoint_interval` | 10 | 每多少个 batch 保存 checkpoint |
| `maximize` | `True` | 当前四个正式 task 的优化方向 |
| `random_seed` | 每个 repeat 显式提供 | 算子、并列和参考抽样复现 |

正式 runner 必须把这些值完整写入 `run_config.json`。任何会改变搜索行为的参数差异都
构成不同实验配置，不得在恢复时静默接受。

## 19. 与 V8.2、V8.3 的边界

| 维度 | V8.2 | V8.3 正式/credit | V9-Core |
| --- | --- | --- | --- |
| 树 | 完整单父树 | 完整单父树 | 完整单父树 |
| 根 | 固定 10 | 正式固定 10，后续版本曾扩根 | 固定 10 |
| 选择 | 确定性递归 UCT + adaptive new child | PW、路线信用、概率 UCB 等版本差异 | 固定采用 V8.2 基线 |
| 深度 | 无上限 | 正式版 10，credit 取消 | 无上限 |
| 生成 | 一次 Idea + Code | 两次调用生成 code/description | 一次 Idea + Code |
| 算子 | 四个 | 五个 | 四个 |
| 历史 | 最近 8 形成边 + 8 直接分支 | 局部 3 + 3 等版本差异 | 明确冻结为真实匹配 8 + 8 |
| 在线附加信用 | batch subtree best | 路线趋势/直接回报/均值信用等 | 无新增信用公式 |
| 协议身份 | `traceaad-v8.2-adaptive-expand` | V8.3 各自协议 | `traceaad-v9-core` |

V9-Core 继承 V8.2 的搜索行为是明确的控制设计。独立包、协议和 checkpoint 防止实验
工件混用；它们不被宣称为算法性能创新。

## 20. 在版本谱系中的位置

V9-Core 以 V8.2 的完整树和自适应扩展为搜索骨架，删除趋势、路线信用、算子信用、progressive widening、固定深度与在线去重，只保留解释核心主张所必需的机制：完整单父树、真实匹配历史、四个语义算子、subtree max-backup 与自适应 `new_child`。搜索行为接近 V8.2 是有意的控制设计，使历史输入成为唯一可单独改变的因素。

实验事实与机制诊断见 [版本实验事实与机制诊断](../analysis/TraceAAD-版本实验事实与机制诊断.md)。
