# TraceAAD V9.5 完整机制设计

> 状态：已按本文档完成独立实现、机制测试、TraceAAD 回归测试与四任务
> 最小真实模型冒烟；尚未进行正式实验。Evidence、Generation、Quality-Guided Optimistic
> Allocation 及其必要生命周期边界已经冻结为 V9.5 第一版。
> 文中的机制判断与科学假设不等同于实验结论。

## 1. 主张与机制概览

### 1.1 为何需要改进历史

算法改进是逐步引入思想、试错和修正的过程。一次修改值不值得做、应该往哪个方向做，取决于
这条路线此前从什么方案出发、引入过什么思想、得到什么结果。TraceAAD 的基本主张是：**算法
改进历史应作为下一次修改的重要条件。**

V9.5 把这个主张落实为三个核心机制：

1. **Anchor-Centered Local Evidence**：围绕当前历史状态抽取 recent formation corrections
   与 exact-state direct attempts。
2. **Anchor-Centered Evidence-Conditioned Generation**：模型读取当前完整代码和局部证据，
   生成一个 optional Idea + mandatory Full Code candidate。
3. **Quality-Guided Optimistic Allocation**：全局搜索只依据当前质量与已获得的 candidate
   机会数，决定下一份预算给哪个历史状态。

三者的职责严格分开：

```text
历史事实内容 -> EvidenceBuilder -> 改变 LLM generation
当前质量 q + state 访问次数 n + 固定尺度 s -> Budget allocation
```

历史的具体内容主要在 generation 中发挥作用。历史 outcome 只经两条通道影响后续搜索：改变
下一次生成读到的证据内容，以及使该 state 的机会计数增加。Allocation 本身不估计 trajectory
productivity。

### 1.2 核心对象

V9.5 使用 **Search Forest** `F_t`，而不是单根 Search Tree。初始化会产生 K 个独立 root
states；forest 保存截至时刻 `t` 已发生的全部状态、attempt、分支和评价事实。

| 对象 | 定义 |
| --- | --- |
| `ProgramArtifact` | 由 evaluator 实际执行代码唯一确定的程序及其真实 fitness |
| `AnchorState a` | 某个 artifact 在一条具体形成历史中的搜索状态，拥有自己的 direct attempts 与 `n(a)` |
| `Lineage L(a)` | 从某个 root state 到 `a` 的唯一真实形成路径 |
| `AttemptRecord` | 一份已经完成 candidate 生命周期、可供 EvidenceBuilder 使用的 proposal 事实 |
| `E_t(a)` | 围绕 `a` 抽取的 formation corrections 与 exact-state direct attempts；不含当前代码 |
| `c_{t+1}` | 尚未生成、尚未评价的下一份 candidate；这是当前循环中唯一的未来对象 |

祖先、已经生成的子代和兄弟分支都属于过去事实。结构上的“下游”不等于时间上的未来。

### 1.3 完整信息链

```text
Search Forest F_t
    -> choose AnchorState a
    -> lineage L(a) + exact-state direct attempts
    -> E_t(a) = Extract(F_t, a)
    -> artifact(a).evaluator_input_code + E_t(a)
    -> LLM outputs optional Idea + mandatory Full Code
    -> parse / normalize / diff / evaluate-or-reuse
    -> finalize AttemptRecord and optionally create Artifact/State
    -> F_{t+1}
```

`Lineage` 是客观形成路径；`E_t(a)` 是为了下一次生成从 forest 中抽取的局部证据视图。两者
不得在 schema、日志或论文描述中混称。

## 2. 状态与事实模型

### 2.1 ProgramArtifact 与 AnchorState

模型的真实决策条件是 `(code, evidence)`。因此唯一 executable program 与一次具体历史状态
不能合并为同一个节点对象。

`ProgramArtifact` 保存代码与评价事实：evaluator 实际执行的输入代码及其 hash、evaluator
contract hash、真实 fitness 与有向 fitness、代码规模和首次发现次序。搜索内部统一定义越大
越好的有向质量：

```text
q(p) = fitness(p)      maximize task
q(p) = -fitness(p)     minimize task
```

`AnchorState` 保存形成历史与搜索访问状态：所属 artifact、parent state、incoming attempt、
depth、创建次序，以及该状态自己的 candidate 机会计数 `n`。完整字段见附录 A.1。

对 anchor `a`，`q(a)` 来自其 artifact；`n(a)` 只属于该 AnchorState。同一 artifact 可以由
不同独立路径到达并形成不同 states，因为这些 states 的 formation、direct attempts 和下一次
`E_t(a)` 可能不同。

### 2.2 Attempt 的原子生命周期

Candidate budget 和 `n` 的逻辑计数点是 **response completion**，但只有完成整个 candidate
处理后，AttemptRecord 才能被 EvidenceBuilder 使用。顺序固定为：

```text
LLM response completed
    -> atomically persist PendingAttempt + candidate_count += 1
       + n(anchor) += 1 when an anchor exists
    -> parse / build evaluator input / diff / cache lookup / evaluate
    -> atomically finalize AttemptRecord
       + optionally create ProgramArtifact and AnchorState
    -> EvidenceBuilder may read the finalized attempt
    -> global reselection
```

`PendingAttempt` 是 checkpoint 中的恢复状态，不是 prompt evidence。若进程在 parse 或 evaluator
期间中断，resume 必须继续处理同一 pending response，不得重新调用模型，也不得再次增加
budget 或 `n`。EvidenceBuilder 只查询 `status=finalized` 的 AttemptRecords，半成品事实不能进入
历史上下文。

完成后的 AttemptRecord 保存 anchor/child state 与 artifact 归属、declared idea、raw code 与
evaluator input 的 hash、actual diff 及其统计、父子 fitness 与有向增量、outcome 与 kind、失败
类别与反馈，以及 evaluator 是否被调用和 candidate 次序（附录 A.1）。其中两个枚举承担机制
含义：

- `direct_outcome` 为 `improve / plateau / regress / invalid`。有效或重复 root 没有 parent
  comparison，因此可以为 null。
- `attempt_kind` 区分 `root_new`、`root_duplicate`、`new_artifact`、`cached_artifact`、
  `no_op`、`repeated_duplicate`、`ancestral_return` 与 `invalid`。

EvidenceBuilder 根据同一份 AttemptRecord 生成不同 evidence view：valid child 是
CorrectionEvidence；invalid 是 FailureEvidence；no-op、repeated duplicate 与 ancestral return
是对应的 AttemptEvidence。无效 proposal 会成为真实历史，但不会被伪造成 correction edge。

### 2.3 严格 normalization contract

Artifact identity、actual diff 和 fitness cache 必须基于 evaluator 真正执行的同一份代码。
V9.5 固定以下不变量：

1. 从模型响应提取 `raw_code`，保存 `raw_code_hash`；
2. 通过确定性的 task parser 构造 `evaluator_input_code`；
3. evaluator **逐字执行 `evaluator_input_code`**；
4. `evaluator_input_hash` 对这份实际输入计算，artifact key 为
   `(evaluator_contract_hash, evaluator_input_hash)`；
5. actual diff 只比较 parent 与 candidate 的 `evaluator_input_code`；
6. cache 只按同一个 artifact key 复用 fitness。

Normalization 的允许范围严格限定为任务执行契约必需且确定性的提取、换行和包装处理：不做 AST
重写、import 重排或变量重命名，也不删除 comment 与 docstring。若以后增加任何转换，evaluator
仍必须执行转换后的准确结果，并通过单独测试证明 artifact identity 与执行输入一致。

这条 contract 防止 evaluator 执行 raw code、系统却用另一份 normalized code 合并 cache 的
错误。ProgramArtifact 保存 evaluator input；不同 raw responses 的 hash 保存在各自 attempts。

## 3. Anchor-Centered Local Evidence

### 3.1 Evidence 的唯一来源

给定 Search Forest `F_t`、AnchorState `a` 和事件预算 `B`：

$$
E_t(a)=\operatorname{Extract}(F_t,a;B)
=E_{recent\ formation}(a)+E_{direct}(a).
$$

两个来源分别回答一个问题：

1. **Recent Formation Corrections**：`a` 最近怎样形成；
2. **Exact-State Direct Attempts**：从这个具体 AnchorState 出发已经试过什么。

当前 artifact 的完整代码是与 `E_t(a)` 并列的输入，不计入历史事件预算。

失败信息只有一条通道：当前 anchor 的 invalid proposal 作为 direct FailureEvidence 进入
`E_t(a)`。其他 states 的失败不会以全局摘要、失败库或隐藏 prompt 字段再次注入。

### 3.2 Direct attempts

所有满足 `attempt.anchor_state_id == a.state_id` 且已经 finalized 的 records 构成 direct pool，
包括 improve、plateau、regress、invalid、no-op、repeated duplicate 和 ancestral return。
引用相同 ProgramArtifact 的其他 AnchorStates 不会混入。

Direct 是最高优先级来源，因为下一次生成仍从同一个 `(artifact, local history state)` 出发。

### 3.3 Recent formation

沿 `parent_state_id` 从 root state 到 `a` 的 incoming attempts 构成真实 lineage：

```text
root_state --e1--> ... --ek--> a
```

Formation 只取 lineage 末端最近的有效 corrections，并保持真实形成顺序。更早的 facts 直接
省略，既不做 prefix summary 或统计摘要，也不追加一次 LLM 总结。

### 3.4 Direct evidence 去重与选择

原始 AttemptRecords 全部保留，但 prompt selection 先对 exact repeated evidence 去重，避免多个
等价 attempts 占满 8 个槽位。每组保留 `candidate_order` 最新的代表：

```text
if evaluator_input_hash exists:
    evidence_key = (evaluator_input_hash, direct_outcome, failure_category)
elif raw_code_hash exists:
    evidence_key = (raw_code_hash, direct_outcome, failure_category)
else:
    evidence_key = (attempt_kind, failure_category, sha256(failure_feedback))
```

这个去重只影响 prompt view，不删除事实，也不减少 candidate budget 或 `n`。审计工件记录
代表 attempt 及其折叠的 attempt ids。

默认 `MAX_EVIDENCE_ITEMS=8`。在去重后的 direct representatives 上执行：

1. 按 `improve / plateau / regress / invalid` 分组；
2. 每个非空 outcome 先取最近一个代表；
3. 剩余位置按 `candidate_order` 从新到旧补足；
4. direct 不足时，用最近 formation corrections 补足；
5. prompt 内 formation 和 direct 分别按真实时间从早到晚展示。

选择规则完全由 outcome 覆盖与 recency 确定，不引入 fitness 排名、embedding、语义相似度、
额外 LLM 或 learned retriever。冻结的 selector id 为：

```text
v95_dedup_direct_outcome_coverage_then_recent_formation_v1
```

### 3.5 Rich Record 与 Minimal Prompt View

AttemptRecord、state id、candidate order、diff statistics、hash 和完整 diff 属于审计事实，不
全部进入小模型 prompt。

有效 correction 的最小展示为：

```text
Idea: <short label or unavailable>
Change: <deterministic diff excerpt>
Result: <improve / plateau / regress; parent fitness -> child fitness>
```

Invalid 展示为：

```text
Idea: <short label or unavailable>
Failure: <bounded verified failure category/message>
```

No-op、duplicate 和 ancestral return 只展示简短 Change/Result。Prompt 不展示 state id、
attempt id、candidate order、attempt kind 名称或 diff statistics。diff 被截断时只追加简短
`[diff truncated]` marker；完整 diff 和统计始终保存在本地工件。

### 3.6 Context 截断

Prompt 优先级固定为：

1. task/function contract、fitness 方向、当前完整 evaluator input code、输出契约；
2. selected evidence 的 Idea/Change/Result 或 Idea/Failure；
3. diff excerpt 细节。

超限时先缩短 diff context，再删除最早 formation，最后删除 direct 中按 recency 补入的事件。
实际进入 prompt 的代表 attempt ids、折叠 ids、excerpt hash、token count 和删除原因必须可审计。

正式配置使用 32768-token 总上下文与 8192-token 最大输出，因此完整 chat-templated prompt 最多
占用 24576 tokens。Token count 必须由当前生成服务对完整 messages、generation prompt 和实际
chat-template 参数计算；tokenizer 请求短暂失败时重试，持续失败记为 infrastructure failure，
不得以 UTF-8 字节数代替 token 数并据此报告 context overflow。

正式运行前必须用最大 code-output token bound 验证最小 prompt 可容纳。若只保留 task、当前
代码和输出契约仍然 overflow，则属于 configuration failure，而不是交给在线 eligibility
controller 处理的搜索事件。

## 4. Evidence-Conditioned Generation

### 4.1 输入

单个 candidate 的条件输入严格为：

$$
(Task, artifact(a).evaluator\_input\_code, E_t(a)).
$$

Prompt 顺序为：

```text
Task contract and fitness direction
Current anchor fitness and full evaluator-input code
Recent Formation Corrections
Direct Attempts from This Exact Anchor State
Minimal output contract
```

模型看到的条件到此为止：没有 global failure feedback、operator、reference、allocation score
或内部调度字段。核心指令保持简短：

```text
Improve the current algorithm using the provided search history. Preserve useful
mechanisms, consider previously tested modifications and their outcomes, and propose
one coherent modification. Historical outcomes are evidence rather than strict
prohibitions; previously unsuccessful ideas may be revisited with a materially
different implementation.
```

### 4.2 输出：Code mandatory，Idea optional

模型被要求输出：

````text
Idea: <one short semantic label>
Code:
```python
<full executable implementation>
```
````

但 candidate 有效性的硬条件只有：**能够提取一份完整 Code**。Idea 是 best-effort optional：

- Idea 存在时保存简短文本；
- Idea 缺失、为空或无法单独解析时，`declared_idea=null`；
- Evidence prompt 以后展示 `Idea: unavailable`；
- 不得因为 Idea 缺失丢弃一份可解析、可执行的 Code。

Idea 不是 reasoning，也不是对真实修改的权威描述。Actual diff 才是系统可验证的代码变化。
模型只被要求给出 Idea 与 Code 两个字段，输出契约中没有 Decision、Evidence used、
Refine/Explore 分类、chain-of-thought、diff 或 patch。

冻结的 generation policy id 为：

```text
v95_anchor_evidence_optional_idea_full_code_v1
```

### 4.3 单候选生成

Candidate multiplicity 固定为 `m=1`：

```text
select one anchor
    -> build its current evidence
    -> obtain one completed candidate response
    -> finalize its lifecycle
    -> reselect globally
```

同一 anchor 可以连续被选中，但后一次读取写回后的 `E_{t+1}(a)`。每份 candidate 完成后都重新
进行全局选择，一次选择不携带 follow-up commitment。

修改方式由模型隐式决定：V9.5 不预定义 `ideate / refine / synthesize / transfer` operator
portfolio，行为分类只可用于离线分析。

### 4.4 Full Code 与 system-derived diff

```text
Generation:      Anchor Code + Evidence -> Optional Idea + Full Code
Fact derivation: Parent Evaluator Input + Candidate Evaluator Input -> Actual Diff
Evidence:        Idea or unavailable + Actual Diff/Failure + Observed Result
```

V9.5 是 correction-aware search，不是 edit-based code search。修改记录由系统从父子 evaluator
input 推导，模型既不输出也不应用 patch。

## 5. Quality-Guided Optimistic Allocation

### 5.1 预算分数

对任一有效 AnchorState `a`：

```text
q(a) = artifact(a) 的有向真实 fitness
n_t(a) = 已经以该 state 为起点完成的 candidate response 数
```

令 `s >= 0` 为初始化后固定的一步变化尺度：

$$
\boxed{S_t(a)=q(a)+\frac{s}{\sqrt{n_t(a)+1}}}.
$$

每轮在全部有效 AnchorStates `A_t` 上选择：

$$
a_t^*=\arg\max_{a\in A_t}S_t(a).
$$

完全同分时依次选择 `n(a)` 更小、创建更早、`state_id` 更小者。这样 `s=0` 且 `q` 同分时，
较少获得机会的 state 优先。

### 5.2 准确含义

`S(a)` 只表达当前 executable quality 与尚未充分开发程度。它是确定性的预算优先级，不是
expected return、trajectory value 或统计置信上界。

之所以不把它当作价值估计，是因为同一 state 每次被选择后 direct evidence 都可能变化：

$$
\pi_\theta(\cdot\mid a,E_1(a))
\neq \pi_\theta(\cdot\mid a,E_2(a)).
$$

生成分布随证据改变，过去 direct gains 就不是同一 productivity distribution 的重复样本。
V9.5 因此不计算 mean gain、trend、momentum、volatility、maturity，也不在 ancestor 与
descendant 之间传递 credit。

结果历史只通过两条路径影响后续：

1. finalized AttemptRecord 改变 `E_{t+1}(a)`；
2. completed response 使 `n(a)+=1`，降低 scarcity bonus。

成功 child 的质量已经由新的 executable state 表达，无需把正增益复制回 parent。

### 5.3 行为边界

新 state `x` 的分数为 `S(x)=q(x)+s`。它获得有限 optimism，但不会获得无穷大或强制扩展。
这可能偏好高质量附近的局部精炼，并错过需要先大幅退步的路线；V9.5 接受这一可证伪边界，
不为理论深谷恢复多步 credit。

`argmax` 在全部有效 states 上进行，选择权完全由 `S(a)` 决定，其上不再叠加 Top-K、clade
quota、固定利用—覆盖节奏、active population 或第二层 UCB。

## 6. Search Lifecycle

### 6.1 Initialization

默认 `K=8`，用于与既有实验保持连续性；K 是显式配置，不是理论常数。

```text
Task -> LLM -> Optional Idea + Mandatory Full Code
```

Root 生成与后续生成使用同一条通道：不先生成 strategy cards，不调用 planning agent，也不为
roots 人工指定 operator。每份完成 root response 都消耗一份 candidate search budget。有效且
此前未出现的 artifact 创建一个 root AnchorState；invalid 或重复 root 记录 finalized attempt
并继续，直到得到 K 个唯一有效 roots。Transport failure 按工程重试，不计 candidate budget。

随后每个 root state 恰好完成一次 bootstrap：使用同一个 GenerationPolicy，完成 response 后
该 root 的 `n+=1`。Valid child 按正常 artifact/state 规则处理；invalid、no-op、duplicate 和
ancestral return 进入 root 的 direct attempts。

初始化不做 Top-K 淘汰，所有 root states 与有效 bootstrap child states 都进入全局候选集合。
若预算耗尽前无法形成 K 个唯一 roots 或完成 K 次 bootstrap，run 标记为
`initialization_failure`，不进入正式比较。

### 6.2 固定 optimism scale

`s` 需要与任务的真实一步变化幅度同量级，因此由 bootstrap 观测直接给出。对 bootstrap 中所有
创建有效 child state 的 transitions 计算：

$$
d_i=|q(child_i)-q(root_i)|.
$$

改善和退步都进入 `D_init`；invalid、no-op、repeated duplicate 和 ancestral return 不进入。
跨路径命中已有 artifact 但合法创建 child state 时，其父子差值可以进入。

$$
s=
\begin{cases}
\operatorname{median}(D_{init}), & D_{init}\neq\varnothing,\\
0, & D_{init}=\varnothing.
\end{cases}
$$

若全部 `d_i=0`，中位数自然为 0。估计量只有中位数本身，不附加 IQR fallback、epsilon 或
task-specific coefficient。正式搜索开始后 `s` 永久固定，resume 恢复原值，后续结果不能重估。

### 6.3 Candidate search budget

正式方法预算称为 `candidate_search_budget`：一份预算是一份由模型完成返回的 candidate
response。

| 事件 | Candidate budget | `n(anchor)` | Finalized attempt | Evaluator call | New state |
| --- | ---: | ---: | ---: | ---: | ---: |
| valid new artifact | +1 | +1 | yes | yes | yes |
| cached artifact, legal new relation | +1 | +1 | yes | no | yes |
| no-op | +1 | +1 | yes | no | no |
| repeated duplicate | +1 | +1 | yes | no | no |
| ancestral return | +1 | +1 | yes | no | no |
| parse/runtime/timeout/non-finite invalid | +1 | +1 | yes | as applicable | no |
| transport failure without completed response | 0 | 0 | no | no | no |

Root response 同样增加 candidate budget，但没有起点 state，因此没有 `n(anchor)`。

运行分别记录 LLM requests、completed candidates、evaluator calls、valid artifacts、states、
invalid、no-op、duplicate、ancestral return 和 cache hit。这些计数不能互相替代。

### 6.4 Invalid 与 transport failure

模型完成 response 后，无法提取 mandatory Code、syntax error、runtime error、evaluator timeout、
NaN 或非有限 fitness 都是一次真实 proposal：消耗 candidate budget，增加 `n(anchor)`，最终
形成 FailureEvidence，不创建 artifact/state。

Idea 缺失不属于 invalid。只有 mandatory Code 或执行结果失败才使 candidate invalid。

HTTP error、连接中断或服务端没有完成 response 属于 transport failure。它只进入工程错误日志，
按显式 `transport_retry_limit` 重试；重试耗尽时 run 标记为 infrastructure failure，不能伪装成
正常停止或搜索 invalid。

### 6.5 Duplicate、ancestral return 与 cache

Candidate 成功构造 evaluator input 后，按以下顺序处理：

1. `candidate_artifact == artifact(anchor)`：记录 `no_op`，不建 state；
2. candidate artifact 出现在 `ancestor_artifact_ids(anchor)` 中：记录 `ancestral_return`，不建
   state；
3. `(anchor.state_id, candidate_artifact_id)` 关系已经存在：记录 `repeated_duplicate`，不建
   state；
4. artifact 全局存在且不属于以上情况：复用确定性 fitness，创建新的 AnchorState；
5. artifact 不存在：调用 evaluator；有效时创建 ProgramArtifact 与 AnchorState，无效时记录
   invalid。

这套顺序保留了有意义的汇聚，同时切断了刷 optimism 的循环：独立分支 `A -> X` 与 `B -> X`
可形成不同 states，而同一 lineage 不能通过 `X -> Y -> X -> ...` 循环创建无限 `n=0` states。
冻结的 identity policy 为：

```text
v95_parent_state_artifact_relation_no_ancestral_return_v1
```

No-op 的 outcome 为 plateau。Repeated duplicate、ancestral return 和 cached artifact 的 outcome
由已有 artifact 与 parent artifact 的有向质量差按统一规则确定。

Fitness cache 要求同一 evaluator contract 下同一 evaluator input 的结果确定。正式任务必须在
冒烟前通过重复评价验证；若 evaluator 有随机性，ProgramArtifact 单 fitness 与 cache 均不适用，
不能静默沿用本规格。

### 6.6 Anchor eligibility 与事实保留

所有引用有效 ProgramArtifact 的 AnchorStates 都有资格，包括 improve、plateau 和 regress
child states。低质量 state 是否再次获得预算完全由 `S(a)` 决定，其上没有 active/archive
population、退步阈值、定期剪枝或 Top-K survival。

Artifacts、states 和 finalized attempts 一经产生就永久保留。Pending attempt 只在恢复完成后
转为 finalized fact。

### 6.7 停止与最终选择

Core 的唯一正常停止条件是 `candidate_search_budget exhausted`。连续无改善、trajectory
maturity、confidence threshold 与 convergence detector 都不构成停止条件。Initialization、
transport、evaluator infrastructure 或 configuration failure 是失败 run，不是算法正常提前停止。

最终只在唯一 ProgramArtifacts 中按真实 objective 选择：

$$
p^*=\arg\max_p q(p).
$$

完全同分时先选 `len(evaluator_input_code)` 更小者，再选首次发现更早者。Allocation score、
AnchorState history 和 `n(a)` 永远不参与最终答案选择。

## 7. 完整算法

```text
Generate K unique valid root artifacts/states
    -> one bootstrap response for each root
    -> build first local evidence
    -> set fixed s from valid bootstrap transitions

while candidate_count < candidate_search_budget:
    a = deterministic_argmax_all_states(q(a) + s / sqrt(n(a)+1))
    E = Extract(F, a, max_items=8)
    response = LLM(Task, artifact(a).code, E)

    if transport failure:
        retry without budget/n change
        continue

    atomically:
        persist PendingAttempt(response)
        candidate_count += 1
        n(a) += 1

    resume-safe candidate processing:
        parse mandatory Code; Idea optional
        build exact evaluator input and deterministic diff
        classify invalid / no-op / ancestral return / repeated duplicate
        otherwise evaluate new artifact or reuse deterministic cache

    atomically:
        finalize AttemptRecord
        optionally create ProgramArtifact and AnchorState
        clear PendingAttempt

return best unique ProgramArtifact by true objective
```

## 8. 本版机制范围

V9.5 被有意约束为一个最小可检验对象。四个界面各只保留一组通道：

**Evidence.** 只有 exact-state direct attempts 与 recent formation corrections 两个来源，全部
来自 finalized AttemptRecords 的真实字段。Descendant 与 follow-up 摘要、formation ancestor 的
sibling/counterfactual attempts、early-formation prefix summary、subtree/branch best 与 delayed
development 标签、global failure memory、global reflection 与 global free-text summary、Idea
Bank、cross-lineage reference，以及任何由第二次 LLM 调用生成的历史摘要都不在本版范围内。

**Generation.** 一次调用、一个 candidate、optional Idea + mandatory Full Code，修改类型由模型
隐式决定。Operator portfolio、显式 refine/explore 分类、model-generated patch 应用、sibling
batch 与 multi-step rollout 都不在本版范围内。

**Allocation.** 只有 `q`、`n` 和固定 `s`。Trajectory trend、momentum、volatility、maturity、
mean gain、ancestor/descendant/correction 可传递 credit、artifact-level 共享 `n`、learned
retriever、surrogate、critic 与 RL 都不在本版范围内。

**Lifecycle.** 全部有效 states 参与 argmax，预算按 completed candidate response 计，停止条件
只有预算耗尽。Top-K、clade quota、active/archive population、在线剪枝、convergence detector 与
无改善提前停止都不在本版范围内。

这些信息通道和控制层未被采用，不等于它们无效。删除它们是为了让被检验的对象足够小、让机制
差异可归因；其中若有成分确实有价值，应由后续版本在有过程证据的前提下逐项加入。方法版本
各自独立实现，本版不为旧版本保留兼容路径。

## 9. 风险与科学边界

### 9.1 Actual diff 不是因果解释

一次父子 diff 可能包含多个耦合变化，fitness 也可能受 evaluator 协议影响。V9.5 只能声称 prompt
包含了更真实的修改记录，不能声称系统识别了导致改善的具体代码行。

### 9.2 Full Code 可能产生无关重写

完整代码生成可能引入与 Idea 无关的变化或代码漂移。第一版仍使用 Full Code，因为 patch
generation 会增加定位、格式、应用和上下文匹配失败。Actual diff 只负责记录，不按 diff 大小
拒绝 candidate。

### 9.3 Direct 可能占满上下文

当一个 state 有至少 8 个去重后的 direct representatives，formation 不进入本轮 prompt。这是
direct 优先的明确取舍，必须报告发生频率；本版不为 formation 保留隐藏 quota。

### 9.4 被省略的历史可能有价值

Deep descendants、counterfactual siblings 和早期 formation 可能有用。本版删除它们是为了
识别最小问题，不等于证明这些信息无效。

### 9.5 Allocation 具有局部搜索偏置

`S(a)` 不预测下一 candidate 或最终 global best。它可能低估需要先大幅退步的路线；不能称为
learned value、expected return 或无偏估计。

### 9.6 初始化尺度可能不稳定

有效 bootstrap transitions 数量很小，中位数会随初始化样本变化。正式报告必须给出 `D_init`、
最终 `s`、零尺度状态和敏感性 replay，不能把 data-derived scale 描述为理论常数。

### 9.7 跨分支重复 artifact 仍可能增加状态数

Ancestral-return 约束阻止同一 lineage 循环洗 optimism，但独立分支仍可汇聚到同一 artifact 并
形成不同 states。这是 history-conditioned generation 的有意结果。必须报告 state/artifact 比、
cache hit、重复 artifact states 的预算占比。

### 9.8 Cache 依赖 evaluator 确定性与 normalization contract

若同一 evaluator input 在同一 contract 下可能得到随机结果，或 hash code 与真实执行 code
不一致，fitness cache 会错误合并状态。实现必须验证这两个前提，不能根据接口名称假设安全。

## 10. 科学假设

在正式实验前，V9.5 只提出以下可验证假设：

1. 相比只提供 Idea 与标量 outcome，在 local evidence 中加入 system-derived actual diff，是否
   改善下一次 code generation 的质量、有效率或有限预算搜索结果；
2. 相比固定 topology-nearby window，exact-state direct attempts + recent formation 的局部证据，
   是否改善有限预算搜索；
3. 相比纯 `q` 选择，固定尺度的 completed-response-count optimism 是否改善预算覆盖与最终
   held-out quality。

“Actual diff 与真实代码变化更对齐”和“exact-state direct 与当前 state 更匹配”属于表示定义，
不是待实验重复证明的性能结论。联合版本变强也不能自动归因到任一单独组件；独立作用需要
相应消融和过程证据。

---

# Appendix A：Schema 与 Prompt Contract

## A.1 状态与事实 schema

```text
ProgramArtifact {
    artifact_id
    evaluator_contract_hash
    evaluator_input_hash
    evaluator_input_code
    fitness
    directed_fitness
    code_length
    program_loc
    first_discovery_order
}

AnchorState {
    state_id
    artifact_id
    parent_state_id | null
    incoming_attempt_id | null
    depth
    creation_order
    generation_count_n
}

AttemptRecord {
    attempt_id
    status = finalized
    anchor_state_id | null
    child_state_id | null
    artifact_id | null

    declared_idea | null
    raw_code_hash | null
    evaluator_input_hash | null
    actual_diff | null
    diff_statistics | null

    parent_fitness | null
    child_fitness | null
    directed_delta | null
    direct_outcome | null
    attempt_kind

    failure_category | null
    failure_feedback | null
    evaluator_called
    candidate_order
    creation_time
}
```

## A.2 Frozen policy ids

```text
evidence_selector_id = v95_dedup_direct_outcome_coverage_then_recent_formation_v1
generation_policy_id = v95_anchor_evidence_optional_idea_full_code_v1
candidate_multiplicity_policy_id = v95_single_candidate_reselect_v1
budget_policy_id = v95_quality_guided_optimistic_allocation_v1
initialization_policy_id = v95_k_independent_roots_one_bootstrap_v1
optimism_scale_policy_id = v95_median_valid_bootstrap_abs_delta_v1
state_identity_policy_id = v95_parent_state_artifact_relation_no_ancestral_return_v1
candidate_accounting_policy_id = v95_completed_response_budget_v1
stop_policy_id = v95_candidate_budget_exhaustion_v1
normalization_policy_id = v95_evaluator_input_is_artifact_identity_v1
```

## A.3 Parser contract

- Full Code mandatory；Idea optional。
- Idea 缺失时 `declared_idea=null`，不影响 Code validity。
- 只能从 response 中提取一个主 candidate Code。
- Parser 输出 raw-code hash、evaluator-input code/hash 和明确 failure category。
- Actual diff 只由父子 evaluator inputs 计算。

## A.4 Prompt contract

- 输入只含 Task、当前 code 和 `E_t(a)`。
- Evidence event 只显示 Idea/Change/Result 或 Idea/Failure。
- 不显示内部 id、order、kind、diff statistics、allocation score 或 global memory。
- 输出不要求 reasoning、Evidence used、operator label 或 patch。

# Appendix B：配置、Checkpoint 与审计

## B.1 Method config

正式方法配置至少记录：

- protocol/design version；
- logical model name `Qwen3.6-27B`；
- task、maximize/minimize、candidate search budget；
- 附录 A.2 中全部 frozen policy ids 与实际参数；
- `K=8`、event budget `8`、fixed `s`、tie-break；
- context limit、max code-output tokens、diff excerpt rule；
- evaluator contract hash 与 deterministic-cache flag；
- transport retry limit；
- 所有方法内部随机种子。

配置只写本版实际采用的字段，不为未采用的机制预留空字段。

## B.2 Generator environment metadata

Generation operator 会被 decoding 配置改变，因此本地 environment manifest 还必须记录：

- logical model name；
- tokenizer identity/version 与 chat-template hash；
- temperature、top-p、top-k、max new tokens、sampling seed/seed support；
- max total context；
- serving API/software 名称与版本（若服务暴露）；
- prompt-renderer version/hash。

按照本仓库统一实验协议，正式结果不区分服务源，也不记录具体量化版本，统一写作
`Qwen3.6-27B`。因此这里保证的是逻辑模型、prompt 与 decoding protocol 层面的可追溯性，不声称
跨不同服务部署能够 bitwise reproduction。这一限制必须在实验说明中明确，而不能只留下模型名
后声称完整生成器复现。

## B.3 Checkpoint

Checkpoint 必须恢复：

- ProgramArtifacts、AnchorStates、finalized AttemptRecords；
- PendingAttempt 及其原始 response、计数已提交标记和处理阶段；
- artifact cache、ancestor-artifact index、parent-state relation index；
- lineage/direct indexes 与 selector identity；
- 每个 state 的 `n`、fixed `s`、global best；
- LLM request、candidate、evaluator call 和各 outcome counts；
- RNG states、config identity、schema version。

Resume 要求 schema 严格匹配，不匹配直接拒绝。恢复 pending candidate 时禁止重复模型调用、
预算计数或 `n+=1`。

## B.4 Rich audit artifacts

本地审计至少保存：

- root/bootstrap attempts 与 `D_init`；
- 每轮所有 states 的 artifact、`q`、`n`、optimism、`S`、tie-break 和选择；
- direct/formation pools、exact-evidence dedup groups、代表 attempts、折叠 ids 和截断原因；
- raw response/code hash、evaluator input/hash、完整 diff/hash/statistics；
- cache hit、ancestral membership、relation-exists、evaluator-called；
- final outcome/failure 与 optional state/artifact creation；
- prompt snapshot 与 minimal rendered evidence。

Prompt 使用 Minimal View，日志保存 Rich Record。原始 prompt、response、code、diff 和 evaluator
结果只留本地实验工件，不进入 Git。

# Appendix C：实现验证协议

## C.1 State、normalization 与 duplicate

- evaluator 执行输入必须与 artifact code/hash 逐字一致；
- 同一 artifact key 只创建一个 ProgramArtifact；
- `A->X` 重复不建 state；`A->X` 与独立 `B->X` 可建不同 states；
- `X->Y->X` 被识别为 ancestral return，不建新 state；
- cache hit 不调用 evaluator，outcome 仍按父子真实 q 计算；
- 对正式 tasks 重复评价同一 evaluator input，验证 cache 确定性前提。

## C.2 Evidence

- Formation/direct 映射到真实 finalized attempt ids；pending 不可见；
- 相同 artifact 的不同 states 不混用 lineage 或 direct attempts；
- exact repeated evidence 在 prompt selection 前折叠，但原始 records 全部保留；
- direct outcome coverage 与 recency 在 dedup representatives 上正确执行；
- 总事件不超过 8，且只含 §8 允许的两个来源；
- prompt 只含 Minimal View，Rich fields 只存在于工件；
- diff truncation marker、完整工件与 token-boundary test 正确。

## C.3 Generation 与 atomic attempt

- 有 Code、无 Idea 的 response 可以正常评价并保存 `declared_idea=null`；
- 无 Code 的 completed response 计 budget/n 并 finalize invalid；
- transport failure 不计 budget/n；
- response completion 后 pending、budget 与 n 原子提交；
- parse/evaluator 中断后 resume 同一 pending candidate，不重复调用/计数；
- finalize 后 attempt 才进入 EvidenceBuilder；
- response 只要求 Idea 与 Code 两个字段。

## C.4 Allocation 与 lifecycle

- maximize/minimize 统一成 directed `q`；
- score 只由 `q`、`n`、fixed `s` 决定；
- 所有有效 states 参加 argmax；
- K 个唯一 roots、每 root 一次 bootstrap、初始化不淘汰；
- `s` 严格等于有效 bootstrap deltas 的 median，空集为 0；
- candidate budget 精确停止；
- 最终只按唯一 artifact 的 objective/code length/discovery order 选择。

## C.5 Resume 与 smoke

- checkpoint/resume 后 forest、pending/finalized attempts、cache、RNG、预算和下一选择一致；
- 四任务最小真实模型 smoke 验证模型调用、评价方向、分数区分、state 更新与工件闭合；
- smoke 只证明机制运行，不作为性能证据。

# Appendix D：正式实验与报告

## D.1 正式协议

- Qwen3.6-27B；
- CVRP、Online Bin Packing、OP、TSP；
- 每任务三个独立重复；
- 每 run 1000 candidate proposals，并报告实际 evaluator calls；
- 全部重复完成后进行完整 held-out test。

V9.5 内部预算是 completed candidate responses。跨方法报告必须同时给出各自正式方法预算和
共同 evaluator-call 截面，不能把 1000 proposals 静默等同于 1000 evaluations。

## D.2 分开报告三类证据

**Search results：** best-so-far、最终 search-set best、candidate/evaluator counts、有效率与失败率。

**Held-out results：** 每 task/scale 三重复 mean ± sample SD、泛化方向与重复稳定性。

**Process evidence：**

- formation/direct 可用与入选数、dedup 折叠率、截断率；
- direct 占满 8 条与 formation 缺席比例；
- Idea missing rate、actual diff availability、schema/Code parse failure；
- invalid/no-op/repeated duplicate/ancestral return/cache-hit rates；
- `D_init`、fixed `s`、`q/n/S` 分布和 optimism 改变 argmax 的频率；
- state/artifact ratio、重复 artifact state 预算占比、root-clade/state/artifact 预算集中度；
- pending recovery 与 transport/configuration failures。

过程差异是机制诊断，不直接构成因果结论。正式质量结论只来自完成批次的 held-out
`results.json`。

## D.3 消融顺序

联合版本足够强且运行健康后，优先比较：

1. current code only vs. 完整 local evidence；
2. Idea + outcome vs. optional Idea + actual diff + outcome；
3. formation + direct vs. direct only；
4. `q+s/sqrt(n+1)` vs. pure `q`；
5. `0.5s / s / 2s` sensitivity；
6. `m=1` vs. multiple siblings 仅作为后续预算单位扩展。

# Appendix E：设计依据与其边界

[研究认识](../knowledge/研究认识.md)给出历史条件化单步生成与有限预算持续搜索的总体认识。
[V9.3/V9.4 联合分析](../analysis/TraceAAD-V9.3-V9.4机制与实验分析.md)用于暴露既有选择机制的
缺口，不能证明 V9.5 有效。

[DGA²D 阅读笔记](../references/LLM自动算法设计方法阅读笔记/51-DGA2D.md)和
[MEMOIR 阅读笔记](../references/LLM自动算法设计方法阅读笔记/47-MEMOIR.md)只支持修改记录、
branch-local history 与 cross-branch knowledge 的一般表示启发，不直接证明 V9.5 的 selector、
generation 或 allocation 有效。

V9.5 的最终科学结论必须来自完成的正式搜索、held-out evaluation、过程审计和相应消融。
