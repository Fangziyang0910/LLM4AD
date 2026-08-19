# TraceAAD V9.5：锚点证据与质量—机会分配

> 历史版本，10/12 个计划运行完成（TSP、CVRP 各缺一次）；过程诊断见 [V9.5–V9.6 机制诊断](../analysis/TraceAAD-V9.5-V9.6机制诊断.md)。V9.6 只改变历史表示与预算单位。

## 1. 研究目标

V9.5 把两个问题分开：给定当前程序时，哪些局部历史应进入生成；给定全部历史状态时，下一份生成机会应分配给哪里。

方法由三个组件组成：

1. **锚点局部证据**：当前锚点的 recent formation 与 exact-state direct attempts；
2. **证据条件生成**：当前完整代码加局部证据，生成一个 optional Idea 和 mandatory Full Code；
3. **质量—机会分配**：只用当前质量、锚点已获得的生成机会和固定尺度选择下一锚点。

历史内容只通过生成上下文发挥作用；分配器不计算趋势、成熟度、平均增益或祖先信用。

## 2. 搜索事实

搜索状态是一片由 8 个初始根开始的森林。V9.5 区分程序事实与历史状态：

| 对象 | 定义 |
| --- | --- |
| ProgramArtifact | evaluator 实际执行的一份唯一代码及其真实 fitness |
| AnchorState $a$ | 某个程序沿一条具体形成路径到达的状态，拥有独立历史和机会计数 $n(a)$ |
| Lineage $L(a)$ | 从根状态到 $a$ 的唯一形成路径 |
| AttemptRecord | 一次已经完成解析、评价或失败分类的生成事实 |
| $E_t(a)$ | 围绕 $a$ 抽取的局部历史视图，不含当前代码 |

同一 ProgramArtifact 可以沿不同路径形成多个 AnchorState；这些状态共享代码与 fitness，不共享形成历史、direct attempts 或 $n(a)$。搜索内部把任务目标统一为越大越好的有向质量：最大化任务取 $q=fitness$，最小化任务取 $q=-fitness$。

只有完成生命周期的 AttemptRecord 能进入后续历史。记录的 Idea 是模型声明；父子实际代码差异、评价结果和失败类别是系统可核验事实。

## 3. 锚点局部证据

给定森林 $F_t$、锚点 $a$ 和历史事件上限 $B=8$：

$$
E_t(a)=E_{direct}(a)+E_{recent\ formation}(a).
$$

### Direct attempts

direct pool 只包含从当前 AnchorState 出发的已完成尝试，包括 improve、plateau、regress、invalid、no-op、重复和祖先返回。引用同一程序但来自其他 AnchorState 的尝试不混入。

完全相同的证据先按执行代码与结果去重。选择时，先从 `improve / plateau / regress / invalid` 每个非空类别取最近一条，再按发生时间从新到旧补足。direct 优先占用 8 个位置；剩余位置才由 formation 填充。

### Recent formation

formation 沿 $L(a)$ 取最靠近当前锚点的有效形成事件，并按真实发生顺序呈现。更早事件直接省略，不生成全局摘要。

有效事件最小展示为：

```text
Idea: <declared idea or unavailable>
Change: <deterministic parent-child diff excerpt>
Result: improve | plateau | regress; parent fitness -> child fitness
```

无效尝试展示经过验证的失败类别。内部 id、hash、完整 diff 和统计字段不进入提示。

当前代码、任务契约和输出要求优先级最高。上下文超限时先压缩 diff 细节，再删除最早 formation，最后删除按 recency 补入的 direct 事件。正式设置为 32768 总上下文、8192 输出上限。

## 4. 证据条件生成

单个候选的条件输入为

$$
(Task,\ Code(a),\ E_t(a)).
$$

模型只输出一个候选：Idea 可缺失，完整可执行代码必须存在。Idea 不作为推理链或真实修改的权威说明；实际修改由系统比较父子 evaluator input 得到。

每次锚点选择只生成一个候选。候选处理完成后立即全局重选；不会因选中某锚点而承诺连续多步生成。V9.5 不预设 Refine/Explore 或其他 operator portfolio。

## 5. 质量—机会分配

对任一有效 AnchorState $a$，令 $q(a)$ 为其程序的有向真实质量，$n_t(a)$ 为已经从该状态完成的候选响应数。固定尺度 $s\ge0$ 下，预算分数为

$$
S_t(a)=q(a)+\frac{s}{\sqrt{n_t(a)+1}}.
$$

每轮选择

$$
a_t^*=\arg\max_{a\in A_t}S_t(a).
$$

完全同分时，依次选择 $n(a)$ 更小、创建更早的状态。该分数只是确定性的预算优先级，不是 expected return、trajectory value 或置信上界。成功子代以自己的质量成为新 AnchorState，不把正增益复制回祖先。

新状态的初始分数为 $q+s$。有限乐观项可能偏好高质量附近的局部细化，并低估需要先大幅退步的路线；V9.5 不加入多步信用来补偿这一偏置。

## 6. 初始化与尺度

初始化持续生成，直到获得 $K=8$ 个唯一有效根。随后每个根恰好接受一次同协议 bootstrap 生成。所有根和有效 bootstrap 子状态进入全局候选集合，不做 Top-K 淘汰。

对 bootstrap 中形成有效 child state 的转换，计算

$$
d_i=|q(child_i)-q(root_i)|.
$$

改善、退步和持平都进入 $D_{init}$；invalid、no-op、重复和祖先返回不进入。固定尺度为

$$
s=
\begin{cases}
\operatorname{median}(D_{init}), & D_{init}\neq\varnothing,\\
0, & D_{init}=\varnothing.
\end{cases}
$$

正式搜索开始后不再重估 $s$。它是初始化一步变化的启发式尺度，不是理论常数或路线潜力估计。

## 7. 预算、重复与最终选择

V9.5 的预算单位是**完成的候选响应**。解析失败、运行错误、评价超时、no-op、重复和祖先返回都消耗一份候选预算，并使起点锚点的 $n(a)$ 加一；模型传输失败不计预算。

若候选与当前程序相同，记为 no-op；若返回祖先程序，记为 ancestral return；若同一父状态到同一程序的关系已存在，记为 repeated duplicate。这三类都不创建新状态。若程序已在其他分支出现，可以复用确定性 fitness，并在当前历史下创建新的 AnchorState。

所有有效 AnchorState 都可参与后续选择，包括由退步产生的状态。事实一经形成不删除，也不使用 active/archive 生存淘汰。

正常停止条件是候选响应预算耗尽。最终只在唯一 ProgramArtifact 中按任务真实目标选择；完全同分时偏好代码更短、发现更早的程序。分配分数、历史和 $n(a)$ 不参与最终答案排序。

## 8. 算法

```text
Generate 8 unique valid root programs.
Generate one bootstrap candidate from each root.
Set s to the median absolute bootstrap quality change.

While completed-response budget remains:
    Score every anchor by q(a) + s / sqrt(n(a) + 1).
    Select the highest-scoring anchor.
    Build up to 8 direct-first local evidence events.
    Generate one optional Idea and one complete program.
    Count the completed response and increment n(anchor).
    Classify the result; evaluate or reuse deterministic fitness.
    Record the attempt and create a child state when allowed.
    Reselect globally.

Return the best unique program by the true objective.
```

## 9. 科学边界

- Actual diff 提高事实对齐，但多行耦合修改不能提供行级因果归因。
- Full Code 可能包含与声明 Idea 无关的重写；V9.5 不按 diff 大小拒绝候选。
- direct 优先可能占满 8 个位置，使 formation 完全缺席；这一缺口推动 V9.6 重构。
- 被省略的早期形成事件、后代和跨路线知识可能有价值；未进入本版不等于无效。
- $S(a)$ 不预测下一候选或最终突破，不能称为学习到的价值函数。
- 固定初始化尺度可能随 8 个 bootstrap 样本变化；零尺度与敏感性属于结果报告内容。
- V9.5 联合检验证据、生成与分配，不能识别任何单项的独立收益。
