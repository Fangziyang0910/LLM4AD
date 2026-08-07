# TraceAAD V8.3 正式版：以局部探索脉络驱动树搜索

> 本文定义原始正式 V8.3 协议 `traceaad-v8.3`，对应 Git 提交 `04aaef9`、训练批次
> `v83_20260805_final` 和 held-out 批次 `eval_best_v83_20260806`。V8.3-credit 是
> 后续未完成的选择信用修正版，其详细设计保留在
> [V8.3 初始化机制设计](TraceAAD-v8.3初始化机制设计.md)，不与本文的正式结果混用。

## 1. 研究问题

V8 的完整树保存了程序的结构来时路，但树路径本身仍可能过长，且调度器未必把当前节点
附近最相关的尝试交给 LLM。V8.3 研究的问题是：**如何从算法改进树中提取与当前程序匹配
的局部探索脉络，并用不同的改进意图指导下一次修改？**

V8.3 假设算法改进由中间程序、设计思想、实际代码和评价结果组成。保存这些事实后，LLM
可以判断当前程序如何形成、最近哪些思想产生了什么结果、从当前程序已经试过哪些方向，
再决定延续、修复、校准、简化或换向。MCTS 负责在有限预算下选择节点；轨迹脉络负责让
生成条件与当前代码保持一致。

## 2. 个体与生成协议

评价有效的算法个体为：

```text
AlgorithmRecord = {
    design_idea,
    code,
    description,
    fitness,
    evaluation_time,
}
```

`design_idea` 是 Call 1 在生成代码前提出的设计意图；`description` 是 Call 2 读取实际
代码后给出的事实描述。两者不互相替代。每个候选固定经过：

```text
Call 1: task + current algorithm + local context -> design_idea + code
Call 2: task + design_idea + code -> description
Evaluation: code -> finite fitness + evaluation_time
```

Call 1 的有效输出必须包含 `Design Idea:` 和代码围栏；Call 2 必须包含 `Description:`。
两次调用使用文本协议，不使用 JSON Schema。每个调用失败时在首次请求之外最多重试三次；
生成或解析失败不消耗 evaluator，不建立节点。evaluator 一旦启动即计一次预算，无论最终
成功、timeout、runtime error、输出缺失或 NaN/Inf。失败候选只写入运行记录，不入树。

## 3. 初始化

初始化目标是获得 10 个评价有效算法，而不是完成 10 次尝试。首个初始算法使用独立的
`i1` 语义；后续算法使用 `e1` 从已经得到的有效集合选择参考，生成新的完整算法。
参考只展示 `design_idea` 和完整 `code`，不展示 fitness、description、排名或 global best。
所有有效初始算法直接连接虚拟根，参考关系不改变结构父代。

初始化在以下任一条件满足时结束：有效根节点达到 10 个、evaluator 预算耗尽、连续 20
次尝试未得到有效算法。未达到 10 个时记录实际数量和停止原因，不用失败候选填充树，
也不进入后续搜索。

## 4. 树、边与轨迹

虚拟根不包含算法记录，保存根节点集合和根访问数。原始正式 V8.3 中，初始化后虚拟根
只在已有根节点中执行递归选择，不直接生成新的根方案。

每个有效算法是一个 `TreeNode`，保存算法记录、唯一 `parent_id`、按创建顺序的
`child_ids`、`visit_count` 和 `expansion_attempts`。每个有效子节点有且只有一条
`TreeEdge`：

```text
TreeEdge = {
    parent_id,
    child_id,
    operator,
    reference_node_id,
    parent_quality,
    child_quality,
}
```

边上的 parent/child quality 是该次扩展创建时冻结的秩快照，用于直接回报；参考节点只
提供生成信息，不构成第二结构父代。任意节点 $n$ 的来时路为：

$$
\tau(n)=(n_0,e_1,n_1,\ldots,e_k,n_k=n).
$$

完整树永久保留所有评价有效节点。LLM 只接收有限局部脉络，不直接读取完整树。

## 5. MCTS 选择和信用

最小化任务先取负，最大化任务保留原值，得到有向 fitness $y(n)$。节点的子树最好值为：

$$
G(n)=\max\left(y(n),\max_{c\in Children(n)}G(c)\right).
$$

当前全树有效节点的有向 fitness 转为并列感知的中秩百分位 $Z(x)\in[0,1]$；只有一个值
或全部同分时取 0.5。节点自身成果和子树成果分别为：

$$
F(n)=Z(y(n)),\qquad B(n)=Z(G(n)).
$$

总 evaluator 预算为 $T$、已启动 evaluator 数为 $t$ 时，剩余预算比例为：

$$
r_t=\operatorname{clip}\left(\frac{T-t}{T},0,1\right).
$$

在父节点 $n$ 中，进入已有子节点 $c$ 的分数为：

$$
S_{down}(c\mid n)=B(c)+0.1r_t
\sqrt{\frac{\log(1+N(n))}{N(c)}}.
$$

原始正式 V8.3 还用最近路线是否持续刷新前沿，构造从当前节点再开直接分支的弱先验。
对路径 $\tau(n)=(v_0,\ldots,v_k=n)$，令 $z_i=F(v_i)$、
$b_i=\max_{0\le j\le i}z_j$。在最近 $h=8$ 条边窗口中，前沿推进量和更新频率为：

$$
P_h(n)=b_k-b_s,
\qquad
C_h(n)=\frac{1}{k-s}\sum_{i=s+1}^{k}\mathbf 1[b_i>b_{i-1}],
$$

其中 $s=\max(0,k-h)$；没有边时两项取 0。路线推进信号和扩展先验为：

$$
M_h(n)=\sqrt{P_h(n)C_h(n)},
\qquad
\pi(n)=\min(1,F(n)+0.1M_h(n)).
$$

节点 $n$ 直接产生子节点 $c$ 时，按创建时冻结的 $F(n)$ 和 $F(c)$ 定义：

$$
D(n,c)=[F(c)-F(n)]_+,
$$

$$
R(n,c)=0.75F(c)+0.25D(n,c).
$$

若一次扩展在上下文、生成、解析或 evaluator 阶段失败，回报为 0。设 $A(n)$ 为节点已经
获得的直接扩展尝试数，则 `new_child` 的质量为：

$$
Q_{new}(n)=\frac{\pi(n)+\sum_{c\in Children(n)}R(n,c)}{1+A(n)}.
$$

`new_child` 的最终分数为：

$$
S_{new}(n)=Q_{new}(n)+0.1r_t
\sqrt{\frac{\log(1+N(n))}{1+A(n)}}.
$$

后代结果可以通过 $G$ 抬高已有子节点的 `descend` 价值，但不会追溯改写产生该子节点时
冻结的直接扩展回报。global best 只由原始 fitness 决定，不受上述搜索分数影响。

## 6. 递归选择与宽度下限

每轮从虚拟根开始。根只在已有 10 个根节点中按 `S_down` 选择，不生成新的根方案。到达
程序节点 $n$ 后，同时考虑 `new_child(n)` 和全部 `descend(c)`：

1. 若 $|Children(n)|<\max(1,\lfloor N(n)^{0.5}\rfloor)$，优先执行 `new_child`，保证
   节点访问增加后逐步获得基本宽度；
2. 达到该宽度后，`S_new(n)` 与全部 `S_down(c\mid n)` 直接竞争；
3. `new_child` 获胜时停止下降并生成一个候选；已有子节点获胜时进入该节点继续递归；
4. 最高分完全相同时使用 seeded RNG 打破并列；
5. 只有深度小于 `max_depth=10` 的节点可以继续作为 `descend` 目标。

一次搜索扩展最多生成一个候选，因此 $|Children(n)|\le A(n)$。失败尝试增加选择路径访问
和父节点的 $A(n)$，以零回报进入 $Q_{new}$，但不增加节点、边或 $G$。

这里的平方根条件只是 `new_child` 的宽度下限。达到宽度后，`new_child` 仍可因自身价值
继续胜出，并不存在严格的宽度上界。正式工件中出现的单节点 sibling 爆炸正是这一机制的
实际后果。V8.3-PW 后续尝试改为严格 progressive widening 并扩展到虚拟根，但该批次未完成，
不属于本文正式协议。

## 7. 局部探索脉络

当前节点始终完整展示 `code`、`description` 和原始 fitness。除此之外，prompt 默认提取：

1. 最近 3 条形成边；
2. 当前节点最多 3 个代表性直接子分支。

形成边和分支统一使用：

```text
思想：<design_idea>
形成的算法：<description>
结果：<parent fitness> -> <child fitness>，<改进 / 持平 / 退步>
```

代表分支优先展示子树最好分支、最近创建的未改进分支和最近的其他分支，重复命中时只
展示一次。若直接子节点本身退步但其后代取得更好子树结果，额外记录后续最好 fitness。
这条记录说明该方向后来实际发展过，不把后代结果解释为直接修改的因果证明。

上下文不包含 UCT、访问次数、节点排名、global best 或其他搜索调度数值；这些量控制
预算，不是单节点生成所必需的设计信息。上下文超限时保留任务契约、当前完整代码、描述、
fitness 和 `trace_crossover` 的参考算法，删除较旧轨迹单元和低优先级分支。若必需内容仍
无法容纳，本轮按失败尝试记账。

## 8. 五个改进算子与参考节点

搜索阶段使用五个等概率算子：

| 算子 | 主要问题 | 额外参考 |
| --- | --- | --- |
| `trace_refine` | 当前机制的哪个薄弱点值得继续发展或修复 | 无 |
| `trace_tune` | 哪些参数、阈值、尺度或触发条件限制了当前机制 | 无 |
| `trace_simplify` | 哪些机制或代码可以删除、合并或简化 | 无 |
| `trace_innovate` | 哪个明显不同的核心思路值得另开路线 | 无 |
| `trace_crossover` | 另一算法节点的哪个互补思想适合选择性融入 | 一个参考节点 |

前四个算子只读取当前程序与局部脉络。`trace_crossover` 从除当前节点外的全树有效节点
中选择一个参考，不按根分支、亲缘关系、树距离或历史引用次数过滤。参考概率为标准化
有向 fitness 的 softmax，温度 $\tau_{ref}=1.0$，候选全同分时均匀抽样。参考节点的完整
代码、description 和 fitness 加入 prompt，但新候选仍只连接当前节点。

算子表示生成前的主要改进意图，不是静态代码类型检查器。候选可以包含实现主意图所需的
配套修改；是否有效由完整代码、描述和 evaluator 共同决定。

## 9. 单次扩展和停止

```text
1. 从虚拟根按 `S_down`、宽度下限和 `S_new`/`S_down` 竞争选择当前节点。
2. 记录选择路径访问和一次扩展尝试。
3. 等概率选择可用算子；若为 crossover，按 fitness-softmax 选择一个参考节点。
4. 提取局部探索脉络并调用 Call 1 生成 design_idea + code。
5. 调用 Call 2 生成 description；任一生成阶段失败则不评价、不入树。
6. 评价代码；evaluator 已启动即计预算。
7. 成功则写入一个节点和一条边，更新冻结直接回报、子树价值和 global best。
8. 失败则保留访问、扩展尝试和零回报，重新从根选择。
```

搜索在 evaluator 预算耗尽或连续 20 次未产生有效算法时停止。候选级扩展串行执行；
一个节点可多次扩展并拥有多个子节点，但每个有效子节点只有一个结构父代。

## 10. 工件、配置与实验边界

节点、树、边和信用保存在 checkpoint；每次扩展保留选择路径、算子、参考节点、局部
上下文、两次 LLM 调用、评估结果、入树结果和 global-best 更新。运行预算为 1000 evaluator，
初始化有效根节点为 10，`max_depth=10`，局部历史为最近 3 条边和最多 3 个直接分支。
选择使用 `lambda_0=0.1`、宽度指数 `alpha=0.5`、扩展先验权重 `beta=1`、直接增益
权重 `rho=0.25`、路线推进权重 `kappa=0.1` 和窗口 `h=8`；crossover 参考 softmax
温度为 1.0。

原始正式批次完成四任务各三次 1000 evaluator 搜索和完整 held-out。其 15 个 held-out
规模平均名次为 7.667；结果页显示 V8.3 在 TSP、CVRP 和 OP 整体弱于 V8/V8.2，OBP
相对更有竞争力。该结果描述的是原始正式 V8.3，不适用于后续未完成的 PW 或 credit
修正版，也不支持把局部脉络、五算子或某个信用权重单独解释为性能原因。

## 11. 研究边界

V8.3 正式版同时引入树结构、局部上下文、五算子、两次 LLM 调用、扩展信用和递归选择，
联合结果不能识别单个机制的因果贡献。后续 credit 修正版删除固定深度并改变路线信用，
因此必须使用独立协议、独立训练完成状态和独立 held-out 后再评价，不能覆盖本文的正式
结果或复用其协议身份。

实现真相源为 Git 提交 `04aaef9` 的 `llm4ad/method/traceaad_v8_3/`；正式结果入口为
`experiments/<task>/traceaad_v8_3/eval_best_v83_20260806/`。
