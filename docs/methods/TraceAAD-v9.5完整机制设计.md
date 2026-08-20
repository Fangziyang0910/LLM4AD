# TraceAAD V9.5：锚点证据与质量—机会分配

## 1. 研究问题

在 evaluator 驱动的算法进化中，下一步生成需要两类决定：模型应看到哪些与当前程序相关的历史，以及有限生成机会应投向哪个算法状态。V9.5 将这两个问题分开处理：历史只改变生成上下文，分配器只决定下一次从哪里生成。

## 2. 搜索状态

搜索从 8 个初始根程序开始。一个 `ProgramArtifact` 是 evaluator 实际执行过的一份唯一代码；一个 `AnchorState` 是程序沿一条具体形成路径到达的状态，拥有独立的形成历史与机会计数 $n(a)$；`AttemptRecord` 记录从锚点发起的一次生成及其结果。

同一程序沿不同路径形成时，可以对应多个锚点。它们共享代码和 fitness，不共享形成历史、direct attempts 或机会计数。任务目标统一为越大越好的有向质量 $q$。

## 3. 锚点局部证据

给定锚点 $a$，提示中的局部证据由两部分组成：

$$
E_t(a)=E_{direct}(a)+E_{\mathrm{recent\ formation}}(a).
$$

`Direct attempts` 是从当前锚点发起的已完成尝试，包括 improve、plateau、regress、invalid、no-op、重复和祖先返回。来自其他锚点的尝试不混入。

`Recent formation` 是沿当前锚点父链回溯的最近形成事件。每条事件按真实顺序呈现：

````text
Idea: <declared idea or unavailable>
Change: <deterministic parent-child diff excerpt>
Result: improve | plateau | regress
Fitness: parent -> child
````

direct 证据先按结果类别去重并取最近事件，再由 formation 填充剩余位置。当前代码、任务契约和输出要求始终保留；上下文超限时压缩 diff 细节，再删除较早 formation。

## 4. 证据条件生成

单个候选的条件输入为：

$$
(Task, Code(a), E_t(a)).
$$

模型输出一个可选 Idea 和一份完整可执行代码。Idea 是生成时的声明，实际修改由系统比较父子代码得到。每次选中锚点只生成一个候选，处理完成后立即重新选择，不预先承诺连续多步生成。

V9.5 不设置 Refine/Explore 算子组合；生成任务由共同的候选协议给出。

## 5. 质量—机会分配

固定尺度 $s\ge0$ 下，锚点分数为：

$$
S_t(a)=q(a)+\frac{s}{\sqrt{n_t(a)+1}}.
$$

每轮选择分数最高的锚点。完全同分时优先访问次数更少、创建更早的状态。新锚点的初始乐观项为 $s$。

该分数让当前质量与访问不足共同决定下一份生成机会。它不计算趋势、成熟度、平均增益或祖先信用；成功子代以自己的质量成为新锚点，不把正增益复制回祖先。

## 6. 初始化与尺度

初始化持续生成，直到获得 8 个唯一有效根。随后每个根接受一次 bootstrap。对形成有效子状态的 bootstrap 转换，计算：

$$
d_i=|q(child_i)-q(root_i)|.
$$

改善、退步和持平都进入 $D_{init}$，固定尺度为：

$$
s=\begin{cases}
\operatorname{median}(D_{init}), & D_{init}\neq\varnothing,\\
0, & D_{init}=\varnothing.
\end{cases}
$$

正式搜索开始后不再重估 $s$。

## 7. 预算与最终程序

V9.5 的预算单位是完成的候选响应。解析失败、运行错误、评价超时、no-op、重复和祖先返回都消耗一次候选预算，并使起点锚点的 $n(a)$ 加一；模型传输失败不计预算。

无效和重复事实保留，但不创建新锚点。其他路线已有相同程序时可以复用确定性 fitness，并在当前历史下创建新的锚点。

正常停止于候选响应预算耗尽。最终在唯一程序中按任务真实目标选择，完全同分时偏好代码更短、发现更早的程序。

````text
Generate 8 unique valid root programs.
Generate one bootstrap candidate from each root.
Set s to the median absolute bootstrap quality change.

While completed-response budget remains:
    Score every anchor by q(a) + s / sqrt(n(a) + 1).
    Select the highest-scoring anchor.
    Build local direct and formation evidence.
    Generate one optional Idea and one complete program.
    Count the response and increment n(anchor).
    Classify the result; evaluate or reuse deterministic fitness.
    Record the attempt and create a child state when allowed.
    Reselect globally.

Return the best unique program by the true objective.
````
