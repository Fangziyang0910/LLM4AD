# TraceAAD V8：以树搜索调度算法改进历史

> 状态：机制实现与脚本化验证已完成，真实模型冒烟和正式实验尚未进行。
>
> 版本日期：2026-08-04。计划协议标识为 `traceaad-v8`，首次实现使用独立
> checkpoint schema，不兼容 V5/V7 checkpoint。V5、V7 继续由各自文档和实现维护。

## 1. 科学问题与核心假设

算法改进会逐步引入思想、观察结果，再继续细化、回退、修复、重组或换向。
一个暂时较差的中间程序可能包含仍需多步开发的思想；当前 fitness 只能说明它在
当前状态下的结果，不能完整说明从它继续发展能够达到什么位置。

TraceAAD V5 把“当前程序及其来时路”作为搜索个体，并用固定规模的 active
轨迹种群分配繁衍预算。派生图保存了退出 active 的历史事实，但 archived 轨迹
不再直接繁衍，也不再作为参考。因此，种群生存决策会把“当前是否保留”与
“未来是否仍值得发展”绑定在一起。

V8 的核心设计假设是：

> 用一棵完整的单父代搜索树保存所有评价有效的程序和改进关系；把根到当前节点
> 的路径作为下一步生成所需的算法改进历史；通过子树信用、递归 UCT 和渐进扩展
> 分配预算，使暂时较差的节点仍保有以后被继续开发的机会。

该假设包含三个可分离的主张：

1. **历史主张**：与当前程序严格匹配的形成路径和已测试分支，能够改善下一步
   Action 的判断。
2. **结构主张**：完整树取消固定种群的硬淘汰后，能够保留更多“先退步、后突破”
   的路线。
3. **调度主张**：子树最佳结果回传、UCT 和渐进扩展能够在完整保留与有限预算之间
   形成有效取舍。

这些主张均是待验证假设。V8 的实现正确性、过程现象和最终 held-out 结果需要分别
报告；任何单次最好分数都不能证明树结构或信用回传普遍有效。

## 2. V5 基线、MCTS-AHD 借鉴与 V8 边界

V8 以 V5 的生成机制为基线，替换搜索状态和预算调度，不把 MCTS-AHD 整体复制为
TraceAAD。

### 2.1 从 V5 保留的机制

- 程序节点、单结构父代修改边和可追溯 provenance；
- `trace_ideate`、`trace_refine`、`trace_synthesize`、`trace_transfer` 四个轨迹语义算子；
- Action 与 Code 两阶段协议；
- Action 与 Code 使用同一主轨迹历史和可选参考轨迹历史；
- 双轨迹算子只把参考程序作为知识来源，不形成第二结构父代；
- 真实 evaluator 决定程序 fitness；
- fitness 优先，完全同分时按非空 LOC 择短；
- 不维护在线全局经验或自由文本全局反思。

### 2.2 从 MCTS-AHD 借鉴的机制

- 虚拟根节点和完整程序派生树；
- 从根开始递归选择，而不是在固定 active 集合中一次性抽取父代；
- 节点访问数与 UCT；
- 子树结果沿祖先路径回传；
- progressive widening 控制开放动作空间中的分支增长；
- 探索压力随剩余预算下降。

### 2.3 V8 不采用的 MCTS-AHD 组件

- 不采用 `i1/e1/e2/m1/m2/s1` 算子集合，继续使用 TraceAAD 四个语义算子；
- 不维护用于 e2 的 top-k elite population；
- 不使用额外 thought-alignment LLM 调用；
- 不把 max-backup 解释成某个祖先思想的因果贡献；
- 不使用固定 population 作为参考选择或最终输出接口；
- 不在 V8 初版同时引入算子 reward、Elo、全局经验或多维价值模型。

因此，V8 的方法定位是“树调度的轨迹引导搜索”。树定义完整搜索拓扑，轨迹定义
某次生成所读取的历史条件，四个算子定义下一步怎样利用这些历史。

## 3. 搜索树、节点与修改边

### 3.1 虚拟根节点

搜索树根节点 $n_r$ 是不包含代码和 fitness 的虚拟节点。所有初始程序都是
$n_r$ 的直接子节点。根节点只负责组织不同初始算法家族和进行第一层预算分配，
初始化完成后不再直接生成新的根子节点。

冻结根扩展有两个原因：第一，V8 与 V5 使用相同数量和生成方式的初始程序，便于
控制比较；第二，`trace_ideate` 已能从现有节点提出新方向，无需额外引入 MCTS-AHD
的根级 e1 交叉机制。

### 3.2 程序节点

每个非根节点 $n$ 表示一次评价有效的程序状态，至少保存：

- `id`、完整代码、简短 Implemented Idea；
- evaluator fitness、非空 LOC、`code_hash`；
- 唯一结构父节点和入边；
- 有序子节点集合和树深度；
- 访问数 $N(n)$；
- 节点自身有向 fitness $f(n)$；
- 子树最佳值 $G(n)$ 及对应的最佳后代节点；
- 创建批次、样本顺序和生成算子。

最小化任务先将 evaluator fitness 取反，统一为越大越好的有向 fitness。原始
fitness 始终保留，报告时仍使用任务原始方向。

### 3.3 修改边

父节点 $p$ 到子节点 $c$ 的边保存：

- Requested Action 与 Implemented Idea；
- 四类算子之一；
- 主节点、参考节点和参考根分支；
- 相对父节点和批前 global best 的有向 fitness 变化；
- `improve / plateau / regress`；
- LOC 变化、代码变化比例和代码哈希；
- 是否产生新的 global best；
- 迭代、批次和 sibling 顺序。

每个评价有效的子节点恰好有一条结构入边。双轨迹参考只写入边的 provenance，
不增加结构入边。由结构边组成的搜索状态始终是一棵树；参考关系可以在离线分析中
视为额外知识边。

## 4. 树中的轨迹语义

对任意非根节点 $n$，从其所属根子节点到 $n$ 的唯一路径定义当前改进轨迹：

\[
\tau(n)=(n_0,e_1,n_1,\ldots,e_k,n_k=n).
\]

树完整保存路径，不再把轨迹截断为搜索状态。发送给 LLM 的历史仍受上下文限制，
默认最多展示最近 8 条祖先修改边。完整路径继续保留在 checkpoint 和 artifacts 中。

轨迹在 V8 中承担三项职责：

- 说明当前程序怎样形成；
- 说明沿当前路径哪些思想产生了改进、停滞或退步；
- 为 Action 提供与当前代码严格匹配的决策证据。

轨迹不再拥有 active/archive 状态，也不参与种群生存。任何树节点都可以通过根到
该节点的递归选择重新获得扩展机会。

## 5. 初始化

V8 默认在总 evaluator 预算内建立 30 个评价有效的初始程序，并把它们连接到虚拟
根节点。初始化 prompt、程序解析、执行契约和多样性提示与 V5 保持一致。

初始化规则如下：

1. LLM 生成一个完整程序和一句 Implemented Idea；
2. 代码解析成功后进入真实 evaluator；
3. evaluator 返回有限数值 fitness 时建立根子节点；
4. evaluator 失败会消耗 evaluator 预算，但不建立树节点；
5. LLM transport 或代码解析失败不消耗 evaluator 预算；
6. 达到 30 个有效初始节点、预算耗尽或连续生成停滞时结束初始化；
7. 只要至少有一个有效根子节点，初始化不足 30 个时仍可进入搜索，但必须在
   summary 中记录实际初始节点数。

每个初始节点设置 $N(n)=1$、$G(n)=f(n)$。根节点访问数初始化为有效根子节点数，
根的 $G(n_r)$ 为全部根子树中的最佳值。

## 6. 子树信用与回传

### 6.1 在线搜索信用

节点自身结果与其子树发展结果分开保存：

\[
f(n)=\text{directed fitness of }n,
\]

\[
G(n)=\max\left(f(n),\max_{c\in Children(n)}G(c)\right).
\]

若 fitness 完全相同，则按非空 LOC 更少的程序确定 `subtree_best_node_id`。LOC 只
决定完全同分时的最佳后代，不作为连续数值加入 $G$。

每当新子节点进入树后，从其父节点开始沿唯一祖先路径重新计算 $G$，直到根节点。
这项 max-backup 表示：

> 从该节点所属子树已经实际到达过的最好程序质量。

它是乐观的“子树发展潜力”代理，用于有限预算下的搜索调度。它不证明祖先程序、
某条 Action 或某个算法思想因果上产生了最佳后代。

### 6.2 访问数

$N(n)$ 的单位是“一个 expansion batch 的选择路径经过该节点”。每轮一旦选定扩展
节点并开始 Action 生成，根到该节点路径上的 $N$ 均增加 1。即使 Action/Code
解析失败或 evaluator 全部失败，该路径仍获得过一次生成机会，因此访问仍计数。

每个新子节点以 $N=1$ 加入。一个 batch 最多生成两个 sibling，但祖先访问只增加
一次，从而保持与 V5 的“主路线获得一次生成批次”口径一致，避免 sibling 数量
机械放大父节点访问。

### 6.3 离线信用信号

V8 同时记录但不在首版在线调度中混合以下信号：

- 子节点相对父节点的即时变化；
- 子树最佳值相对当前节点的长期增益；
- 首次产生 route/global breakthrough 的后代深度；
- 一条边后来是否位于 global-best 祖先路径；
- 节点获得的访问、有效子代数和失败率；
- 算子、Action 和参考节点的后续贡献。

这些信号用于研究真正的多步信用分配。V8 初版只用 $G/N$ 调度，避免在树结构、
趋势信用、算子信用和思想信用之间同时引入多个未经验证的耦合项。

## 7. 递归 UCT 选择

### 7.1 子树值归一化

设当前全部有效程序节点的有向 fitness 下界和上界为 $g_{min}$、$g_{max}$。子树值
归一化为：

\[
Z(n)=
\begin{cases}
\dfrac{G(n)-g_{min}}{g_{max}-g_{min}}, & g_{max}-g_{min}>10^{-12},\\
0.5, & \text{otherwise}.
\end{cases}
\]

归一化只服务于 UCT 的跨任务尺度稳定性，不改变原始 fitness 或最终排序。

### 7.2 剩余预算衰减

总 evaluator 预算为 $T$，已经尝试评价 $t$ 个程序，剩余预算比例为：

\[
r_t=\operatorname{clip}\left(\frac{T-t}{T},0,1\right).
\]

无限预算配置下定义 $r_t=1$。

父节点 $p$ 的子节点 $c$ 的 UCT 为：

\[
UCT(c)=Z(c)+\lambda_0r_t
\sqrt{\frac{\log(1+N(p))}{N(c)}}.
\]

V8 初版计划使用 $\lambda_0=0.5$。该值高于 MCTS-AHD 原始实现中的 0.1，因为
V8 默认同时维护 30 个根分支，且归一化 exploitation 跨度可达 1；更高的初始
探索系数用于使低初始质量根分支仍有实际被开发的机会。该参数是待验证设置，必须
报告根分支访问覆盖并进行敏感性检查。

### 7.3 选择过程

每轮从虚拟根开始：

1. 根节点固定不再扩展，从根子节点中选择 UCT 最大者；
2. 到达非根节点后，先判断该节点是否满足渐进扩展条件；
3. 满足条件时停止下降并扩展该节点；
4. 不满足条件时，从其子节点中选择 UCT 最大者并继续下降；
5. UCT 完全相同时使用本次运行的 seeded RNG 随机打破并列。

选择是递归的，因此高价值子树可以获得集中预算；访问较少的兄弟节点通过探索项
保留以后被重访的机会。剩余预算下降后，选择逐渐由已观察到的子树结果主导。

## 8. Progressive Widening

LLM 的修改动作空间无法枚举。若每次访问节点都增加新子节点，搜索会过早变宽；
若只允许第一次扩展，后续又无法从有潜力的内部节点尝试新方向。V8 用渐进扩展
控制两者。

非根节点 $n$ 在当前访问数下允许的最大子节点数为：

\[
W(n)=\max\left(A,\left\lfloor N(n)^\alpha\right\rfloor\right),
\]

其中默认 sibling Action 数 $A=2$，$\alpha=0.5$。当
$|Children(n)|<W(n)$ 时，该节点可以再次扩展。

一次扩展实际请求的 Action 数为：

\[
A_n=\min\left(A,W(n)-|Children(n)|\right).
\]

因此，新节点第一次被扩展时最多产生两个 sibling；只有在访问数继续增长后，才
逐步开放第三个、第四个及更多分支。评价失败不会占用 child slot，因为它没有形成
可执行搜索状态；对应失败仍记录在 artifacts 中。

V8 不设置默认树深上限。完整路径长度由 evaluator 预算、UCT 和渐进扩展共同限制，
LLM 上下文则独立保持有界。若真实实验出现无收益深链，再把最大深度作为单独消融，
不在首版预先截断可能需要多步发展的路线。

## 9. 四个轨迹语义算子在树中的适配

V8 保留 V5 的四个算子，并在当前可用算子中等概率选择：

| 算子 | 树中的主节点语义 | 参考需求 |
| --- | --- | --- |
| `trace_ideate` | 根据祖先路径和已有子分支提出尚未尝试的新方向 | 无 |
| `trace_refine` | 对路径中已有价值或已暴露弱点的一个机制做聚焦修正 | 无 |
| `trace_synthesize` | 让当前根分支与另一根分支的两个有支持原则产生功能交互 | 需要其他根分支 |
| `trace_transfer` | 保持当前程序核心结构，只迁移另一根分支的一个有支持思想 | 需要其他根分支 |

V8 不单独设置 MCTS-AHD 的 path-reasoning 算子。所有四个算子的 Action prompt 都
接收当前节点的形成路径，树路径推理是统一生成条件，而不是只有某一个算子才能
使用的能力。

同一节点以后再次满足 progressive widening 时，可以用新的算子继续开放分支。
首版不维护节点级或全局算子 reward，防止树信用与算子信用同时改变预算分配。

## 10. 树分支级参考选择

V8 不维护 active population 或 elite set。双轨迹参考直接从搜索树的其他根分支
选择。

定义 `root_branch(n)` 为从虚拟根到节点 $n$ 路径上的第一个程序节点。当前主节点
$p$ 的参考候选按以下过程建立：

1. 排除 `root_branch(p)`；
2. 对每个其他根分支，取其 `subtree_best_node_id` 作为该分支代表；
3. 排除代表程序与主程序 `code_hash` 完全相同的分支；
4. 使用各根分支归一化后的子树值 $Z$ 做 softmax，默认温度 0.2；
5. 先抽取一个参考根分支，再使用该分支的 subtree-best 程序和形成路径。

这种选择保留了 V5“参考一条有具体历史的优质路线”的语义，同时避免建立需要
生存管理的隐藏参考种群。参考节点必须位于不同初始家族，确保它提供当前主路径
之外的知识。

参考节点不增加 $N$，不改变其 $G$，也不参与主节点的结构回传。若没有其他有效
根分支，双轨迹算子不可用，本轮只在 `trace_ideate/trace_refine` 中等概率选择。

## 11. 生成上下文

### 11.1 当前节点的形成历史

Action 与 Code 都接收 `[How This Program Was Reached]`，默认展示当前节点最近
8 条祖先修改边。每条边至少包含：

- Requested Action；
- Implemented Idea；
- 父子 fitness 和 `improve / plateau / regress`；
- 是否产生 global breakthrough；
- 必要的 LOC 与代码变化事实。

历史边不重复完整程序代码，当前主节点始终完整展示。

### 11.2 已从当前节点测试的分支

树搜索会重新扩展内部节点。只展示祖先路径会让模型不知道该节点此前已经尝试过
哪些直接修改，因此 V8 增加 `[Previously Tested From This Program]`：

- 最多展示 8 个直接子分支；
- 优先选择子树值最高的 4 个分支；
- 其余席位选择最近创建且尚未包含的分支；
- 最终按创建顺序展示，避免排名顺序被误读为时间顺序。

每个分支摘要包含 Requested Action、直接子节点结果、结果分类、该分支当前达到的
subtree-best fitness 及达到最佳后代的深度。它告诉模型哪些方向已经立即失败、
哪些方向经过多步后产生了价值，以及本轮需要避免怎样的重复。

分支摘要只提供真实可追溯事实。`G` 表示实际到达过的最好结果，prompt 不把它
表述为已经证明的思想因果贡献。

### 11.3 参考上下文

双轨迹算子另外接收：

- 参考 subtree-best 程序的完整代码；
- 参考根分支到该程序最近 8 条形成边；
- 参考程序和当前程序各自的 fitness。

Action 必须说明从参考分支学到什么，以及如何按当前算子语义迁移或交互。Code
只能按选定 Action 使用参考程序。

### 11.4 上下文边界

运行必须显式提供模型 input context limit。构造上下文时依次保证：当前程序与任务
契约、当前祖先路径、当前节点已测试分支、参考程序、参考路径。若双轨迹上下文
超限，先缩短参考路径和分支摘要；仍超限时退化为单轨迹算子，不得发送超过模型
上限的 prompt。

## 12. Action、Code 与评价协议

Action 阶段根据任务、当前程序、形成路径、已有子分支、算子约束和可选参考分支，
输出 $A_n$ 条编号、单行、自包含的自然语言修改建议。每条 Action 必须：

1. 说明一个具体可执行修改；
2. 说明它与当前节点已经测试的相关分支有何实质差异；
3. 只依赖目标函数参数和局部计算；
4. 双轨迹时明确参考知识的来源和适配方式；
5. 产生实质算法行为变化，不复述当前程序。

Code 阶段使用与 Action 相同的主历史、主程序和可选参考历史、参考程序，生成一个
完整实现。程序 parser 保留任务模板 imports/preface，允许合法的小型顶层 helper，
并要求目标函数契约不变。

每条 Action 独立生成和评价一个 sibling。只有代码解析成功且 evaluator 返回有限
数值 fitness 的候选进入搜索树。一个 batch 中 sibling 共享批前 global best 和
选择上下文，避免生成顺序改变 delta 和 prompt。

## 13. Global Best、复杂度与重复程序

global best 在全部有效树节点中选择：

1. 先比较任务 fitness；
2. fitness 完全相同时选择非空 LOC 更少的程序；
3. fitness 与 LOC 都相同时使用确定性 node id 打破并列。

复杂度不形成连续惩罚，不影响 UCT 数值，也不拒绝严格更优的长程序。

V8 初版不做在线代码去重或语义去重。相同代码在不同路径下仍是不同的历史条件，
直接合并会同时改变“树结构”和“历史是否有用”两个研究变量。每次评价继续遵循
V5 的预算语义，不使用 code-hash fitness cache；`code_hash` 只用于记录重复率和
后续消融。

若重复 occurrence 大量获得独立 UCT 奖励并浪费预算，后续可比较：

- 完全不去重；
- 共享相同代码的访问统计；
- 保留全部事实边、只允许一个 canonical occurrence 扩展。

该问题不与 V8 首版树调度同时改动。

## 14. 失败、停滞与预算

- LLM transport 失败：记录错误，不消耗 evaluator 预算，不建立节点；
- Action/Code parse 失败：记录失败，不消耗 evaluator 预算，不建立节点；
- evaluator runtime/timeout/invalid result：消耗 evaluator 预算，记录结构化失败，
  不建立节点；
- NaN/Inf：按 invalid result 处理；
- expansion batch 无有效子节点：访问仍回传，节点子树值不变；
- 连续若干 batch 没有增加有效树节点：按停滞上限安全停止；
- 根没有有效子节点：返回空结果并明确记录初始化失败；
- 预算耗尽：完成当前已开始的单个 evaluator 调用后停止，不超出配置预算。

V8 的正式 evaluator 预算默认仍为 1000，并包含初始化评价。除了 evaluator 数，
实验必须同时报告 Action/Code LLM 调用数、输入/输出 token 和运行时间。

## 15. 完整搜索流程

```text
1. 创建虚拟根节点。
2. 在总评价预算内生成 30 个评价有效的初始程序：
   - 每个程序成为一个根子节点；
   - 初始化 N、f、G 和 global best。
3. 重复直到预算耗尽或安全停止：
   a. 计算剩余预算比例和全局 fitness 归一化边界；
   b. 从根开始递归计算 UCT；
   c. 在满足 progressive widening 的非根节点停止下降；
   d. 沿选中路径增加一次 batch visit；
   e. 选择当前可用的轨迹语义算子；
   f. 双轨迹时，从其他根分支按 Z-softmax 选择参考 subtree best；
   g. 构造祖先路径、已有直接分支和可选参考路径上下文；
   h. 按剩余 child slots 生成 1–2 条 Actions；
   i. 对每条 Action 生成完整程序并真实评价；
   j. 每个有效 sibling 从当前节点写入唯一结构边和新子节点；
   k. 沿祖先路径 max-backup 子树最佳值 G；
   l. 在批前快照上确定本 batch 的 global-best winner；
   m. 写入原始工件并按周期保存 checkpoint。
4. 返回全树 fitness 最优、完全同分时 LOC 最少的程序。
```

流程中没有 active frontier、archive、生存收缩、compact-best 锚点、全局经验或
反思 LLM 调用。

## 16. Checkpoint、工件与可恢复性

### 16.1 Checkpoint

V8 checkpoint 至少保存：

- 虚拟根、全部程序节点、结构边和父子顺序；
- 每个节点的 $N/f/G$、subtree-best node id 和创建信息；
- global best、fitness 归一化边界和样本顺序；
- evaluator 预算、下一批次 id、连续停滞次数；
- 完整搜索配置、任务/模型非密钥身份和 RNG 状态。

恢复时验证：树连通且无环、每个非根节点恰好一个父代、父子深度一致、fitness
有限、LOC/code hash 可重算、$G$ 与 subtree-best pointer 一致、global best 正确、
访问数均不小于 1。旧版本 checkpoint 不自动迁移。

### 16.2 原始工件

沿用 TraceAAD 的监控、分析和恢复三分开结构：

```text
<run>/
  run_config.json
  logs/progress.log, errors.jsonl, summary.json
  artifacts/candidates.jsonl, edges.jsonl, llm_calls.jsonl, decisions.jsonl
  checkpoints/latest.json
```

`decisions.jsonl` 额外记录：

- 每轮完整 selected path；
- 每层子节点的 $Z/G/N/UCT$；
- progressive-widening capacity 和实际 child slots；
- 算子、参考根分支和参考节点；
- batch visit；
- 每次 backup 前后的 $G$ 与 subtree-best node id。

落盘只保存原始事实和在线决策，不在运行时写入 LRR、PCD、长期贡献或因果结论。
这些指标由离线分析脚本从 candidates、edges 和 decisions 重建。

## 17. 实现不变量

- 虚拟根不包含代码或 fitness，初始化后不再直接扩展；
- 每个评价有效程序节点只有一个结构父代；
- 全部评价有效节点永久留在搜索树，没有 active/archive 状态；
- 参考节点只提供知识，不形成第二结构父代或访问回传；
- 当前主节点就是本轮唯一结构锚点；
- 根到主节点的路径是本轮主轨迹；
- 重新扩展内部节点时必须展示已测试的直接子分支；
- Action 与 Code 使用同一主历史和可选参考历史；
- $G$ 只由实际评价有效的自身或后代 fitness 决定；
- $N$ 以 expansion batch 计数，生成失败仍算访问；
- progressive widening 只决定是否增加子节点，不删除任何节点；
- complexity 只在 fitness 完全相同时择短；
- 首版不使用在线去重、算子 reward、趋势 $P$、全局经验或共享反思；
- checkpoint 只负责恢复，不改变选择、扩展、评价或回传机制。

## 18. 计划默认配置

| 配置 | V8 计划默认值 |
| --- | ---: |
| evaluator 预算 | 1000 |
| 初始根子节点 | 30 个评价有效程序 |
| 根后续扩展 | 关闭 |
| 每次 expansion 的最大 sibling Actions | 2 |
| progressive widening $\alpha$ | 0.5 |
| 允许子节点数 $W(n)$ | $\max(2,\lfloor N(n)^{0.5}\rfloor)$ |
| 树最大深度 | 无硬上限 |
| 主祖先历史 | 最近 8 条边 |
| 已测试直接分支摘要 | 最多 8 条 |
| UCT 初始探索系数 $\lambda_0$ | 0.5，待敏感性验证 |
| UCT 探索衰减 | 剩余 evaluator 预算比例 |
| UCT 选择 | 最大值，seeded random tie-break |
| 在线回传 | subtree max fitness |
| 四算子 | 参考可用时等概率 |
| 双轨迹参考 | 其他根分支按 $Z$-softmax |
| 参考 softmax 温度 | 0.2 |
| Action / Code 输出上限 | 1024 / 8192 token |
| 复杂度 | fitness 完全同分时非空 LOC 择短 |
| 在线重复过滤 | 无 |
| 在线全局经验 | 无 |
| 计划实验目录 | `traceaad_v8/version8/` |

## 19. 实现状态

V8 已使用独立 `llm4ad/method/traceaad_v8/` 包实现，没有修改 V5/V7 的搜索实现。
首版包含以下模块：

1. 定义虚拟根、树节点、修改边和运行结果 schema；
2. 实现 tree memory、祖先路径、根分支和 subtree-best 查询；
3. 实现 $G$ backup、UCT、递归 selection 和 progressive widening；
4. 复用 V5 的四算子语义、Action/Code parser、evaluator 和复杂度比较器；
5. 扩展 context renderer，加入直接子分支摘要和树分支参考；
6. 接入 TraceAADArtifacts、checkpoint 和 runner version 路由；
7. 已补齐机制测试、入口测试和脚本化全流程冒烟；
8. 真实模型冒烟和受控实验仍待启动，不提前加入额外信用或调度机制。

最低测试边界包括：

- 单父代树与根不变量；
- 祖先路径和根分支识别；
- max-backup 和完全同分择短；
- UCT 归一化、预算衰减与 seeded tie-break；
- progressive widening 的 child slots；
- 内部节点重新扩展与直接分支上下文；
- 双轨迹参考来自其他根分支；
- 参考不增加访问或结构父代；
- 生成/解析/evaluator 失败的预算和访问语义；
- checkpoint 恢复后的下一次选择与不中断运行一致；
- 全流程没有 active/archive 或种群收缩。

## 20. 实验与可证伪条件

### 20.1 主对照

第一组正式实验比较 V5 与 V8，固定：

- LLM、temperature、任务和 evaluator 数据；
- 初始程序生成协议和初始数量；
- 四算子与等概率选择；
- Action/Code prompt 内容中除树所必需分支信息外的共同部分；
- evaluator 预算 1000；
- 三次独立重复和完整 held-out 评估。

同时报告 evaluator 数、LLM 调用数、输入/输出 token 和墙钟时间。V8 保留 V5 的
Action/Code 两阶段，因此应避免 MCTS-AHD thought-alignment 带来的额外调用混入
树结构比较。

### 20.2 核心消融

1. **结构消融**：V5 固定种群与 V8 完整树，其他生成机制相同；
2. **信用消融**：节点自身 fitness、subtree max、subtree mean；
3. **渐进扩展消融**：progressive widening 与每次访问都开放新分支；
4. **探索消融**：固定探索、剩余预算衰减、$\lambda_0$ 敏感性；
5. **历史消融**：匹配树路径、打乱路径、只有当前程序；
6. **分支上下文消融**：有无 `[Previously Tested From This Program]`；
7. **重复状态消融**：独立 occurrence、共享访问、canonical occurrence。

### 20.3 过程指标

- 最终 search fitness 和完整 held-out 指标；
- 三重复均值、标准差和最差 run；
- 严格 global breakthrough 数、首次和后半程 breakthrough；
- 产生突破节点的初始质量百分位；
- “先退步、后突破”路径数量和深度；
- 根分支访问覆盖、访问集中度和未被扩展根分支数；
- 树深度、宽度、每层节点数和 progressive-widening 触发次数；
- 父子局部改进率、路线最好推进和子树最佳更新；
- 四算子有效候选率、参考分支覆盖和重复代码率；
- LLM 调用、token、evaluator 失败和运行时间。

### 20.4 证伪条件

以下结果会削弱或否定 V8 的核心设计假设：

- V8 没有增加低初始质量节点后来产生突破的机会；
- 完整树主要增加重复状态和无效深链，没有改善最终或 held-out 质量；
- max-backup 使预算过早集中到偶然高分子树，并降低重复稳定性；
- progressive widening 使 1000 预算内的大量根分支从未获得一次实际扩展；
- 树路径或直接分支上下文没有改变 Action，或导致更多重复失败；
- V8 的改进只来自更高 LLM 调用/token 成本；
- search fitness 改善但 held-out 泛化下降。

若树结构有效而 max-backup 无效，下一步应修改信用分配，不回退到种群；若完整树
本身没有改善有价值路线的后续开发，则需要重新检查“硬淘汰是当前主要瓶颈”这一
科学假设。

## 21. 研究边界与相关资料

V8 当前已有独立实现和脚本化机制验证。本文中的默认参数、预期行为和实验指标仍不
构成真实搜索结果；真实模型冒烟、三重复搜索和完整 held-out 评估完成前不得更新
正式结果汇总。

相关内部资料：

- [TraceAAD V5 完整机制设计](TraceAAD-v5完整机制设计.md)
- [TraceAAD V7 完整机制设计](TraceAAD-v7完整机制设计.md)
- [轨迹上下文与搜索评分](../research/RQ-003-轨迹上下文与搜索评分.md)
- [TraceAAD 版本机制分析](../research/TraceAAD版本机制诊断.md)
- [AAD 搜索机制：种群、树、轨迹与图](../references/AAD搜索机制综合.md)
- [TraceAAD 研究认识](../knowledge/研究认识.md)

外部真相源：

- MCTS-AHD 论文：`../../../papers/MCTS-AHD/icml2025.tex`
- MCTS-AHD 原始实现：`../../../reference_code/MCTS-AHD-master/source/`

论文定义 MCTS-AHD 的方法主张，原始实现定义其真实运行行为；V8 只借鉴树调度、
渐进扩展、探索衰减和子树回传思想，其余机制以本文为准。
