# TraceAAD V8 初版：以完整搜索树保存算法改进历史

> 本文定义 V8 初版协议 `traceaad-v8`，对应 Git 提交 `85e97ff` 和正式训练批次
> `v8_20260804_173300`。V8.1 的 direct-code 变化和 V8.2 的自适应扩展另见
> [V8.2 完整机制设计](TraceAAD-v8.2完整机制设计.md)。V8 初版四任务三重复训练与
> held-out 已完成，正式结果见[跨任务实验总汇](../results/实验总汇.md)。

## 1. 研究问题与方法定位

算法改进通常经过多个中间方案。某个中间方案当前得分较低，并不等于从它继续发展的
路线没有价值。若搜索只保留当前高分程序，可能丢失“先退步、后修正、再突破”的形成
条件。V8 研究的问题是：**完整保存算法改进关系，能否让有限 evaluator 预算更有针对性
地继续开发已经发生的路线？**

V8 的核心假设是：将每个评价有效程序作为树节点，将实际修改作为单父代边，并把根到
当前节点的路径作为下一次生成的历史条件；随后以子树结果回传和 UCT 递归选择，在完整
保留历史与有限预算之间分配搜索机会。

这个假设拆成三个可检验主张：

1. 当前程序的形成路径和已测试直接分支能够改善下一次修改的判断。
2. 完整树比 active population 更少丢失暂时退步、但仍可继续发展的路线。
3. 子树最好值回传、剩余预算衰减和渐进扩展能够避免搜索在固定根或无界横向分支上浪费预算。

这些是设计假设，不是由机制定义推出的性能结论。实验分别检查实现事实、搜索过程和
held-out 结果，不能用单次 best 候选替代三重复测试。

## 2. 与 V5 和 MCTS-AHD 的关系

V8 保留 V5 的四个轨迹语义算子、单结构父代、双轨迹参考、Action/Code 共享历史、真实
evaluator 评价和同分择短。V8 只改变搜索状态和预算分配：固定 active 轨迹种群被完整
程序树替代。

V8 借鉴 MCTS-AHD 的虚拟根、递归选择、UCT、子树 max-backup、progressive widening 和
剩余预算探索衰减。它不采用 MCTS-AHD 的算子集合、top-k elite population、额外
thought-alignment 调用或祖先思想因果归因。参考路线仍只是生成知识，不构成第二个结构父代。

## 3. 搜索树与轨迹

虚拟根不包含代码和 fitness。初始化产生的有效程序都是根的直接子节点；初始化结束后
根不再生成新路线，以保持根数量和初始化协议可控。

每个非根节点至少保存：

- 完整代码、Implemented Idea、fitness、非空 LOC 和 `code_hash`；
- 唯一结构父节点、入边、有序子节点和深度；
- 访问数 `N`、自身有向 fitness `f`、子树最佳值 `G` 及最佳后代节点；
- 生成批次、样本顺序、算子和参考分支 provenance。

父节点到子节点的修改边保存 Requested Action、Implemented Idea、算子、主节点、参考
节点、父子及批前 global-best 的有向变化、`improve/plateau/regress`、LOC、代码变化
比例、哈希、批次和 sibling 顺序。每个有效节点恰好一条结构入边；参考关系只记录在
边上，不改变树结构。

对节点 $n$，从所属根子节点到 $n$ 的唯一路径为：

$$
\tau(n)=(n_0,e_1,n_1,\ldots,e_k,n_k=n).
$$

完整路径写入 checkpoint 和原始工件。发送给模型的历史默认取最近 8 条祖先边，并另外
摘要当前节点已经测试的直接分支。历史边提供真实的修改、结果和代码事实，不把模型自述
的 Idea 当作已经验证的因果解释。

## 4. 初始化

V8 初版在总 evaluator 预算内生成 30 个有效初始程序，连接到虚拟根。初始化程序保留
V5 的任务契约、解析和 evaluator 规则：LLM 生成完整程序与一句 Implemented Idea，解析
成功后执行真实评价；只有有限 fitness 才建树节点。

- evaluator 失败消耗一次 evaluator 预算，但不建节点。
- transport、解析或不完整响应失败不消耗 evaluator 预算。
- 达到 30 个有效根节点、预算耗尽或连续停滞时结束初始化。
- 若至少有一个有效根节点，初始化不足 30 个时仍可搜索，并在 summary 中记录实际数量。
- 每个有效根节点以 `N=1`、`G=f` 加入；根访问数为有效根节点数。

## 5. 子树信用与 UCT

不同任务先把 fitness 统一为越大越好的有向值。节点自身质量为 $f(n)$，子树质量为：

$$
G(n)=\max\left(f(n),\max_{c\in Children(n)}G(c)\right).
$$

fitness 完全相同时，子树最佳节点按非空 LOC 择短。$G$ 表示该子树实际到达过的最好
程序质量，用于预算调度；它不证明祖先程序或某一条 Action 对后代成功具有因果贡献。

为了避免任务量纲直接进入 UCT，V8 初版用当前全树有效节点的 fitness 极差归一化：

$$
Z(n)=
\begin{cases}
\dfrac{G(n)-g_{min}}{g_{max}-g_{min}},&g_{max}-g_{min}>10^{-12},\\
0.5,&\text{otherwise}.
\end{cases}
$$

总预算为 $T$、已启动 evaluator 数为 $t$ 时，剩余预算比例为：

$$
r_t=\operatorname{clip}\left(\frac{T-t}{T},0,1\right).
$$

父节点 $p$ 的子节点 $c$ 的 UCT 为：

$$
UCT(c)=Z(c)+\lambda_0r_t\sqrt{\frac{\log(1+N(p))}{N(c)}}.
$$

V8 初版使用 $\lambda_0=0.5$。每轮从虚拟根开始递归选择：根在已有根子节点中选择；
非根节点若可以扩展则根据渐进扩展规则停止下降，否则选择 UCT 最大的子节点继续向下。
完全同分时用 seeded RNG 打破并列。

## 6. 渐进扩展

LLM 的修改空间不能枚举。V8 用渐进扩展限制每个节点的分支增长。节点 $n$ 的访问数为
$N(n)$ 时，其允许子节点数为：

$$
W(n)=\max\left(2,\left\lfloor N(n)^{0.5}\right\rfloor\right).
$$

当有效直接子节点数小于 $W(n)$ 时，节点获得新扩展机会；否则搜索先在已有子树中按
UCT 下降。一次 expansion 最多生成两个 sibling，实际请求数量不超过剩余 child slots。
评价失败不占 child slot，但会保留访问和失败记录。V8 初版不设置树深上限，树深由预算、
UCT 和 widening 共同决定；模型上下文独立限制为最近历史窗口。

## 7. 四个轨迹算子与参考路线

四个算子在参考可用时等概率选择；没有其他有效根分支时只在单轨迹算子中等概率选择。

| 算子 | 生成意图 | 参考 |
| --- | --- | --- |
| `trace_ideate` | 根据当前路径和已测试分支提出尚未尝试的新方向 | 无 |
| `trace_refine` | 聚焦修复或继续发展当前机制 | 无 |
| `trace_synthesize` | 使当前根分支与另一根分支的两个原则发生功能交互 | 另一根分支 |
| `trace_transfer` | 从另一根分支迁移一个适合当前程序的思想 | 另一根分支 |

参考候选从其他根分支取得。每条候选根分支以其 `subtree_best_node_id` 表示，排除当前
根分支和相同 `code_hash` 的程序，再按归一化子树质量 $Z$、温度 0.2 的 softmax 抽取。
参考程序和参考路径进入 prompt，但不增加参考节点访问，不参与主路线回传。

## 8. Action、Code 与评价

每轮在冻结的主节点、路径、参考和批前 global-best 快照上生成最多两条编号、单行、自包含
的自然语言 Action。每条 Action 再单独调用 Code 阶段生成完整程序。Action 与 Code 看到
同一主轨迹及可选参考上下文，避免“决定修改”和“实现修改”之间丢失历史条件。

上下文至少包含任务契约、当前完整程序、最近祖先历史、当前节点已测试的直接分支、算子
要求，以及双轨迹时的参考程序和路径。超出模型 input limit 时先裁剪较早参考历史和分支
摘要；仍无法容纳时退化为单轨迹，不发送超限请求。

transport 或解析失败不消耗 evaluator；evaluator 一旦启动，无论 timeout、runtime error、
NaN/Inf 或其他无效结果都计入预算。只有有限结果进入树。global best 按原始任务 fitness
比较，完全同分时按 LOC 和确定性节点顺序择优。复杂度不是连续惩罚，也不拒绝更长但更优
的程序；`code_hash` 只记录重复率，V8 初版不在线去重。

## 9. 完整搜索流程

```text
1. 创建虚拟根。
2. 在总预算内生成并评价 30 个有效根程序。
3. 重复直到预算耗尽或安全停止：
   a. 根据全树质量和剩余预算计算 UCT；
   b. 从根开始递归选择，并按 progressive widening 决定何时停止下降；
   c. 选择四个轨迹算子之一，必要时选择其他根分支参考；
   d. 构造当前路径、已测试分支和参考上下文；
   e. 生成最多两条 Action 及其完整 Code 并真实评价；
   f. 有效候选从唯一主节点写入子节点和边；
   g. 沿祖先路径 max-backup G，更新 global best；
   h. 保存 candidates、edges、LLM 调用、决策和 checkpoint。
4. 返回全部有效节点中 fitness 最优且同分最短的程序。
```

安全停止包括连续生成/解析失败、连续批次没有有效候选和 evaluator 预算耗尽。停止时
保留已经形成的树事实，不把失败候选转换为节点。

## 10. 可恢复性与实验边界

checkpoint 保存虚拟根、全部节点和边、$N/f/G$、最佳后代、global best、预算进度、
停滞计数、搜索配置、任务和模型非密钥身份以及 RNG 状态。恢复时验证树无环、每个非根
节点恰有一个父代、fitness 有限、LOC 和 hash 可重算、$G$ 与 best pointer 一致。旧版本
checkpoint 不自动迁移。

原始运行目录分为：

```text
<run>/
  run_config.json
  logs/progress.log, errors.jsonl, summary.json
  artifacts/candidates.jsonl, edges.jsonl, llm_calls.jsonl, decisions.jsonl
  checkpoints/latest.json
```

过程审计应统计树深度、宽度、根分支覆盖、widening 触发、严格突破、父子结果、算子有效率、
重复代码率、LLM 调用/token、evaluator 失败和运行时间。过程指标不能替代 held-out。

V8 初版正式实验使用统一 1000 evaluator 预算、四任务、每任务三次独立运行和完整
held-out。结果显示 V8 在十一方法同场中的 15 个规模平均名次为 4.067；该数字是当前
任务、数据、模型、种子和重复数下的描述性结果，不构成树机制普遍有效的证明。V8.2 的
机制修改后独立报告，不能把两版结果合并为一个协议。

## 11. 证伪条件

以下观察会削弱 V8 的设计假设：完整树没有增加低初始质量节点获得后续突破的机会；树
主要产生重复状态或无效深链；max-backup 使预算过早锁定偶然高分路线；progressive widening
使大量初始路线在预算内无法获得扩展；轨迹和已测试分支不改变生成或增加失败；或者 search
fitness 提高但 held-out 泛化下降。

相关实现位于 `llm4ad/method/traceaad_v8/`。V8.2 的独立设计见
[TraceAAD V8.2](TraceAAD-v8.2完整机制设计.md)，实验依据见[实验配置](../experiments/配置.md)
和[实验总汇](../results/实验总汇.md)。
