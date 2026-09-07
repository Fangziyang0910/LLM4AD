# TraceAAD V9.16 完整机制设计

V9.16 是一个机制识别版本。V9.15 的重放显示，`C_traj` 会广泛改变父节点排序，初生保护 `B` 对真实选择的影响很小；Explore child 的出生回撤与后续机会不足是值得检验的现象，但尚未证明“三步续段”就是根因。

本版本把问题收窄为一个可检验的假设：只按当前质量一步步重选，会不会让一部分 Explore 子代在还没改几步之前就失去机会？干预是对有效 Explore 子代随机提供一次固定的、连续的三步 Refine。

## 1. 研究对象

一次评价预算回答三个操作问题：

1. 从哪个已有算法继续；
2. 用哪条形成历史作为生成上下文；
3. 本次生成采用 Refine 还是 Explore 意图。

V9.16 只改变第一个问题的一小部分：成功 Explore child 有固定概率获得 landing，landing 连续沿同一入口进行三次 Refine。形成历史仍只用于提示，Explore/Refine 仍只改变生成意图。

`entry` 是来源记录单位，不等同于算法簇。初始根各自建立 root entry；成功 Explore child 建立新的 Explore entry；该 child 的 Refine 后代继承入口 ID；后续 Explore child 建立新入口。入口 ID 只用于记录 landing、投入和谱系，不作为质量标签。

## 2. 固定控制与直接因果对照

V9.16 的算子先验固定为 `Refine=0.7`、`Explore=0.3`。

V9.16 的直接因果对照是同一固定 proposal protocol 下的：

```text
q baseline       = quality-only allocation, no landing
q + landing      = the same baseline with fixed Explore landing
```

两者固定模型、提示、初始根、随机种子、错误处理和 primary evaluator slots。只有这组对照用于归因 landing 的作用。

V9.16 的父节点质量分数为：

$$
S(a)=q(a).
$$

其中 `q` 是当前真实质量。所有有效节点经过 Boltzmann 分布采样，逆温度由目标 `ESS=max(0.1N,2)` 求解；`N` 为有效节点数。该规则沿用 V9.15 的抽样强度，但不再使用 `B` 或 `C_traj`。

算子先验固定为：

$$
P(\mathrm{Refine})=0.7,\qquad P(\mathrm{Explore})=0.3.
$$

`p_E` 不再随全局停滞变化。这样 V9.16 不同时改变新入口产生率。

这一改动检验一个明确的取舍：普通协议在每次评价后立即重新选择，具有即时纠错能力；landing 暂时禁止全局重选，用少量连续预算换取结构性 proposal 的短期发展视界。结果应解释为这两种分配协议的比较。

## 3. Landing 规则

### 3.1 预算

总预算为 `1000` 个 `primary evaluator slots`，初始化建立 `8` 个有效根。初始化后的剩余预算中，最多 `10%` 用于 landing；在标准预算下为 `99` 个 slot，因此最多完成 `33` 个三步 landing。未用完的 landing 配额回到普通搜索。

每个 landing 步的第一次候选占用一个 primary slot。解析失败、运行失败、超时和两次修复失败都算该步失败并消耗该 slot，失败后 landing 继续尝试下一步。修复可以再次调用 evaluator，但只增加 `repair evaluator calls`，不增加 primary slots。

### 3.2 抽样

每个成功 Explore child 以固定 `landing_probability=0.125` 独立抽签一次。抽签只在 child 创建时进行，结果写入 checkpoint；同一 entry 不重复获得 ticket。抽签随机数使用独立于普通父节点选择的派生种子，恢复运行不会改变既有 ticket。

`0.125` 与 `10%` 上限按固定 `p_E=0.3`、每次 landing 三个 slot 粗略匹配：每个普通 slot 的期望 landing 消耗为 `0.3 * 0.125 * 3 = 0.1125`，对应总预算约 `10.1%`。有效 Explore 比例会使实际占用低于上限，因此 `10%` 是最大干预预算，不是保证每个任务使用的比例；正式报告实际 landing ratio。

抽中后立即执行该 entry 的三步 landing；landing 完成后恢复普通搜索。landing 配额不足三步时不再发放 ticket，未用配额由普通搜索使用。普通搜索不会因为 landing 结果改变父节点分数。

### 3.3 三步延续

landing 从获得 ticket 的 Explore child 开始，连续执行最多三步 Refine：

1. 第一步父节点是 Explore child；
2. 后续步骤父节点是上一步产生的最新有效 child；
3. 生成提示仍包含该父节点和完整形成历史；
4. landing 步固定使用 Refine 意图；
5. 没有有效 child 时，下一步仍按原入口的最近有效节点继续；若不存在有效节点，landing 提前结束。

landing child 正常加入主树、参与 best 和后续普通搜索。这是一个在线预算干预，V9.16 报告其搜索影响，不把它解释成无干预反事实。`H=3`、连续执行和 Refine-only 是本版本的具体 policy；结果不外推到所有 continuation policy。

## 4. 记录的过程量

每个入口和每次 landing 记录：

- ticket 是否抽中以及抽签时的 evaluator slot；
- 三步的父节点、子节点、状态、fitness、修复次数和耗时；
- 有效步数、严格改进步数、相对入口父代的最终质量差；
- 是否恢复入口父代、是否超过入口父代、最大增益；
- 入口内连续 Refine 长度、全局 best-at-budget 曲线和运行间离散度。

对 `q baseline` 还要记录 Explore child 的出生质量差：

$$
\Delta q_{birth}=q(child)-q(parent).
$$

按出生回撤分层报告普通搜索中的入口重访概率和首次重访等待时间。这样可以先检查“出生回撤导致普通分配早夭”这一前提，再把 landing 的入口内发展、后代增益和有限预算结果串起来。

这些量作为过程分析证据，质量分数严格由 evaluator 评价决定。

## 5. 错误处理与预算口径

V9.16 沿用 V9.15 的错误处理：一次初始候选最多两次有界修复，完整错误反馈返回模型。初始尝试计入 `budget_slots`，修复单独计入 `repair LLM calls` 与 `repair evaluator calls`；总 evaluator 调用数不作为 primary budget。所有 ordinary 和 landing 初始尝试使用同一规则。

运行日志必须区分 `ordinary` 与 `landing`，并记录 `landing_id`、`landing_step`、`entry_id`、`parent_id`、`attempt_kind` 和状态。checkpoint 必须保存未完成 ticket、当前 landing step、普通搜索计数和树，支持中断恢复。

## 6. 原子运行协议

````text
Input: task, evaluator, LLM, budget B = 1000

Generate 8 valid roots.
Set Refine/Explore prior to 0.7/0.3.
Set landing cap to floor(0.10 * (B - 8)).

While evaluator slots remain:
    If a landing ticket is active and the landing cap has room:
        run the next Refine step from that ticket's latest valid child;
        charge one primary evaluator slot;
        record the step and keep the ticket active until 3 steps;
    Else:
        sample intent with fixed 0.7/0.3 prior;
        sample parent by q(a) through the fixed ESS rule;
        generate, evaluate, and record one ordinary child;
        when a successful Explore child is created, draw its one landing ticket.

Return the best valid algorithm by the true search objective.
````

## 7. 实验对照与判断

V9.16 的主对照是：

| 对照 | 唯一变化 | 目的 |
| --- | --- | --- |
| `q` baseline | 固定 `0.7/0.3`，无 landing | 质量分配基线 |
| `V9.16` | 在同一基线上加入固定 landing | 检验短期连续精炼机会 |

V9.15 作为已完成版本保留其原始结果。V9.16 的 5 个正式任务为 `tsp_construct`、`cvrp_aco`、`op_aco`、`online_bin_packing` 和 `vrptw_construct`；每个任务使用 `1000` 个 primary evaluator slots 和三次重复。三重复是筛选批次，不单独承担“机制已经稳定改善”的结论；若出现清晰信号，再用新的独立 seeds 将关键任务扩展到至少六次重复。

报告顺序为：landing 激活与等待、入口内发展过程、`100/250/500/750/1000` best-at-budget、训练集 best、测试集。过程改善不替代最终测试结果。

若 landing 没有过程或有限预算改善，结论限定为“这一随机、连续、三步 Refine landing 未显示收益”，不据此否定所有 delayed continuation，也不继续向 V9.16 添加信用项。只有筛选批次出现清晰过程信号并经独立重复确认后，才在后续版本研究 landing 对象和视界的自适应分配。
