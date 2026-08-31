# TraceAAD V9.22：校准的假设机会与双基准实现信用

## 1. 版本决定

TraceAAD V9.22 把一次改写机会仍然定义为一个待检验的算法思想假设及其实现，
但修正 V9.21 首跑中三个直接影响预算判断的信号问题：质量尺度不再依赖固定的
root-MAD，工作实现和稳定脚手架分别结算响应，`continue` 与 `branch` 的预算由
动作 UCB 决定。

搜索对象不是一份脱离形成过程的代码。每个假设同时保留一份可以返回的
`stable scaffold`、一份正在兑现该假设的 `working implementation`，以及所有
实现尝试的 evaluator 证据。

V9.22 在线只运行标准任务 evaluator，保留 V9.21 的短因果链：

$$
\text{idea hypothesis}
\rightarrow
\text{independent realizations}
\rightarrow
\text{evaluator evidence}
\rightarrow
\text{next opportunity}
$$

BehaveSim、Idea embedding、per-instance objective 和预定义语义算子不进入在线
控制器。它们需要各自的匹配实验。

## 2. 任务对象

一个 AAD 任务由问题描述、可编辑程序模板和 evaluator 组成：

$$
\mathcal T=(d_{\mathcal T},K,\mathcal E),
\qquad P_r=K[r],
\qquad q(r)=\mathcal E(P_r;S_{train}).
$$

五个正式任务在 runner 中都转换为 higher-is-better fitness。primary budget 只
计算真实候选槽位；repair evaluator 调用单独记录，不增加 primary slot。

## 3. 状态对象

### 3.1 Program Node

有效且非重复的候选建立一个 `ProgramNode`，保存：

```text
id, code, fitness, parent_id, hypothesis_id
idea, role, primary slot
```

`parent_id` 是真实形成边。无效、超时和重复响应保留在 evaluator ledger 中，
不建立新的有效节点。

### 3.2 Idea Hypothesis

每个假设保存：

```text
entry_idea
source_node_id
stable_scaffold_node_id
working_node_id (optional)
parent_hypothesis_id
public_card provenance (optional)
response_working_values
response_scaffold_values
action statistics
```

`entry_idea` 是模型本轮声明的可检验主张，不是已经确认的算法簇标签。

### 3.3 Stable Scaffold

`stable_scaffold` 是该假设随时可以返回的最高质量有效代码。

- root 假设的 scaffold 是 root program；
- branch 假设的 scaffold 是产生它的 source scaffold；
- 实现严格超过当前 scaffold 时更新 scaffold；
- scaffold fitness 在单个假设内单调不降。

退步实现不会覆盖 scaffold，也不因退步获得质量信用。

### 3.4 Working Implementation

`working` 是当前最接近兑现该假设的有效实现。它可以低于 scaffold：

- 有效候选高于当前 working 时更新 working；
- 候选超过 scaffold 时同时更新 scaffold；
- 新 branch 的 working 初始为空，第一次实现从 scaffold 开始。

working 用于 continue 的上下文和实现信用。它保留了“正在修复但还没有恢复到
稳定脚手架”的方向。

## 4. 原子搜索循环

初始化阶段生成 8 个独立 root，每个有效 root 消费一个 primary slot。普通 batch
开始时先冻结完整状态：

- 选中的 hypothesis、scaffold 和 working；
- scaffold 的真实 formation path；
- 当前 hypothesis 最近的 realization ledger；
- 一张可选的 public experiment card；
- 当前 scaffold 层的质量参考集合；
- 本 batch 的 action plan。

随后每个 batch 生成 2 个 Idea。Idea 的 proposal 可以是 `continue` 或 `branch`，
每个 Idea 从同一冻结快照独立生成 2 份完整代码。两份 realization 之间不共享
响应、代码或修复上下文。四个候选各占一个 primary slot，尾部预算不足时只执行
剩余前缀并记录实际 batch 大小。

```text
freeze hypothesis state and scaffold-layer ranks
    -> choose two proposal actions with action-UCB
    -> generate one Idea for each action
    -> independently realize each Idea twice
    -> evaluate each candidate and bounded repairs
    -> update working/scaffold, hypothesis credit, and action credit
    -> save checkpoint and select again
```

完整 batch 的上下文在第一份 realization 结算前就生成。第一份结果不会污染同一
Idea 的第二份 realization，也不会改变同 batch 另一 proposal 的 parent 快照。

## 5. 动态 scaffold mid-rank

### 5.1 质量参考集合

在 batch 开始时取当前 hypothesis 的 stable scaffold 节点，并按节点去重得到
质量参考集合 $B_t$。后代节点只作为形成和实现证据，不重复进入质量标尺。

### 5.2 质量分位

对 higher-is-better fitness，定义带并列处理的经验 mid-rank：

$$
Q_t(q)=\operatorname{midrank}_{B_t}(q)\in[0,1].
$$

最差和最好分别接近 0 和 1；全部相等或只有一个参考值时返回 0.5。该标尺在
每个 batch 重新计算，因此不会因为某个任务的 fitness 数值范围或 root-MAD
过小而整体饱和。

质量选择只使用 scaffold 层面的 $Q_t(q_{scaffold})$。working 节点可以被放入
同一参考集合中计算它当前所处的经验位置，但它不会改变 $B_t$。

### 5.3 响应

候选结算时冻结 $B_t$，并把候选与参考值加入临时样本，以便严格超过当前最大
scaffold 的候选仍产生正响应：

$$
r_w(c)=\operatorname{clip}\left(
Q_{B_t\cup\{q_c,q_w\}}(q_c)-
Q_{B_t\cup\{q_c,q_w\}}(q_w),-1,1\right),
$$

$$
r_s(c)=\operatorname{clip}\left(
Q_{B_t\cup\{q_c,q_s\}}(q_c)-
Q_{B_t\cup\{q_c,q_s\}}(q_s),-1,1\right).
$$

$q_w$ 和 $q_s$ 分别是本次 proposal 的 working 与 scaffold 基准。`invalid`、
`timeout` 和 `duplicate` 没有 fitness，统一写入最低可靠性响应 `-1`，但在统计
中保留各自的失败类型。

## 6. Hypothesis opportunity

实现响应先转为 $[0,1]$ 的 bounded credit，使用 hypothesis 内的
`response_working_values` 计算一步 progress UCB：

$$
\widehat C_t(h)=\operatorname{mean}\left((r_w+1)/2\right),
$$

$$
U_t(h)=\min\left(1,
\widehat C_t(h)+
\sqrt{\frac{\log(t+2)}{2(n_h+1)}}\right).
$$

其中 $n_h$ 是已经结算的 realization 数。将 $U_t$ 还原成 rank 增量上界：

$$
\Delta_t(h)=2U_t(h)-1.
$$

对 scaffold rank $Q_s$ 和 working rank $Q_w$，机会值为：

$$
O_t(h)=\max\left(Q_s,Q_w+\Delta_t(h)\right).
$$

这表示下一次机会的有限上界：已经兑现的 scaffold 质量不会被较差的 working
覆盖；仍有较大实现不确定性的 hypothesis 可以保留一次继续兑现的机会。这里的
机会值是排序分数，允许暂时超过 1 以保留 scaffold rank 的相对顺序。没有 trial
的新 branch 仍有 UCB 上界，不会因为 `working_node_id` 为空而被自动排除。

## 7. Action-UCB 与 proposal plan

动作层回答的是“哪种 proposal 更可靠地产生下一个可用 working 实现”，不把连续
质量差异和严格改善成功率混成一个量。对 `continue` 和 `branch` 分别记录 trial
数 $n_a$ 与严格改善 working 的次数 $s_a$：

$$
A_t(a)=\frac{s_a+1}{n_a+2}+
\sqrt{\frac{\log(t+2)}{2(n_a+1)}}.
$$

每个 batch 的两个 action slot 先确保尚未观察过的动作各出现一次；两种动作都
已有证据后，使用虚拟 trial 的 UCB 逐 slot 选择最大者。因此 plan 可以是
`[continue, branch]`、`[continue, continue]` 或 `[branch, branch]`，不再固定
为 50:50。动作的连续响应、usable 数、invalid、timeout 和 duplicate 仍写入
`action_stats`，用于离线分析和失败边界。

## 8. Proposal 上下文

### 8.1 Continue

Continue 看到任务契约、stable scaffold、working implementation（若两者不同）、
entry idea、scaffold formation path 和当前 hypothesis 的 realization ledger。
它可以修复、回退、重新实现或精炼同一个思想，但不能悄悄替换研究假设。

### 8.2 Branch

Branch 看到任务契约、stable scaffold、scaffold formation path、source hypothesis
的实现证据和最多一张 public card。它不看到 source working implementation，避免
新假设被一条未兑现的失败实现锚定。branch 的实现从 scaffold 开始。

### 8.3 Public card

全局 memory 只保存真实形成边上的严格改善节点。branch 最多接收一张来自其它
formation branch 的 card，包括 measured fitness transition、记录的 Idea 和
完整代码。card 是可复核的实验事实，不是算法簇标签或控制器评分。

## 9. 实现失败与恢复

每个候选在 evaluator 前写入 pending checkpoint。模型输出无效、执行异常或超时
时，最多生成两次 bounded repair；repair 保留 Idea、目标函数签名和失败信息，
评价调用单独计数。最终结果只结算一次 primary realization。

可靠性按 `invalid`、`timeout`、`duplicate`、`plateau`、`regress` 和 `improve`
分别记录。失败不会被默认 fitness 或静默 fallback 覆盖，也不建立 Program Node。

checkpoint 保存：

```text
nodes, hypotheses, realizations, attempts
primary/evaluator/LLM counters
action statistics and rank count
pending candidate and unfinished batch context
public memory ids and RNG state
```

unfinished batch 在写入 pending 时先标记当前 sibling 已被消费。进程恢复后结算
同一 response，不重新生成已经写入的 realization。

## 10. 预算与固定参数

| 参数 | V9.22 |
| --- | ---: |
| Primary evaluator slots | 1000 |
| Initial roots | 8 |
| Ideas per ordinary batch | 2 |
| Realizations per idea | 2 |
| Nominal primary slots per batch | 4 |
| Formation history shown | latest 8 edges |
| Realization evidence shown | latest 6 events |
| Response range | [-1, 1] |
| Allocation | hypothesis opportunity UCB |
| Action allocation | continue/branch action-UCB |
| Public cards per branch | at most 1 |
| Public card archive | latest 64 strict improvements |
| Max bounded repairs | 2 per candidate |
| Online BehaveSim | disabled |
| Idea embedding | disabled |
| Per-instance objective | disabled |

## 11. 可检验预测与识别实验

V9.22 的设计预测包括：

1. scaffold rank 的跨任务排序不会出现 V9.21 root-MAD 的固定尺度饱和；
2. 候选高于 working 但低于 scaffold 时，$r_w>0$ 且 $r_s<0$，这类方向仍会留下
   后续机会；
3. 新 hypothesis 不会因为没有 working 节点而失去初始探索机会；
4. action plan 会在动作成功率不同的任务上偏向观测到更可靠的 proposal；
5. branch prompt 的 working 隔离和批内冻结可以在日志中被直接审计。

这些是过程层预测，不是性能结论。最小匹配实验为：

- root-MAD 与动态 scaffold rank：保持生成、预算和机会公式的其余部分不变；
- scaffold-quality allocation 与 V9.22 opportunity：保持 action plan、2×2
  realization、prompt 和错误处理不变；
- 固定 50:50 与 action-UCB：保持 hypothesis 选择和上下文不变；
- `1 realization` 与 `2 realizations`：保持 Idea、parent 和总 primary budget
  口径一致；
- `private-only` 与一张真实 public card：保持 branch prompt 和 donor 规则一致。

完整搜索必须完成三次重复和 held-out 评估后，才报告 best-at-budget、跨任务均值
或泛化结论。联合版本结果只能评价整套 V9.22 搜索行为。

## 12. 证据边界

V9.22 规范确认的是状态、上下文、分配公式、错误边界和 checkpoint 协议。单元或
toy 测试可以确认实现遵守这些规则；它们不证明真实任务上的搜索质量。Idea 文本
不是算法簇真值，形成 path 不是语义标签，单步 response 也不是长期思想潜力的
估计。正式性能主张必须来自完整 primary budget、全部重复和 held-out 工件。
