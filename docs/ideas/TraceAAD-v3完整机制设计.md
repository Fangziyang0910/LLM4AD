# TraceAAD v3：基于算法改进轨迹的自动算法设计

> 本文描述当前仓库实际实现并用于 `version3` 实验的 TraceAAD v3。TraceAAD v4 的轨迹种群、softmax 选择和语义搜索算子设计见[TraceAAD v4 完整设计](TraceAAD-v4完整机制设计.md)。

## 1. 问题与方法概览

大模型驱动的自动算法设计反复执行生成、评估和筛选。每次评估之后，搜索需要决定下一步从哪个方案出发，并引入何种算法思想。当前程序及其适应度只描述算法的当前截面；算法改进轨迹进一步记录程序如何形成，包括引入过什么思想、哪些修改带来提升、退步发生在何处，以及当前路线是否仍有继续发展的可能。

TraceAAD v3 将轨迹作为搜索个体。程序是实际评估对象，轨迹是父代选择和 LLM 生成上下文的基本单位。搜索维护一个有界活跃轨迹种群，通过轨迹评分分配预算，通过四种搜索算子产生新程序，再将评估结果写回程序派生图和轨迹种群。

核心科学主张是：`[待验证]` 在当前程序、LLM 和评估预算相同的条件下，与当前程序真实匹配的改进历史能够提供更有效的下一步算法设计信息。

给定任务描述、待设计函数、LLM、评估器和评估预算 $B$，TraceAAD v3 在训练集反馈下搜索适应度最优的有效程序，测试集结果不进入搜索过程。

## 2. 程序、修改与轨迹

### 2.1 程序节点

每个有效程序记为

$$
p_i=(c_i,\mathrm{idea}_i,f_i),
$$

其中 $c_i$ 是完整程序代码，$\mathrm{idea}_i$ 是生成时记录的算法思想，$f_i$ 是训练集适应度。评估失败的程序消耗预算，但不进入程序派生图，也不建立轨迹。

### 2.2 父子修改边

除独立重新探索产生的新起点外，每个新程序只有一个直接父程序。一条父子修改边记为

$$
e=(p_u,p_v,\mathrm{action},\mathrm{op},\Delta,\mathrm{outcome}).
$$

- $p_u$ 是修改前的父程序，$p_v$ 是修改后的子程序；
- $\mathrm{action}$ 是 LLM 在搜索算子约束下提出的具体修改；
- $\mathrm{op}$ 是本次实际使用的搜索算子；
- $\Delta$ 是该步修改的有向适应度变化；
- $\mathrm{outcome}$ 是由 $\Delta$ 自动标记的提升、停滞或退步。

最大化和最小化任务分别使用

$$
\Delta=
\begin{cases}
f_v-f_u, & \text{最大化},\\
f_u-f_v, & \text{最小化}.
\end{cases}
$$

当 $\Delta>10^{-6}$ 时标记为提升，当 $\Delta<-10^{-6}$ 时标记为退步，其余情况标记为停滞。该结果由评估器自动产生，只描述当前一步的直接效果，不包含后续收益回传。

### 2.3 有界轨迹

一条轨迹为

$$
\tau=(p_0,e_1,p_1,\ldots,e_L,p_L).
$$

轨迹同时包含当前终点程序 $p_L$ 及其来时路。默认每条轨迹最多保存 8 个程序节点，超出后保留最近后缀。

从终点继续修改会形成更长的新轨迹；从中间程序重新分叉会保留此前前缀并接出新路线；独立重新探索建立新的单节点轨迹。原轨迹不会被覆盖。程序派生图在 v3 中是单父结构，每个子程序至多有一条入边。

轨迹具有 `active` 和 `archived` 两种状态。只有活跃轨迹参与父代选择、归一化、算子供体选择和种群生存；归档轨迹继续保存在实验状态中。

### 2.4 跨轨迹成败经验

当前轨迹是下一步生成的主要历史。系统还会从程序派生图中选取少量其它路线的成功与失败 action：优先选择与当前算子相同的修改，再按 $|\Delta|$ 排序，规范化后相同的 action 只保留一条。每次 action 提示最多加入 2 条成功经验和 2 条失败经验。

## 3. 完整搜索流程

```text
算法 TraceAAD v3
输入: 任务描述、待设计函数、LLM、评估器、预算 B
输出: 搜索过程中最优的有效程序 p*

1.  生成并评估 4 个具有不同思想的初始程序
2.  每个有效初始程序建立单节点轨迹，形成活跃轨迹种群 P
3.  记录当前最好程序 p*

4.  while 已评估程序数 < B do
5.      计算活跃轨迹的 Q、P、D 和 UCB，选择得分最高的轨迹 tau
6.      根据算子历史收益和 UCB 选择一个当前可用算子 op
7.      op 确定父程序 base；novelty_jump 不设置父程序
8.      若存在 base：基于轨迹历史、跨轨迹经验和 op 提出 2 条 action，
        再分别实现为完整候选程序
9.      若不存在 base：直接生成 2 个独立完整程序
10.     逐个评估候选程序
11.     对每个有效候选记录程序节点、父子边和新轨迹
12.     相似度达到 0.92 且没有刷新全局最好时，将新轨迹归档
13.     更新 p*、算子批次收益和实际使用轨迹的访问次数
14.     去除重复路径；活跃轨迹超过 160 条时执行生存筛选

15. return p*
```

历史在三处参与搜索：轨迹评分决定继续哪条路线，算子决定如何利用路线中的提升、退步与其它路线思想，提示词把相应历史提供给 LLM。新结果随后成为下一轮的搜索证据。

## 4. 轨迹评分与父代选择

轨迹价值包含终点质量、沿途改进潜力和路线差异：

$$
V(\tau)=0.55Q(\tau)+0.25P(\tau)+0.20D(\tau).
$$

所有归一化和相似性比较都以当前活跃轨迹种群为参照。

### 4.1 终点质量 $Q$

先统一优化方向：

$$
\tilde f(p)=
\begin{cases}
f(p), & \text{最大化},\\
-f(p), & \text{最小化}.
\end{cases}
$$

设 $\tilde f_{\min}$、$\tilde f_{\max}$ 为活跃轨迹终点的最小和最大方向统一适应度，则

$$
Q(\tau)=
\frac{\tilde f(p_L)-\tilde f_{\min}}
{\tilde f_{\max}-\tilde f_{\min}}.
$$

当所有活跃终点适应度相同时取 $Q(\tau)=0.5$。

### 4.2 沿途改进潜力 $P$

使用相同的活跃种群适应度范围归一化轨迹节点：

$$
q_i=\frac{\tilde f(p_i)-\tilde f_{\min}}
{\tilde f_{\max}-\tilde f_{\min}},
\qquad r_i=q_i-q_{i-1}.
$$

近期步骤权重更高，折扣系数为 $\gamma=0.8$：

$$
P(\tau)=
\frac{\sum_{i=1}^{L}\gamma^{L-i}r_i}
{\sum_{i=1}^{L}\gamma^{L-i}}
+0.25\frac{\#\{i:r_i>10^{-6}\}}{L}
-0.5\frac{\sum_{i=1}^{L}\gamma^{L-i}\max(-r_i,0)}
{\sum_{i=1}^{L}\gamma^{L-i}}.
$$

三项分别描述近期净变化、提升步骤出现比例和近期回撤。单节点轨迹或活跃种群没有有效归一化尺度时取 $P(\tau)=0$。该值是局部延续性信号，不是长期信用分配。

### 4.3 路线差异 $D$

两条轨迹的相似度结合终点代码和修改结果模式：

$$
\mathrm{sim}(\tau_a,\tau_b)
=0.7\,\mathrm{sim}_{\mathrm{code}}(\tau_a,\tau_b)
+0.3\,\mathrm{sim}_{\mathrm{trace}}(\tau_a,\tau_b).
$$

代码相似度是规范化 token 集合的 Jaccard 相似度；轨迹相似度是 $(\mathrm{op},\mathrm{outcome})$ 集合的 Jaccard 相似度。路线差异为

$$
D(\tau)=1-\max_{\tau'\in\mathcal P,\tau'\ne\tau}
\mathrm{sim}(\tau,\tau').
$$

当前没有其它活跃轨迹时取 $D(\tau)=1$。

### 4.4 访问次数与选择

记 $n(\tau)$ 为轨迹访问次数，$N$ 为活跃轨迹总访问次数：

$$
\mathrm{score}(\tau)
=V(\tau)+0.4\sqrt{\frac{\log(N+2)}{n(\tau)+1}}.
$$

每轮确定性选择得分最高的轨迹。$V$ 提供当前利用价值，UCB 为较少访问的活跃轨迹提供探索机会。

## 5. 搜索算子与真实提示词

轨迹选择决定继续哪条路线，搜索算子决定如何继续。v3 使用四种算子，并为每种算子维护尝试次数 $n_i$ 和平均批次收益 $\bar r_i$：

$$
S_i=\bar r_i+0.5\sqrt{\frac{\log(N_{\mathrm{op}}+1)}{n_i}}.
$$

未尝试过的可用算子优先获得一次机会。候选程序相对父程序的有向变化先除以当前活跃种群的适应度尺度，再经 $\tanh$ 压缩到 $[-1,1]$。一次生成多个有效程序时取平均效用；没有有效程序时收益为 $-1$。

### 5.1 共享的两阶段生成

除 `novelty_jump` 外，算子先让 LLM 提出 action，再逐条实现完整代码。Action 提示的真实结构为：

````text
[Task Description]
<TASK_DESCRIPTION>

[Algorithm Improvement History]
The selected trajectory records the modifications that led to the current program.
<FITNESS_DIRECTION>
<RECENT_HISTORY_UP_TO_5_STEPS>

[Cross-Trajectory Action Evidence]
Successful actions:
<UP_TO_2_SUCCESSFUL_ACTIONS>
Failed actions:
<UP_TO_2_FAILED_ACTIONS>

[Operator]
name=<OPERATOR_NAME>
Constraint: <OPERATOR_CONSTRAINT>

[Base Program To Modify]
Continue from Node p<BASE_NODE_ID>. Selection reason: <BASE_REASON>.
Idea: <BASE_IDEA>
Code:
```python
<BASE_PROGRAM_CODE>
```

[Target Function Contract]
Only evolve:
```python
<TARGET_FUNCTION_WITH_EMPTY_BODY>
```

[Instruction]
Use the selected trajectory as the main account of how the current program was formed.
Use cross-trajectory actions only as supporting evidence of what worked or failed.
Propose exactly 2 concrete next-step modifications.
Each modification must change one main algorithmic idea and follow the operator constraint.
Do not output code or rationale.
Return only a numbered list of exactly 2 ideas, one per line.
````

每条 action 随后使用统一代码生成提示：

````text
[Task Description]
<TASK_DESCRIPTION>

[Current Program]
Node p<CURRENT_NODE_ID>
Idea: <CURRENT_IDEA>
Code:
```python
<CURRENT_PROGRAM_CODE>
```

[Requested Modification]
<ACTION>

[Target Function Contract]
<TARGET_FUNCTION_WITH_EMPTY_BODY>

[Instruction]
Implement the requested modification as a new complete implementation of the target function.
Keep the function name, arguments, return type, and output contract unchanged.
Return only the new idea and complete code in this format:
Idea: <brief algorithm idea>
Code:
```python
<complete function implementation>
```
Do not include rationale, analysis, tests, or extra text.
````

### 5.2 `endpoint_refine`：继续改进终点

该算子始终可用，以当前选中轨迹的末节点为父程序：

```text
Continue refining the current best direction. Propose ONE targeted modification
that strengthens the mechanism which recently improved fitness. Use the recorded
trajectory as evidence and avoid directions that regressed.
```

### 5.3 `backtrack_branch`：从中间程序重新分叉

该算子扫描活跃种群中的可分叉轨迹。候选锚点包括最近退步前的父程序、连续停滞前最近一次提升后的程序，以及轨迹内适应度最高的程序。锚点质量、此前正向改进和此后回撤共同构成分叉分数；只有最终锚点位于末节点之前时，该轨迹才可分叉。

```text
The selected trajectory's endpoint regressed or saturated, but an earlier prefix
was strong. Branch from that high-value prefix and propose a modification DIFFERENT
from the one that caused the regression or plateau.
```

### 5.4 `mechanism_crossover`：引入互补思想

该算子在存在另一条活跃轨迹时可用。它以当前轨迹末节点为唯一父程序，从其它活跃轨迹选择思想供体。供体得分同时考虑与当前路线的差异和供体终点质量：

$$
\mathrm{donor\_score}
=1-\mathrm{sim}(\tau,\tau_d)+0.3Q(\tau_d).
$$

供体只提供末节点记录的算法思想；新程序只连接当前轨迹，不连接供体轨迹。

```text
Recombine: transplant exactly ONE clear algorithmic idea from a donor trajectory
into the current base program. Donor idea for reference: <DONOR_ENDPOINT_IDEA>.
Do NOT replace the whole program; keep the existing structure and change only
that single idea.
```

### 5.5 `novelty_jump`：重新探索

该算子始终可用，不选择父程序。系统从活跃轨迹中选出价值最高的至多四个不同思想，要求 LLM 生成与它们不同的完整程序，并建立新的单节点轨迹。

````text
<TASK_DESCRIPTION>

Generate a complete implementation for the target Python function. Novelty jump:
design a NEW complete algorithm that uses a clearly different algorithmic idea
from the current active elites.<AVOID_IDEAS_CLAUSE> Build a fresh solution from
scratch; do not continue an existing program.
Keep the function name, arguments, return type, and output contract unchanged.

Output format:
Idea: <brief algorithm idea>
Code:
```python
<TARGET_FUNCTION_WITH_EMPTY_BODY>
```
````

## 6. 候选过滤与种群生存

### 6.1 相似候选过滤

每个有效候选先写入轨迹记录，再计算它与其它活跃轨迹的最大软相似度。相似度达到 $0.92$ 时，新轨迹立即归档；如果新程序刷新全局最好，则无论相似度如何都保留。该机制试图减少近似路线重复消耗评估预算，但相似度代理及阈值缺少独立验证。

### 6.2 重复路径与活跃上限

完全相同的轨迹路径只保留访问次数更多的一条。随后重新计算全部活跃轨迹的选择分数。当活跃轨迹超过 160 条时，保留得分最高的 160 条，并保证至少留下一条终点为全局最好程序的轨迹。其它轨迹转为归档状态，不再参与后续父代选择。

因此，v3 同时包含两类预算控制：相似过滤决定新轨迹能否进入活跃种群，生存筛选决定活跃种群超过上限后保留哪些轨迹。

## 7. 默认实验配置

| 机制配置 | v3 默认值 |
| --- | ---: |
| 初始程序评估数 | 4 |
| 每轮 action 数 | 2 |
| 轨迹最大程序节点数 | 8 |
| 活跃轨迹上限 | 160 |
| 相似过滤阈值 | 0.92 |
| $V$ 中 $(Q,P,D)$ 权重 | $(0.55,0.25,0.20)$ |
| 代码与轨迹相似度权重 | $(0.7,0.3)$ |
| 路径折扣 $\gamma$ | 0.8 |
| 轨迹 UCB 系数 | 0.4 |
| 算子 UCB 系数 | 0.5 |
| 当前轨迹历史最大步数 | 5 |
| 跨轨迹成功/失败 action 数 | 2 / 2 |

## 8. 机制边界与验证问题

v3 是已经运行的实验基线。它同时使用轨迹历史、活跃种群、相似过滤、轨迹 UCB 和算子 UCB，因此最终结果不能直接归因于某一个组件。其主要待验证问题包括：

1. 真实且匹配当前程序的轨迹历史是否提高下一步改进成功率；
2. $Q/P/D+\mathrm{UCB}$ 是否比只按终点质量选择更有效；
3. 相似过滤和 160 条生存上限是否改善样本效率，还是提前排除有潜力路线；
4. 算子 UCB 是否提供有效调度，还是引入不必要的归纳偏置；
5. 四种算子是否真正利用了不同历史信息。

TraceAAD v4 将这些观察转化为更清晰的轨迹种群设计：取消模糊相似过滤，使用固定规模种群和 softmax 完成父代选择与生存更新，并以面向算法思想的语义算子替换 v3 算子。
