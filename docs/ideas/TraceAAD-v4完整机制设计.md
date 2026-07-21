# TraceAAD v4：基于轨迹种群与语义算子的自动算法设计

> 状态：完整设计提案，尚未实现，需通过受控实验验证。当前已实现并运行的基线见[TraceAAD v3 完整机制设计](TraceAAD-v3完整机制设计.md)。

## 1. 核心设计

TraceAAD v4 将算法改进轨迹作为种群个体。每条轨迹保存一个算法如何逐步形成，包括引入过哪些思想、每次修改产生什么结果、哪些方向持续改进，以及哪些尝试形成失败边界。搜索选择一条或两条轨迹作为父代上下文，从轨迹中确定具体修改的锚点程序，再由语义算子要求 LLM 完成一种明确的算法设计任务。

v4 的搜索由四个核心部分构成：

1. **轨迹种群**保存当前参与繁衍的高价值算法改进路线；
2. **轨迹评分与 softmax**在有限预算下平衡高质量路线利用和其它路线探索；
3. **语义搜索算子**规定 LLM 本轮要完成的新思想提出、已有思想深化、双轨迹综合或思想迁移；
4. **完整轨迹档案**保存所有评估历史，使退出当前种群的路线仍可用于经验分析和跨轨迹证据检索。

程序是评估对象，轨迹是搜索个体。一次双轨迹生成同时使用两条完整轨迹作为设计上下文；两条轨迹分别提供锚点程序，新程序只评估一次，再分别接到两条来源轨迹上，形成两条新的轨迹个体。

核心科学主张是：`[待验证]` 以完整算法改进历史为父代个体，并让搜索算子直接规定如何利用历史中的有效思想和失败边界，能够提高下一步算法生成的有效性和有限预算下的搜索质量。

### 1.1 问题设定

给定算法设计任务 $x$、待设计函数接口、LLM、程序评估器 $F_x$ 和程序评估预算 $B$，目标是在预算内找到有效程序集合中的最优程序：

$$
p^*=\arg\max_{p\in\mathcal G_B}\tilde F_x(p),
$$

其中 $\mathcal G_B$ 是预算 $B$ 内得到有效评估结果的程序集合，$\tilde F_x$ 将最小化任务的评估值取反，使后续公式统一为数值越大越好。一次有效或失败的程序评估均消耗一单位预算；代码解析失败不调用评估器，也不进入程序集合、轨迹档案或当前种群。

搜索只使用训练集适应度。测试集结果用于最终评价，不反馈给父代选择、种群更新或 LLM 上下文。第一版假设每个 task 提供一个待设计函数及标量适应度，同一候选程序只评估一次。

### 1.2 v3 到 v4 的机制变化

| 机制 | v3 | v4 |
| --- | --- | --- |
| 搜索组织 | 上限 160 的活跃轨迹池 | 固定规模轨迹种群 |
| 父代选择 | $Q/P/D+\mathrm{UCB}$ 取最大 | $Q/P$ 评分后 softmax 抽样 |
| 种群更新 | 排名截断生存 | 少量精英保留 + softmax 无放回抽样 |
| 相似机制 | 0.92 候选过滤，$D$ 参与评分 | 不过滤候选，路线差异仅作观察指标 |
| 算子调度 | 算子平均收益 + UCB | 可用语义算子等概率选择 |
| 算子语义 | 终点深化、回退、思想供体、重启 | 新思想、思想深化、双轨迹综合、思想迁移 |
| 多轨迹写回 | 供体只提供一句思想，子程序仅接主轨迹 | 两条完整轨迹共同生成，子程序分别接入两条轨迹 |

## 2. 搜索对象与关系

### 2.1 程序节点

一个有效程序记为

$$
p=(\mathrm{code},\mathrm{idea},f),
$$

其中 `code` 是完整待设计函数，`idea` 是生成时记录的算法思想，$f$ 是训练集适应度。一次候选程序只评估一次，程序代码、算法思想和适应度只保存一份。

### 2.2 修改边

一条来源轨迹中的锚点程序与新程序之间形成一条修改边：

$$
e=(p_u,p_v,\mathrm{action},\mathrm{op},\Delta,\mathrm{outcome}).
$$

`op` 是本次语义算子，`action` 是该算子约束下提出的具体算法修改。最大化和最小化任务分别使用

$$
\Delta=
\begin{cases}
f_v-f_u, & \text{最大化},\\
f_u-f_v, & \text{最小化}.
\end{cases}
$$

当 $\Delta>10^{-6}$ 时自动标记为提升，当 $\Delta<-10^{-6}$ 时标记为退步，其余情况标记为停滞。

双轨迹生成时，新程序 $p_v$ 同时连接两个锚点程序。锚点不同时，两条边使用相同的 `action` 和 `op`，并分别根据各自锚点适应度计算 $\Delta$ 和 `outcome`；锚点相同时，两条新轨迹共享同一条修改边。因此程序及其来源关系构成 DAG。

### 2.3 轨迹个体

一条轨迹为

$$
\tau=(p_0,e_1,p_1,\ldots,e_L,p_L).
$$

轨迹给出一条具体算法改进路线。默认最多保存 8 个程序节点，超出后保留最近后缀。轨迹包含的核心信息是：

- 当前末节点程序及其算法思想；
- 形成该程序的有序修改历史；
- 每一步的直接评估结果；
- 轨迹中的最高分程序；
- 从某个中间程序继续尝试过的后续方向。

### 2.4 当前种群与完整档案

当前轨迹种群记为 $\mathcal P_t$，规模上限为 $M=10$。它承担父代选择、轨迹评分归一化和繁衍。完整轨迹档案记为 $\mathcal A_t$，保存搜索过程中产生的全部有效轨迹，不执行相似过滤和历史删除。

退出当前种群的轨迹继续保存在档案中，可用于：

- 机制观察和实验复现；
- 检索跨轨迹成功与失败 action；
- 分析思想重复、路线覆盖和长期信用；
- 后续轨迹学习与强化学习数据构造。

档案不直接参与父代 softmax，也不进入当前种群的质量归一化。

同一个程序可以出现在多条轨迹中。只要来时路不同，这些轨迹就是不同的种群个体，分别参与评分和生存；程序代码、算法思想和适应度仍只保存一份。

## 3. 完整搜索流程

```text
算法 TraceAAD v4
输入: 任务描述、待设计函数、LLM、评估器、预算 B
输出: 搜索过程中最优的有效程序 p*

1.  生成并评估 10 个具有不同算法思想的初始程序
2.  每个有效程序建立单节点轨迹
3.  初始化当前轨迹种群 P 和完整轨迹档案 A，记录最好程序 p*

4.  while 已评估程序数 < B do
5.      在当前种群 P 内计算每条轨迹的质量 Q、趋势 P 和价值 V
6.      按 softmax(V / T) 抽取主父代轨迹 tau_1
7.      从当前可用的语义算子中等概率抽取 op
8.      若 op 使用两条轨迹：
            从 P \ {tau_1} 按 softmax(V / T) 抽取 tau_2
9.      分别从来源轨迹中选择末节点或轨迹内最高分节点作为锚点
10.     将任务、锚点程序、形成历史、后续尝试和必要的第二条轨迹
        组织为可解释的算法设计上下文
11.     LLM 在 op 的约束下提出 2 条 action
12.     分别将每条 action 实现为一个完整候选程序并评估一次
13.     单轨迹算子为每个有效程序建立 1 条新轨迹；
        双轨迹算子为每个有效程序建立 2 条新轨迹
14.     所有新轨迹进入完整档案 A
15.     将当前种群与本轮新轨迹合并，执行精英保留和 softmax 生存抽样，
        得到下一轮固定规模种群 P
16.     更新最好程序 p*

17. return p*
```

一轮只评估实际生成的程序。双轨迹算子产生两个 action 时，评估预算增加 2，轨迹档案最多增加 4 条轨迹。

## 4. 轨迹评分与父代选择

轨迹评分服务于当前种群内的预算分配。v4 第一版使用终点质量和近期路径趋势：

$$
V(\tau)=0.6Q(\tau)+0.4P(\tau).
$$

路线差异不参与第一版选择分数。代码相似度和轨迹模式差异继续记录为观察指标，用于检验语义多样性是否需要显式进入搜索。

### 4.1 终点质量 $Q$

终点质量采用当前比较集合中的适应度百分位排名。父代选择时比较集合为当前种群；种群更新时比较集合为“当前种群与本轮新轨迹”的并集。

统一优化方向后的终点适应度记为 $\tilde f(\tau)$。设当前比较集合有 $K$ 条轨迹，$\operatorname{rank}(\tau)\in\{1,\ldots,K\}$ 按从差到好排序，则

$$
Q(\tau)=
\begin{cases}
0.5, & K=1,\\
\dfrac{\operatorname{rank}(\tau)-1}{K-1}, & K>1.
\end{cases}
$$

适应度相同时使用平均名次。百分位排名避免历史异常值持续压缩当前种群内部的质量差异，也避免同一程序的绝对分数尺度跨 task 不可比较。

### 4.2 近期路径趋势 $P$

每条修改边根据 `outcome` 转换为方向信号：

$$
z_i=
\begin{cases}
1, & \mathrm{outcome}_i=\mathrm{improve},\\
0, & \mathrm{outcome}_i=\mathrm{plateau},\\
-1, & \mathrm{outcome}_i=\mathrm{regress}.
\end{cases}
$$

使用 $\gamma=0.8$ 强调近期步骤：

$$
P_{\mathrm{raw}}(\tau)
=\frac{\sum_{i=1}^{L}\gamma^{L-i}z_i}
{\sum_{i=1}^{L}\gamma^{L-i}},
\qquad
P(\tau)=\frac{P_{\mathrm{raw}}(\tau)+1}{2}.
$$

单节点轨迹取 $P(\tau)=0.5$。该定义只使用提升、停滞和退步的方向，避免不同 task 的适应度变化幅度和不同搜索阶段的尺度差异直接改变轨迹权重。原始 $\Delta$ 继续保存在边上，用于过程分析和后续消融。

### 4.3 Softmax 父代选择

主父代从当前种群按 Boltzmann 分布抽取：

$$
\Pr(\tau_i\mid\mathcal P_t)
=\frac{\exp((V_i-V_{\max})/T)}
{\sum_{\tau_j\in\mathcal P_t}\exp((V_j-V_{\max})/T)}.
$$

第一版固定温度 $T=0.2$。减去 $V_{\max}$ 保证数值稳定。低温提高高价值轨迹的选择概率，所有有限分数轨迹仍具有非零概率，因此探索与利用通过同一个可解释参数控制。

双轨迹算子先抽取主父代，再从剩余种群中使用同一分布无放回抽取第二条轨迹。第二条轨迹承担共同设计来源，不通过额外的相似度公式重新排序。

## 5. 种群管理

### 5.1 初始化

搜索先生成并评估 10 个具有不同思想的程序，每个有效程序建立一条单节点轨迹。有效轨迹不足 10 条时继续生成，直到种群达到 10 条或预算用尽。初始轨迹同时进入当前种群和完整档案。

### 5.2 稳态更新

一轮生成结束后，当前种群与本轮新轨迹形成候选集合：

$$
\mathcal C_t=\mathcal P_t\cup\mathcal O_t,
$$

其中 $\mathcal O_t$ 是本轮产生的轨迹后代。系统在 $\mathcal C_t$ 内重新计算 $Q/P/V$，再形成下一轮种群：

1. 保留价值最高的 $E=2$ 条精英轨迹；
2. 从其余轨迹中按 $\mathrm{softmax}(V/T)$ 无放回抽取 $M-E$ 条；
3. 当候选总数不足 $M$ 时全部保留；
4. 所有未进入下一轮种群的轨迹继续保存在完整档案中。

精英保留维持已知高价值路线，softmax 生存为中等分数和新生轨迹保留机会。父代选择与种群更新使用同一轨迹价值和温度，避免引入第二套生存评分。

### 5.3 全局最好程序

全局最好程序从全部已评估程序中维护，独立于轨迹种群生存。包含最好程序的轨迹可能退出当前种群，但程序本身和对应轨迹始终保存在完整档案中。精英保留通常会让高质量路线继续繁衍，无需额外强制保护某一个程序节点。

## 6. 轨迹选择、锚点选择与算子选择

三个决策按固定顺序发生：

1. 轨迹评分和 softmax 选择父代轨迹；
2. 在每条来源轨迹内部选择具体修改的锚点程序；
3. 语义算子规定如何使用来源轨迹完成下一步设计。

### 6.1 锚点程序

每条来源轨迹提供两个候选锚点：

- **末节点**：包含该路线最新累积思想的程序；
- **轨迹内最高分节点**：该路线已经发现的最好程序。

两者相同时直接选择该程序；两者不同时等概率选择。最高分按 task 优化方向判断，适应度相同时选择时间上更靠后的程序。

选择中间的最高分节点后，锚点之后的既有尝试继续作为历史证据提供给 LLM。它们说明从该程序继续尝试过哪些方向以及这些方向产生了什么结果。

### 6.2 语义算子调度

第一版从当前可用算子中均匀选择：

$$
\Pr(\mathrm{op}_i\mid\mathcal O_{\mathrm{available}})
=\frac{1}{|\mathcal O_{\mathrm{available}}|}.
$$

拥有至少两条轨迹时四个算子各占 25%；只有一条轨迹时，`trace_ideate` 和 `trace_refine` 各占 50%。算子统计只用于过程观察，不反馈选择概率。

## 7. 轨迹语义搜索算子

四个算子的差异是 LLM 本轮承担的搜索任务不同。它们共享程序生成与评估流程，每次提出 2 条 action。

| 算子 | 来源轨迹 | 搜索任务 | 主要修改范围 |
| --- | ---: | --- | --- |
| `trace_ideate` | 1 | 基于整条历史提出新算法思想 | 可调整整体机制 |
| `trace_refine` | 1 | 深化轨迹中已有的有效思想 | 一个逻辑、公式或参数规则 |
| `trace_synthesize` | 2 | 综合两条轨迹的有效原则 | 建立两种原则的算法关系 |
| `trace_transfer` | 2 | 把一条轨迹中的有效思想迁移到另一条 | 保留主结构并迁移一个思想 |

### 7.1 `trace_ideate`：提出新思想

该算子读取一条轨迹的有效原则、失败尝试和停滞边界，要求提出尚未在这条历史中出现的新算法思想。

```text
[What to design next]
Read the algorithm's development history as design evidence. Preserve useful
principles revealed by improvements, and use later regressions or plateaus as
tested boundaries.

Propose two alternative actions. Each action must contain ONE genuinely new
algorithmic idea for the algorithm shown above. The idea may introduce a new
mechanism, score component, ordering principle, state variable, or interaction
between existing components. Ground each idea in the provided history and make
it materially different from the recorded attempts.

Return exactly two numbered action lines. Each line must state the new idea and
the concrete algorithmic change, without code or general explanation.
```

代码生成提示中的算子约束为：

```text
Implement the requested new idea as a complete valid implementation of the
algorithm shown above. Reorganize the scoring or decision flow when required by
the idea. Keep the function name, arguments, return type, and output contract
unchanged. A previously unsuccessful modification may be used when the action
states the specific repair.
```

### 7.2 `trace_refine`：深化已有思想

该算子选择轨迹中已经显示出价值的一个思想，聚焦修改其逻辑、公式、组件关系或参数配置。Action 使用 `mechanism` 或 `parameter` 标记修改范围。

```text
[What to design next]
Identify one valuable algorithmic idea already present in the development history.
Preserve that idea as the main design principle and propose a focused improvement.
Each action must choose exactly one scope:

- mechanism: change one formula, component, interaction, or decision rule;
- parameter: change one parameter rule, weight, threshold, schedule, or configuration.

Use the later attempts as tested boundaries. Limit each action to one identifiable
algorithmic modification and preserve the target function contract.

Prefix every action with [mechanism] or [parameter]. Return exactly two numbered
action lines, without code or general rationale.
```

代码生成提示中的算子约束为：

```text
Keep the current algorithm's main design idea and implement the requested local
refinement. For [mechanism], modify one identifiable formula, component, or
interaction. For [parameter], modify one identifiable parameter or parameter-
generation rule. Keep the function contract unchanged.
```

### 7.3 `trace_synthesize`：综合两条轨迹

该算子把两条轨迹作为共同设计来源，从各自的改进历史中提取有效原则，并要求建立两种原则之间的明确算法关系。生成的新程序由两条完整轨迹共同指导。

```text
[What to design next]
Compare the two algorithm-development histories as two sources of algorithmic
experience. Identify one useful principle from each history, or a higher-level
relation that explains their successful changes. Propose two alternative actions,
each defining a coherent new algorithm that synthesizes these principles.

Build an explicit operational relationship between the two principles. Use failed
attempts from either history to identify compatible forms of integration. Preserve
the program contract.

Return exactly two numbered action lines. In each line, state the useful principle
from the algorithm to improve, the useful principle from the other history, and
the new relationship or structure to implement.
```

代码生成提示中的算子约束为：

```text
Implement a coherent new algorithm based on the requested synthesis. Preserve the
target function contract. The code must contain an explicit operational interaction
between the two contributing principles.
```

### 7.4 `trace_transfer`：迁移有效思想

该算子使用主轨迹锚点程序作为实现基础，从第二条轨迹中选择一个具有结果证据的思想，并要求说明迁移内容、插入位置及其与主程序机制的关系。

```text
[What to design next]
Keep the algorithm to improve as the implementation base. Read one effective
algorithmic idea from the other design history and adapt that idea to the algorithm
to improve.

Transfer exactly ONE idea in each action. State the idea, its insertion point, and
its interaction with the existing mechanism. Preserve the remaining program
structure and the target function contract. Use a materially different integration
when the current history contains an unsuccessful attempt with a similar idea.

Return exactly two numbered action lines, without code or general rationale.
```

代码生成提示中的算子约束为：

```text
Modify the algorithm to improve by integrating exactly one idea from the other
design history. Preserve its existing structure and untouched mechanisms. Apply
local interface changes required by the integration. Keep the transferred idea
identifiable in the new code and preserve the target function contract.
```

## 8. LLM 上下文与代码生成

### 8.1 上下文内容

LLM 只接收能够影响算法设计判断的信息：

| 信息 | 作用 |
| --- | --- |
| 任务描述、函数契约和优化方向 | 定义问题与有效程序边界 |
| 锚点程序的算法思想和完整代码 | 提供具体实现起点 |
| 形成锚点程序的有序修改历史 | 说明有效思想如何形成 |
| 锚点之后已经尝试的修改及结果 | 提供失败边界和待修复方向 |
| 第二条轨迹的锚点、思想和相关历史 | 支持综合或迁移 |
| 档案中直接相关的少量成功与失败 action | 补充跨轨迹经验 |

轨迹 ID、节点 ID、内部评分、softmax 概率、种群状态、checkpoint 和解析状态保存在系统内部，不进入算法生成提示。

### 8.2 档案经验检索

当前来源轨迹提供主要历史，完整档案只补充少量直接相关的 action 证据。检索规则固定如下：

1. 只使用带有非空 action 且 `outcome` 为提升或退步的真实修改边；
2. 优先选择与当前语义算子相同的历史 action，再选择其它算子记录；
3. 在同一优先级内按 $|\Delta|$ 从大到小排序；
4. 规范化后文本相同的 action 只保留一条；
5. 每次最多提供 2 条成功 action 和 2 条失败 action；
6. 双轨迹算子先完整呈现两条来源轨迹，再补充档案证据。

这些记录只说明“某项修改产生过什么直接结果”。提示词不会根据单次评估自动补充更强的因果解释。

### 8.3 共享上下文模板

````text
[Problem and algorithm interface]
<TASK_DESCRIPTION_AND_FUNCTION_CONTRACT>
<FITNESS_DIRECTION_IN_PLAIN_LANGUAGE>

[Algorithm to improve]
The following algorithm is the concrete starting point for the next design.
It is used because <PLAIN_LANGUAGE_REASON_THIS_ALGORITHM_WAS_SELECTED>.
Its recorded design idea is:
<STARTING_ALGORITHM_IDEA>

```python
<STARTING_ALGORITHM_CODE>
```

[How this algorithm was developed]
The ordered history below explains how this algorithmic idea was reached.
Changes that improved performance provide principles to preserve or develop.
Changes that reduced or maintained performance define tested boundaries.
<HISTORY_LEADING_TO_STARTING_ALGORITHM>

[Later attempts after this algorithm]
<LATER_ATTEMPTS_AFTER_STARTING_ALGORITHM_OR_NONE>

<ANOTHER_DESIGN_HISTORY_SECTION_IF_REQUIRED>

[Relevant lessons from other attempts]
<ONLY_DIRECTLY_RELEVANT_SUCCESS_OR_FAILURE_EVIDENCE>

[What to design next]
<SEMANTIC_OPERATOR_INSTRUCTION>
````

单轨迹算子省略第二条设计历史。双轨迹算子展示第二个锚点程序的完整代码，并解释这条轨迹中的有效原则、相关形成历史和失败边界。

`trace_synthesize` 使用以下第二轨迹模板：

````text
[Another algorithm-development history]
The following trajectory is an equal source of design evidence.
Its selected anchor is useful because <PLAIN_LANGUAGE_RELEVANCE_REASON>.
Its recorded design idea is:
<OTHER_ANCHOR_IDEA>

```python
<OTHER_ANCHOR_CODE>
```

[How the other algorithm was developed]
<HISTORY_LEADING_TO_OTHER_ANCHOR>

[Later attempts from the other anchor]
<LATER_ATTEMPTS_FROM_OTHER_ANCHOR_OR_NONE>

[How to use this history]
Extract one useful principle from each trajectory and define an explicit operational
relationship between them. Treat improvements as evidence to preserve and later
regressions or plateaus as integration boundaries.
````

`trace_transfer` 使用以下第二轨迹模板：

````text
[A tested idea from another algorithm-development history]
The following trajectory contains one idea that may be adapted to the algorithm
to improve. It is relevant because <PLAIN_LANGUAGE_RELEVANCE_REASON>.
The idea to consider is:
<IDEA_TO_ADAPT>

```python
<OTHER_ANCHOR_CODE>
```

[Evidence about this idea]
<HISTORY_AND_RESULTS_RELEVANT_TO_THE_IDEA>

[How to use this history]
Transfer exactly the stated idea. Preserve the main structure of the algorithm to
improve and use this history only as evidence for the selected idea.
````

### 8.4 统一代码生成提示

每条 action 单独生成一个完整程序：

````text
[Problem and algorithm interface]
<TASK_DESCRIPTION_AND_FUNCTION_CONTRACT>
<FITNESS_DIRECTION_IN_PLAIN_LANGUAGE>

[Algorithm to improve]
<STARTING_ALGORITHM_IDEA_AND_COMPLETE_CODE>

[How the algorithm was developed]
<SEMANTIC_HISTORY_OF_THE_STARTING_ALGORITHM>

[Later attempts and known boundaries]
<SEMANTIC_LATER_ATTEMPTS_OR_NONE>

<EXPLAINED_OTHER_ALGORITHM_SECTION_IF_REQUIRED>

[Requested change]
<ACTION_TEXT>

[Implementation requirement]
<OPERATOR_SPECIFIC_CODE_CONSTRAINT>

Implement the requested change as a complete valid implementation. Keep the
function name, arguments, return type, side-effect contract, and output contract
unchanged. Make the requested algorithmic idea identifiable in the code.

Return only:
Idea: <one-sentence description of the resulting algorithm>
Code:
```python
<complete implementation>
```
````

历史叙述按实际顺序说明“进行了什么算法修改、适应度如何变化、该结果在当前优化方向下表示什么”。已有评估只支持把提升步骤视为有利证据、把退步和停滞视为边界，不为单次变化补充未经验证的因果解释。

## 9. 新程序写回

### 9.1 单轨迹算子

设 $\tau_{\le a}$ 是来源轨迹截至锚点程序 $p_a$ 的前缀。新程序 $p_v$ 评估后形成

$$
\tau'=\tau_{\le a}\oplus e_a\oplus p_v.
$$

原轨迹及锚点之后的既有尝试完整保留，新轨迹作为本轮一个后代进入档案和种群候选集合。

### 9.2 双轨迹算子

两条来源轨迹分别选择锚点 $p_a$ 和 $p_b$。同一个 action 生成并评估一次新程序 $p_v$，随后形成

$$
\tau_a'=\tau^{(1)}_{\le a}\oplus e_a\oplus p_v,
\qquad
\tau_b'=\tau^{(2)}_{\le b}\oplus e_b\oplus p_v.
$$

两条新轨迹共享 $p_v$ 的代码、算法思想和适应度，保留不同的来时路。两条来源边分别记录相对各自锚点的 $\Delta$ 和 `outcome`。因此后续搜索可以分别判断“从第一条历史到达该程序”和“从第二条历史到达该程序”的价值。

当两个来源轨迹选择了同一个锚点程序时，程序 DAG 只建立一条 $p_a\rightarrow p_v$ 修改边，两条新轨迹共同引用该边；轨迹个体仍因历史上下文不同而分别保留。

### 9.3 边界条件

| 条件 | 处理方式 |
| --- | --- |
| 当前种群只有一条轨迹 | 只启用 `trace_ideate` 和 `trace_refine` |
| 轨迹只有一个程序 | 末节点与最高分节点相同，直接使用该程序 |
| 剩余评估预算少于 action 数 | 只实现并评估预算允许的 action |
| LLM 未生成可解析 action | 本轮不产生候选程序，不消耗评估预算 |
| 候选代码无法解析 | 不调用评估器，不写入程序或轨迹记录 |
| 程序评估失败 | 消耗一单位评估预算，不进入程序 DAG、档案或种群 |
| 双轨迹选择同一锚点程序 | 共享一条程序修改边，保留两条新轨迹 |
| 候选集合不足种群上限 | 全部保留，不重复填充轨迹 |

## 10. 默认配置

| 机制配置 | v4 第一版 |
| --- | ---: |
| 初始轨迹数 | 10 |
| 当前轨迹种群规模 $M$ | 10 |
| 精英保留数 $E$ | 2 |
| 每轮 action 数 | 2 |
| 轨迹最大程序节点数 | 8 |
| 轨迹价值权重 $(Q,P)$ | $(0.6,0.4)$ |
| 路径结果折扣 $\gamma$ | 0.8 |
| softmax 温度 $T$ | 0.2 |
| 语义算子选择 | 可用算子等概率 |
| 相似候选过滤 | 无 |
| 路线差异 | 仅记录，不参与选择 |
| 随机性 | 父代、锚点、算子和生存抽样共用实验种子 |

这些配置直接写入对应机制。第一版保持固定种群规模、固定温度和等概率算子，便于将实验差异归因到轨迹表示、种群选择和算子语义。每个重复实验显式记录随机种子，使所有 softmax、等概率锚点和算子选择可复现。

## 11. 验证方案

### 11.1 版本级比较

在相同 task、LLM、评估预算和初始程序条件下比较：

1. TraceAAD v3 完整机制；
2. v3 算子 + v4 轨迹种群与 softmax；
3. v4 轨迹种群 + 新语义算子；
4. v4 完整机制。

第二组检验搜索组织变化，第三组检验算子语义变化，第四组检验多来源轨迹写回和完整上下文的综合作用。

### 11.2 核心消融

- 父代选择：softmax、均匀随机和确定性精英选择；
- 轨迹价值：仅 $Q$、$Q+P$、不同 $Q/P$ 权重；
- 种群更新：纯排名截断、精英加 softmax、无精英 softmax；
- 轨迹上下文：只给锚点程序、加入形成历史、再加入锚点后尝试；
- 锚点选择：只用末节点、只用轨迹内最高分节点、两者等概率；
- 双轨迹写回：只接主轨迹、分别接入两条来源轨迹；
- 算子集合：分别移除四个语义算子。

### 11.3 过程指标

最终训练和测试质量之外，还应报告：

- 每轮父代选择概率和实际选择分布；
- 种群终点质量分布与更新率；
- improve、plateau、regress 比例；
- 每个算子的有效评估率和全局最好命中率；
- 末节点与最高分锚点的使用效果；
- 双轨迹后代进入下一轮种群的比例；
- 思想重复率和路线差异，仅作为观察指标；
- 达到给定质量所需的评估次数。

### 11.4 待验证假设

1. 轨迹个体比只包含当前程序的个体提供更有效的下一步生成依据；
2. 种群内百分位质量和结果方向趋势能够跨 task、跨搜索阶段稳定排序轨迹；
3. softmax 父代选择比确定性取最大更好地平衡高价值路线利用和其它路线探索；
4. 精英加 softmax 生存能够维持高质量路线并让新轨迹获得进入种群的机会；
5. 面向算法思想的四个语义算子比 v3 的位置型算子产生更清晰、互补的搜索行为；
6. 双轨迹共同生成并分别写回能够保留更有价值的后续改进历史；
7. 完整轨迹档案能够支持机制分析和后续轨迹信用学习，而不干扰当前种群归一化与父代选择。
