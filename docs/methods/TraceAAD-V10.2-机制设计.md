# TraceAAD V10.2 完整机制设计

TraceAAD V10.2 由 fitness-only ESS-Boltzmann 概率与父节点选择次数共同决定的父节点分配、等概率单算子扩展和轨迹辅助生成组成。机制在有限真实评价预算下优化目标算法。树结构、执行修复与评价口径沿用既有 TraceAAD 平台。

## 1. 搜索流程

每一轮扩展执行以下三个步骤：

1. **父节点选择**：先根据全树有效节点的当前真实 fitness 构造 ESS 校准的 Boltzmann 概率 $p_q(n)$，再除以 $\sqrt{c(n)+1}$ 并重新归一化，抽样选出一个父节点 $n_t$。
2. **单算子生成与评价**：根据选中父节点是否存在可用 donor，确定当轮可用算子集合，从中等概率抽取 Refine、Pivot 或 Fuse 中的一个算子，生成一个子代并调用评估器评价。
3. **更新搜索树**：将成功生成且有效评价的子代加入搜索树，随后进入下一轮父节点选择，直到 1000 次评价预算耗尽。

## 2. 搜索树与轨迹

### 2.1 搜索树结构

搜索树由节点及其单父指针构成：

$$
n = (\mathrm{id},\ \mathrm{code},\ \mathrm{idea},\ \mathrm{fitness},\ \mathrm{evaluation\_id},\ \mathrm{parent\_id},\ \mathrm{operator},\ \mathrm{donor\_id})
$$

- $\mathrm{id}$：节点唯一标识，自增整数
- $\mathrm{code}$：提取出的完整可执行函数代码
- $\mathrm{fitness}$：评估器返回的目标性能指标（数值越大越好）
- $\mathrm{evaluation\_id}$：产生该节点的评价序号（1 到 1000）
- $\mathrm{idea}$：从父节点演化到该节点所采用的设计思想
- $\mathrm{parent\_id}$：直接父节点 ID；根节点为空
- $\mathrm{operator}$：生成该节点的算子（Init、Refine、Pivot 或 Fuse）
- $\mathrm{donor\_id}$：仅 Fuse 记录参考节点 ID，其余为空

有效评估的程序直接加入搜索树。搜索状态保存在 `tree_state.json` 中支持断点恢复，搜索结束在 `run_summary.json` 中记录最优结果。

每次有效生成与评价结果都建立一个新的独立节点，即使其代码与已有节点完全相同。

节点保存、评价与最终输出始终使用模型生成的原始完整代码。构建 Prompt 时另行生成仅移除 `#` 注释的代码视图，不回写节点。

Fuse 的 donor 不是父代。子节点的父节点始终是 target，donor 仅作为参考节点记录在 `donor_id` 中，后续轨迹仅沿 target 父指针向上回溯。

### 2.2 轨迹回溯

轨迹从当前节点沿父指针向上动态回溯生成：

- 仅包含父代与祖先节点（最多追溯 8 代）
- 每一代记录该节点形成时的 Idea 和 Fitness
- 提升/下降（Improved / Degraded / Unchanged）根据相邻代 fitness 动态计算
- 祖先节点仅展示 Idea 与 Fitness，不呈现其历史代码
- 轨迹仅在组织生成上下文时现场回溯，树结构本身不预存轨迹对象

### 2.3 轨迹呈现格式

在 Prompt 中，祖先轨迹按正向演化顺序排列（最老的祖先为 Step 1，直系父节点为最后一步）：

```text
# Historical Design Trajectory

Step 1
Latest Design Idea: <idea>
Fitness: 0.731 (Improved)

Step 2
Latest Design Idea: <idea>
Fitness: 0.746 (Improved)

Step 3
Latest Design Idea: <idea>
Fitness: 0.739 (Degraded)
```

根节点作为起点时，仅展示其 Fitness 数值，不标注趋势。

## 3. 父节点分配与扩展算子

### 3.1 初始化

搜索以 $N_0 = 8$ 个独立生成的有效根节点初始化。每个根节点仅根据 Task Contract 生成并评价，消耗 1 次评测预算。若预算耗尽仍不足 8 个有效根节点，则报错中止。

### 3.2 父节点选择

全树所有有效节点构成候选集 $\mathcal A_t$。首先只根据 fitness 构造 ESS 校准的 Boltzmann 概率：

$$
p_q(n) = \frac{\exp\left(\beta_t (f(n) - f_{\max})\right)}{\sum_{m\in\mathcal A_t}\exp\left(\beta_t (f(m) - f_{\max})\right)}
$$

其中减去当前最大值 $f_{\max} = \max_{m\in\mathcal A_t} f(m)$ 以保证数值稳定性。

逆温度 $\beta_t \ge 0$ 不设跨任务固定常数，而是根据目标有效样本量（Effective Sample Size, ESS）动态自适应求解：

$$
\operatorname{ESS}(p_q) = \frac{1}{\sum_{n\in\mathcal A_t} p_q(n)^2}
$$

$$
E_t = \min\left(N, \max\left(0.10 N, 2\right)\right)
$$

并列最高 fitness 节点数决定非负逆温度下可达到的最低 ESS 下界：

$$
k_{\max} = \left|\left\{n\in\mathcal A_t: f(n) = f_{\max}\right\}\right|, \qquad E_t^* = \max(E_t, k_{\max})
$$

当 $E_t^* < N$ 时，数值求解 $\beta_t \ge 0$ 使实际 $\operatorname{ESS}(p_q)$ 尽可能逼近目标 $E_t^*$；当 $E_t^* = N$ 时取 $\beta_t = 0$。

令 $c_t(n)$ 为节点 $n$ 在本轮抽样前累计被选为父节点的次数，初始值为 0。最终父节点选择概率为：

$$
p_t(n) = \frac{p_q(n) / \sqrt{c_t(n)+1}}{\sum_{m\in\mathcal A_t} p_q(m) / \sqrt{c_t(m)+1}}
$$

按 $p_t$ 抽样得到父节点后，将该节点的选择次数加 1。$c_t(n)$ 表示节点累计获得过多少次生成机会：父节点一经选中就计数，无论本次输出能否成功解析，也无论是否调用 evaluator。Evaluator 预算只统计实际送入 evaluator 的候选。ESS 只校准 fitness-only 分布 $p_q$；$p_t$ 是加入生成机会次数修正后的实际抽样分布。

### 3.3 三个扩展算子

选定父节点后，只选择一个当轮可用的算子组织生成：

- **Refine（深化）**：保持当前算法的核心思想不变，在其基础上寻求更好的实现、调整参数公式或修正局部缺陷。在树上形成纵向深挖。
- **Pivot（转向）**：保留当前代码作为初始脚手架，但放弃其核心假设，引入实质不同的设计机制或决策原则。在树上开辟全新分支。
- **Fuse（融合）**：保留当前节点（target）的核心机制，同时从另一条独立分支（donor）中吸收互补的有益机制，融合成连贯的新算法。实现跨分支的思想重组。

| 算子 | 核心目标 | 与当前思想的关系 | 额外输入 |
| --- | --- | --- | --- |
| Refine | 当前方向还能怎样做得更好 | 保持并深化 | 无 |
| Pivot | 从这个基础能打开什么新方向 | 放弃原假设，引入新机制 | 无 |
| Fuse | 能否吸收外部分支的优势机制 | 保持主体并吸收外部互补机制 | Donor 节点 |

### 3.4 Fuse 的 Donor 选择

对选定的父节点 $n$，候选 donor 集合 $D(n)$ 排除自身、祖先以及所有后代节点：

$$
D(n) = \{r \in H_t: r \ne n, r \notin \operatorname{Ancestors}(n), r \notin \operatorname{Descendants}(n)\}
$$

按 fitness 降序取前 5 名：

$$
D_5(n) = \operatorname{TopK}_{5}(D(n), f)
$$

若 $D_5(n)$ 非空，则 Fuse 是当轮可用算子；当 Fuse 被选中时，从 $D_5(n)$ 中均匀随机选取一个节点作为 donor。若 $D_5(n)$ 为空，则当轮可用算子不包含 Fuse。

### 3.5 等概率单算子扩展

选定父节点 $n$ 后，根据 donor 是否可用构造当轮算子集合：

$$
\mathcal O(n) =
\begin{cases}
\{\mathrm{Refine},\mathrm{Pivot},\mathrm{Fuse}\}, & D_5(n) \ne \varnothing,\\
\{\mathrm{Refine},\mathrm{Pivot}\}, & D_5(n) = \varnothing.
\end{cases}
$$

从 $\mathcal O(n)$ 中均匀随机选择一个算子：

$$
o_t \sim \operatorname{Uniform}(\mathcal O(n)).
$$

存在可用 donor 时，Refine、Pivot 和 Fuse 的选择概率均为 $1/3$；不存在可用 donor 时，Refine 和 Pivot 的选择概率均为 $1/2$。被选中的算子获得一次生成机会；输出成功解析后送入评估器并消耗 1 次评测预算，解析失败不消耗评测预算。有效程序加入搜索树，随后重新选择父节点。

## 4. 提示词与生成上下文

### 4.1 上下文结构

每次向大模型请求生成时，Prompt 由以下部分构成：

````text
# Task Contract
<任务描述与输入输出接口约定>
Objective: maximize the fitness (higher is better)

# Implementation Principle

Prioritize algorithm quality. Complex computation is acceptable when it serves a
distinct and useful algorithmic role. Do not remove effective mechanisms merely
to shorten the implementation. At the same time, avoid redundant accumulation:
when introducing a new mechanism, replace, simplify, or remove existing components
that provide overlapping functionality or are superseded by the new design.

Express the algorithm through executable code rather than explanatory comments.
Avoid verbose comments and do not preserve comments from previous implementations
unless they are essential for correctness.

# Current Algorithm
Latest Design Idea: <当前父节点 idea>
Fitness: <当前父节点 fitness>

```python
<当前父节点移除 # 注释后的完整可执行代码视图>
```

# Historical Design Trajectory
<当前父节点的祖先轨迹，最多 8 代>

# Reference Algorithm (仅 Fuse 提供)
Latest Design Idea: <donor idea>
Fitness: <donor fitness>

```python
<donor 移除 # 注释后的完整可执行代码视图>
```

# Improvement Operator
Operator: <Refine / Pivot / Fuse>

Instruction:
<对应算子的定义指令>
````

### 4.2 统一生成契约

所有算子遵循统一契约：

1. **严格执行指定算子**：Refine 必须保持核心原则，Pivot 必须引入新机制，Fuse 必须融合两方优势。
2. **选择性修改**：按本次设计需要选择、替换、删除或增加机制，不默认累积已有组件。
3. **凝练的 Latest Design Idea 说明**：用一句话说明本次改动的实际算法机制，不写空泛套话或推理流水账。
4. **完整可执行代码**：只输出完整 Python 函数实现，保持函数签名、输入输出契约一致。
5. **只输出 Latest Design Idea + Code**：不夹带额外的思维分析或闲聊。
6. **注释克制**：不在代码中叙述推理、设计讨论或备选方案，仅在非显然的实现约束需要澄清时保留必要注释。

输出契约附加以下指令：

```text
Keep the implementation concise. Do not include explanatory comments, reasoning notes,
design discussion, or commented-out alternatives. Use comments only when strictly
necessary to clarify non-obvious implementation constraints.
Do not narrate your reasoning inside the code.
```

Current 与 Fuse donor 的 Prompt 代码由 Python `tokenize` 生成：仅过滤 `tokenize.COMMENT`，保留字符串字面量和 docstring，并将连续空行压缩为一个空行。过滤结果只用于 Prompt；`Node.code`、evaluator 输入、checkpoint 与最终输出均保留原始代码。

### 4.3 算子指令文本

#### Refine

```text
Continue developing the current algorithmic direction. Preserve the core design
principle of the current algorithm. Use the historical trajectory as evidence to
understand how this direction has developed and what has already been tried,
then make a coherent improvement that better realizes or strengthens the current
idea. Simplification and removal of auxiliary mechanisms are valid improvements
when they better realize the same core design principle. The implementation may
change substantially if needed, but do not replace the core algorithmic principle
with a different one.
```

#### Pivot

```text
Develop a materially different algorithmic direction from the current node.
Treat the current code as a usable starting scaffold, but do not assume that its
core design principle should be preserved. Use the historical trajectory as
evidence to understand how this lineage has developed and avoid reverting to
mechanisms already used along this lineage, then introduce a different primary
algorithmic mechanism. Reuse only implementation components that remain useful
for the new mechanism. Discard legacy mechanisms that are not necessary for the
new direction. The change must be different at the mechanism level, not merely
parameter tuning, coefficient
adjustment, or superficial restructuring.
```

#### Fuse

```text
Create a coherent algorithm by selectively combining complementary mechanisms
from the current algorithm and the reference algorithm. Identify the substantive
mechanism worth retaining from the current algorithm and one compatible mechanism
worth transferring from the reference algorithm. Integrate them according to
their algorithmic roles. The retained target and transferred donor mechanisms
must interact substantively in the resulting decision process. Do not preserve
all mechanisms from either algorithm. When the reference mechanism overlaps with,
supersedes, or makes an existing component unnecessary, replace or remove that
component rather than stacking both. Preserve computationally expensive components
when they play a distinct and useful algorithmic role. Avoid mechanical code
copying, concatenating multiple heuristics, or accumulating several signals that
express essentially the same information.
```

### 4.4 上下文截断规则

代码实现中严格遵守信息完整性：

- Task Contract、当前代码的无注释视图、算子指令以及 Fuse donor 代码的无注释视图始终完整保留，严禁代码截断。
- 当 Prompt 总字符数超出上限时，从最老的祖先节点开始逐代丢弃历史轨迹。
- 若完全移除轨迹后 Prompt 仍超出限制，则抛出异常中止运行。

## 5. 固定参数与记录

| 参数项 | 取值 | 含义 |
| --- | --- | --- |
| 评测预算 $B$ | 1000 | 总真实评估次数 |
| 初始根节点数 $N_0$ | 8 | 搜索起点候选数 |
| 父节点分数 | fitness $f(n)$ | 真实评估指标 |
| 父节点选择修正 | $1 / \sqrt{c(n)+1}$ | 降低已被反复选择节点的后续概率 |
| 父节点访问计数 $c(n)$ | 已分配的生成机会数 | 父节点入选即加 1，与 evaluator 调用次数分开统计 |
| 目标 ESS 比例 $\rho$ | 0.10 | 竞争主要集中于前 10% 节点 |
| 最小目标 ESS $K_{\min}$ | 2 | 最少保证 2 个有效竞争者 |
| 扩展算子集 | Refine, Pivot, Fuse | 纵向深挖、横向开辟与分支重组 |
| 单轮算子选择 | 当轮可用算子等概率抽样 | 每轮只执行一个算子 |
| 单轮子代数 | 1 | 每轮只生成并评价一个子代 |
| Donor 候选池大小 | 5 | 排除直系后的 Top-5 fitness 节点 |
| 轨迹最大代数 | 8 | 最多保留 8 代历史祖先 |

运行过程在 `events.jsonl` 中记录每次生成与评估的轻量事实（时间、步数、算子、父节点、donor、状态、fitness、新节点 ID），并在 `tree_state.json` 中保存搜索树和各节点的父代入选次数以支持断点续跑。

Fitness 是唯一质量目标。代码长度与运行时间不进入 fitness；任务执行时限与 Prompt 上下文上限仅作为候选和搜索能否继续执行的可行性约束。

## 6. 核心消融与诊断

1. **算子有效性识别（Stage P）**：验证 Refine、Pivot 和 Fuse 是否能按指令产生可区分的设计行为与代码改动模式。
2. **轨迹消融（Ablation A）**：对比「有轨迹」与「无轨迹」对后续算法改进质量的净收益。
3. **单算子扩展消融（Ablation B）**：对比「单算子等概率抽样」与「三算子全覆盖」在相同评价预算下的搜索效率差异。
4. **过程诊断**：监控 Pivot 生成子代的重访率与后续突破情况，确保多样性探索能真正转化为全局优势。
