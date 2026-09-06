# TraceAAD V10.3 完整机制设计

TraceAAD V10.3 的搜索机制与 V10.2 完全相同：fitness-only ESS-Boltzmann 父节点概率经 inverse-sqrt 选择次数修正、等概率单算子扩展、轨迹辅助生成。V10.3 只改变生成提示词：删除全部详细行为规定，将算子指令压缩为一句话目标，并以一句算法判断委托代替复杂度治理原则。V10.3 检验的核心命题是：**详细规定生成行为的指令是否必要，还是信任模型的算法判断力即可**。

搜索树、父节点分配、donor 选择、算子采样、上下文截断、断点恢复与评价口径均沿用 V10.2，以下完整陈述，与 V10.2 不同的部分集中在本节与第 5 节。

## 1. 与 V10.2 的差异

全部差异集中在生成提示词（`prompts.py`），共四处：

| 项目 | V10.2 | V10.3 |
| --- | --- | --- |
| 算子指令 | 每个算子 6–10 行详细行为规定（保持核心原则、禁止参数调参式 Pivot、禁止机械拼接等） | 每个算子 2–3 句话，只说明设计目标与可用上下文 |
| Implementation Principle | 独立 Prompt 段落：复杂度治理 + 注释克制原则 | 整段删除，代之以一句 Algorithmic Judgment，嵌入 Improvement Operator 段 |
| 输出契约附加指令 | "Keep the implementation concise. Do not include explanatory comments…" 等注释约束 | 删除，只保留两段式输出格式 |
| Prompt 组装开头 | Task Contract + Implementation Principle | 仅 Task Contract |

Prompt 中不再出现任何关于如何执行算子的规定：不要求 Refine 保持核心原则、不要求 Pivot 避免参数调参、不要求 Fuse 避免机械拼接、不限制注释与代码风格。对生成内容的约束只剩：输出格式（Latest Design Idea 一句话 + 完整函数代码）、任务契约与函数签名、算子指令本身，以及一句委托判断。

## 2. 搜索流程

每一轮扩展执行以下三个步骤：

1. **父节点选择**：先根据全树有效节点的当前真实 fitness 构造 ESS 校准的 Boltzmann 概率 $p_q(n)$，再除以 $\sqrt{c(n)+1}$ 并重新归一化，抽样选出一个父节点 $n_t$。
2. **单算子生成与评价**：根据选中父节点是否存在可用 donor，确定当轮可用算子集合，从中等概率抽取 Refine、Pivot 或 Fuse 中的一个算子，生成一个子代并调用评估器评价。
3. **更新搜索树**：将成功生成且有效评价的子代加入搜索树，随后进入下一轮父节点选择，直到 1000 次评价预算耗尽。

## 3. 搜索树与轨迹

### 3.1 搜索树结构

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

每次有效生成与评价结果都建立一个新的独立节点，即使其代码与已有节点完全相同。

节点保存、评价与最终输出始终使用模型生成的原始完整代码。构建 Prompt 时另行生成仅移除 `#` 注释的代码视图（Python `tokenize` 过滤 `tokenize.COMMENT`，保留字符串字面量与 docstring，连续空行压缩为一个），不回写节点。

Fuse 的 donor 不是父代。子节点的父节点始终是 target，donor 仅作为参考节点记录在 $\mathrm{donor\_id}$ 中，轨迹仅沿 target 父指针向上回溯。

搜索状态保存在 `tree_state.json` 中（含搜索树、RNG 状态、各节点父代入选次数）支持断点恢复；运行事件追加至 `events.jsonl`；结束在 `run_summary.json` 中记录最优结果。

### 3.2 轨迹回溯

轨迹从当前节点沿父指针向上动态回溯生成：

- 回溯时多取 1 代（$8+1$），最多展示 8 代；多出的最老一代仅用于计算最老展示代的趋势
- 每一代记录该节点形成时的 Idea 和 Fitness
- 提升/下降（Improved / Degraded / Unchanged）根据相邻代 fitness 动态计算
- 祖先节点仅展示 Idea 与 Fitness，不呈现其历史代码
- 轨迹仅在组织生成上下文时现场回溯，树结构本身不预存轨迹对象

### 3.3 轨迹呈现格式

祖先轨迹按正向演化顺序排列（最老的祖先为 Step 1，直系父节点为最后一步）：

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

展示范围内最老的一代若无更老祖先（即根节点），仅展示其 Fitness 数值，不标注趋势。

## 4. 父节点分配与扩展算子

### 4.1 初始化

搜索以 $N_0 = 8$ 个独立生成的有效根节点初始化。每个根节点仅根据 Task Contract 生成并评价，消耗 1 次评测预算。若预算耗尽仍不足 8 个有效根节点，则报错中止。

### 4.2 父节点选择

全树所有有效节点构成候选集 $\mathcal A_t$。首先只根据 fitness 构造 ESS 校准的 Boltzmann 概率：

$$
p_q(n) = \frac{\exp\left(\beta_t (f(n) - f_{\max})\right)}{\sum_{m\in\mathcal A_t}\exp\left(\beta_t (f(m) - f_{\max})\right)}
$$

其中减去当前最大值 $f_{\max} = \max_{m\in\mathcal A_t} f(m)$ 以保证数值稳定性。

逆温度 $\beta_t \ge 0$ 根据目标有效样本量（Effective Sample Size, ESS）动态自适应求解：

$$
\operatorname{ESS}(p_q) = \frac{1}{\sum_{n\in\mathcal A_t} p_q(n)^2}, \qquad
E_t = \min\left(N, \max\left(0.10 N, 2\right)\right)
$$

并列最高 fitness 节点数决定非负逆温度下可达到的最低 ESS 下界：

$$
k_{\max} = \left|\left\{n\in\mathcal A_t: f(n) = f_{\max}\right\}\right|, \qquad E_t^* = \max(E_t, k_{\max})
$$

当 $E_t^* < N$ 时，数值求解 $\beta_t \ge 0$ 使实际 $\operatorname{ESS}(p_q)$ 尽可能逼近目标 $E_t^*$；当 $E_t^* = N$ 时取 $\beta_t = 0$（均匀分布）。

令 $c_t(n)$ 为节点 $n$ 在本轮抽样前累计被选为父节点的次数，初始值为 0。最终父节点选择概率为：

$$
p_t(n) = \frac{p_q(n) / \sqrt{c_t(n)+1}}{\sum_{m\in\mathcal A_t} p_q(m) / \sqrt{c_t(m)+1}}
$$

按 $p_t$ 抽样得到父节点后，将该节点的选择次数加 1。父节点一经选中就计数，无论本次输出能否成功解析，也无论是否调用 evaluator。ESS 只校准 fitness-only 分布 $p_q$；$p_t$ 是加入生成机会次数修正后的实际抽样分布。

### 4.3 三个扩展算子

选定父节点后，只选择一个当轮可用的算子组织生成：

- **Refine（深化）**：继续发展当前算法方向，追求最有希望的改进。在树上形成纵向深挖。
- **Pivot（转向）**：探索与当前主机制不同的算法方向。在树上开辟全新分支。
- **Fuse（融合）**：从参考算法中吸收互补思想或机制，发展出更强的连贯算法。实现跨分支的思想重组。

三个算子在 V10.3 中只有目标描述，无行为细则（完整指令文本见第 5.3 节）。

### 4.4 Fuse 的 Donor 选择

对选定的父节点 $n$，候选 donor 集合 $D(n)$ 排除自身、祖先以及所有后代节点：

$$
D(n) = \{r \in H_t: r \ne n, r \notin \operatorname{Ancestors}(n), r \notin \operatorname{Descendants}(n)\}
$$

按 fitness 降序取前 5 名 $D_5(n)$。若 $D_5(n)$ 非空，则 Fuse 是当轮可用算子；当 Fuse 被选中时，从 $D_5(n)$ 中均匀随机选取一个节点作为 donor。若 $D_5(n)$ 为空，则当轮可用算子不包含 Fuse。

### 4.5 等概率单算子扩展

$$
\mathcal O(n) =
\begin{cases}
\{\mathrm{Refine},\mathrm{Pivot},\mathrm{Fuse}\}, & D_5(n) \ne \varnothing,\\
\{\mathrm{Refine},\mathrm{Pivot}\}, & D_5(n) = \varnothing.
\end{cases}
\qquad
o_t \sim \operatorname{Uniform}(\mathcal O(n))
$$

存在可用 donor 时三个算子各 $1/3$；不存在时两个算子各 $1/2$。被选中的算子获得一次生成机会；输出成功解析后送入评估器并消耗 1 次评测预算，解析失败不消耗评测预算。有效程序加入搜索树，随后重新选择父节点。

## 5. 提示词与生成上下文

### 5.1 上下文结构

每次向大模型请求生成时，Prompt 由以下部分构成：

````text
# Task Contract
<任务描述与输入输出接口约定>
Objective: maximize the fitness returned by the evaluator (higher is better).
Design the algorithm using only information available through the target
function interface. ...

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

Use your algorithmic judgment to decide how best to carry out this design move.

Instruction:
<对应算子的一句话指令>

# Output
<两段式输出契约>
````

与 V10.2 相比：Implementation Principle 段落整体不存在；Algorithmic Judgment 以一句话嵌入 Improvement Operator 段的 Operator 行与 Instruction 之间，且仅出现在扩展算子（Init 时无此句）。

### 5.2 输出契约

所有算子（含 Init）遵循统一输出契约：

```text
Respond with exactly two parts and nothing else:
Latest Design Idea: <one sentence stating the actual algorithmic mechanism you introduce,
modify, or combine>
```python
<the complete function implementation>
```
```

对生成内容的全部约束为：两部分输出、Idea 为一句话且须陈述实际引入/修改/合并的算法机制、代码为完整函数实现。V10.2 输出契约后的注释克制指令不再出现。

解析层面要求：响应须含 `Latest Design Idea:` 行与可解析函数；函数须与模板同名（或为唯一函数）且参数签名一致，否则判定为无效输出，不消耗评价预算。连续超过 50 次无效输出视为推理后端故障，中止运行。

### 5.3 算子指令文本

#### Init

```text
Design a novel algorithm for this task from scratch. Propose one clear design
idea and implement it as a complete function that satisfies the Task Contract.
```

#### Refine

```text
Continue developing the current algorithmic direction. Use the historical
trajectory to understand how this idea has evolved, and pursue the improvement
you judge most promising.
```

#### Pivot

```text
Explore a promising algorithmic direction with a different primary mechanism
from the current one. Use the current algorithm and its trajectory as context
for developing the new direction.
```

#### Fuse

```text
Create a stronger coherent algorithm by developing the current algorithm with
complementary ideas or mechanisms from the reference algorithm.
```

### 5.4 上下文截断规则

- Task Contract、当前代码的无注释视图、算子指令以及 Fuse donor 代码的无注释视图始终完整保留，严禁截断。
- 当 Prompt 总字符数超出上限时，从最老的祖先节点开始逐代丢弃历史轨迹。
- 若完全移除轨迹后 Prompt 仍超出限制，则抛出异常中止运行。

上下文预算按 token 计（32768 减去输出预留 16384），以 3.5 字符/token 折算为字符上限。

## 6. 固定参数与记录

| 参数项 | 取值 | 含义 |
| --- | --- | --- |
| 评测预算 $B$ | 1000 | 总真实评估次数 |
| 初始根节点数 $N_0$ | 8 | 搜索起点候选数 |
| 父节点分数 | fitness $f(n)$ | 真实评估指标 |
| 父节点选择修正 | $1 / \sqrt{c(n)+1}$ | 降低已被反复选择节点的后续概率 |
| 目标 ESS 比例 $\rho$ | 0.10 | 竞争主要集中于前 10% 节点 |
| 最小目标 ESS $K_{\min}$ | 2 | 最少保证 2 个有效竞争者 |
| 扩展算子集 | Refine, Pivot, Fuse | 纵向深挖、横向开辟与分支重组 |
| 单轮算子选择 | 当轮可用算子等概率抽样 | 每轮只执行一个算子 |
| 单轮子代数 | 1 | 每轮只生成并评价一个子代 |
| Donor 候选池大小 | 5 | 排除直系后的 Top-5 fitness 节点 |
| 轨迹最大代数 | 8 | 最多保留 8 代历史祖先 |
| 最大上下文 | 32768 tokens | 超限先丢轨迹，再超限中止 |
| 输出上限 | 16384 tokens | 生成预留 |
| 连续无效上限 | 50 | 连续 50 次解析失败判定后端故障 |

生成模型统一 Qwen3.8-27B（thinking 配方：temperature 1.0, top_p 0.95, top_k 20，客户端显式下发）。

运行过程在 `events.jsonl` 中记录每次生成与评估的轻量事实（时间、步数、算子、父节点、donor、状态、fitness、新节点 ID），并在 `tree_state.json` 中保存搜索树和各节点的父代入选次数以支持断点续跑。

Fitness 是唯一质量目标。代码长度与运行时间不进入 fitness；任务执行时限与 Prompt 上下文上限仅作为候选和搜索能否继续执行的可行性约束。

## 7. 设计动机与诊断

V10.2 的详细指令把"怎样做一个好的 Refine/Pivot/Fuse"展开成行为规范（保持核心原则、机制层面变化、避免机械拼接、避免冗余累积、注释克制）。这些规定的隐含假设是：LLM 需要被显式约束才会产生高质量的算法设计行为。V10.3 移除全部规定，只保留算子目标与一句"Use your algorithmic judgment"，检验该假设是否成立：

1. **规定 vs 判断（Ablation C）**：在相同搜索机制与预算下，对比 V10.2（详细规定）与 V10.3（极简委托）的最终 fitness 与收敛速度。若 V10.3 不劣于 V10.2，则详细行为规定属于冗余上下文，其维护成本与偏置风险（如指令间互相冲突、鼓励表面合规的代码装饰）可以消除。
2. **无效输出率**：监控两版本 invalid_output 比例，确认删除注释约束后代码可解析率不显著恶化。
3. **行为多样性**：对比两版本 Refine/Pivot 子代的实际代码改动幅度分布，观察详细规定是否在压缩行为方差（表面合规）而无性能增益。
