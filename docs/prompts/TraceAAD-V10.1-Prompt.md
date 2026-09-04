# TraceAAD-V10.1 提示词

## 1. 上下文组织逻辑

由标准化 Markdown 标题区块统一组装：
1. `# Task Contract`：包含任务描述、目标函数完整代码模板、最大化 Fitness 目标声明与接口契约边界。
2. `# Current Algorithm`：当前算法的 Latest Design Idea、Fitness 以及完整代码。
3. `# Historical Design Trajectory`：渲染最近至多 display 代祖先轨迹（每步包含 Step、Idea、Fitness 及相比父代的趋势标记：`Improved` / `Degraded` / `Unchanged`）。
4. `# Reference Algorithm`（仅 Fuse 算子注入）：提供供融合借鉴的 Donor 算法的 Idea、Fitness 与完整代码。
5. `# Improvement Operator`：指明算子类型并注入对应算子的详细指导指令（Refine / Pivot / Fuse 或 Init）。
6. `# Output`：强制性的两段式输出格式契约（Idea 与单个代码块）。

## 2. 算子逻辑

- **`Refine` 算子**：聚焦当前算法核心机理进行有效局部利用与深化。消除冗余计算，微调启发式权重与规则，禁止用不同机制替代核心原则。
- **`Pivot` 算子**：跳出当前局部机制，探索正交或颠覆性新算法，更换核心决策原则或数据结构，强调机制层面的实质差异。
- **`Fuse` 算子**：双亲融合算子。将当前算法作为目标设计，将参考算法作为外部设计知识来源，保留当前算法实质机制并融入参考算法兼容机制，禁止机械复制或代码拼接。
- **`Init` 算子**：从零设计初始算法，满足 Task Contract 规范。

## 3. 特殊机制说明

- **概率性质量父选择与三算子并发派生**：按当前真实质量概率选择父节点，在该节点上并行派生 Refine、Pivot 与 Fuse 三个子代，逐个真实评价。
- **标准因果来时路**：严格按代际顺序渲染形成路径，明确标注每一步的演化趋势（Improved / Degraded / Unchanged）。
- **两段式严格输出契约**：仅允许输出一行 Idea 与一个完整可执行 Python 代码块，消除解析岐义。

## 4. 真实完整的提示词模板

### 算子指令原文（Operator Instructions）
- **`Refine`**:
````text
Continue developing the current algorithmic direction. Preserve the core design
principle of the current algorithm. Use the historical trajectory as evidence to
understand how this direction has developed and what has already been tried,
then make a coherent improvement that better realizes or strengthens the current
idea. The implementation may change substantially if needed, but do not replace
the core algorithmic principle with a different one.
````
- **`Pivot`**:
````text
Develop a materially different algorithmic direction from the current node.
Treat the current code as a usable starting scaffold, but do not assume that its
core design principle should be preserved. Use the historical trajectory as
evidence to understand which directions have already been explored, then
introduce a different primary algorithmic mechanism. The change must be
different at the mechanism level, not merely parameter tuning, coefficient
adjustment, or superficial restructuring.
````
- **`Fuse`**:
````text
Create a coherent algorithm by combining complementary mechanisms from the
current algorithm and the provided reference algorithm. Treat the current
algorithm as the target design and the reference algorithm as an external source
of design knowledge. Preserve a substantive mechanism from the current
algorithm and incorporate a compatible mechanism from the reference algorithm.
Integrate them according to their algorithmic roles rather than mechanically
copying code, averaging formulas, concatenating logic, or replacing the current
algorithm with the reference algorithm.
````
- **`Init`**:
````text
Design a novel algorithm for this task from scratch. Propose one clear design
idea and implement it as a complete function that satisfies the Task Contract.
````

### 历史轨迹渲染格式（Trajectory Format）
````text
# Historical Design Trajectory

Step {step}
Idea: {node.idea}
Fitness: {node.fitness} ({trend})
````

### 统一装配 Prompt 结构（Assembled Prompt）
````text
# Task Contract

{task_description}

The target function to design is:

```python
{template_program}
```

Objective: maximize the fitness returned by the evaluator (higher is better).
Design the algorithm using only information available through the target
function interface. Do not assume access to unavailable state or future
information.

# Current Algorithm
Idea: {current.idea}
Fitness: {current.fitness}

```python
{current.code}
```

# Historical Design Trajectory
{trajectory_steps}

# Reference Algorithm
Idea: {donor.idea}
Fitness: {donor.fitness}

```python
{donor.code}
```

# Improvement Operator
Operator: {operator}

Instruction:
{OPERATOR_INSTRUCTIONS[operator]}

# Output
Respond with exactly two parts and nothing else:
Idea: <one sentence stating the actual algorithmic mechanism you introduce,
modify, or combine>
```python
<the complete function implementation>
```
````
