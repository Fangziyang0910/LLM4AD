# TraceAAD-V10.3 提示词

V10.3 相对 V10.2 的全部差异：删除 Implementation Principle 段、三算子指令压缩为一句话目标、输出契约删去注释克制指令、以一句 Algorithmic Judgment 嵌入 Improvement Operator 段。不再对生成行为作任何详细规定。

## 1. 上下文组织逻辑

由标准化 Markdown 区块统一组装：
1. `# Task Contract`：任务契约与目标函数空接口。
2. `# Current Algorithm`：当前算法的 Latest Design Idea、Fitness 以及通过 `strip_comments_for_prompt` 清理掉行内注释的可执行代码。
3. `# Historical Design Trajectory`：最近至多 display 代的形成路径；若超长，优先从最老的 generation 开始裁剪；每步标明相对前一步的趋势（`Improved` / `Degraded` / `Unchanged`）。
4. `# Reference Algorithm`（仅 Fuse 时呈现）：清理注释后的 Donor 代码、Fitness 与 Idea。
5. `# Improvement Operator`：当前算子名称、Algorithmic Judgment（非 Init 时）与一句话算子指令。
6. `# Output`：严格的两段式输出契约。

## 2. 算子逻辑

- **`Refine` 算子**：继续发展当前方向，借助轨迹理解演化历程，追求最有希望的改进。
- **`Pivot` 算子**：探索与当前主机制不同的方向，以当前算法与轨迹作为发展新方向的上下文。
- **`Fuse` 算子**：从参考算法吸收互补思想或机制，发展出更强的连贯算法。

三算子均只描述设计目标，无行为细则（无"保持核心原则""禁止参数调参""禁止机械拼接"等规定）。

## 3. 特殊机制说明

- **算法判断委托（Algorithmic Judgment）**：以一句 "Use your algorithmic judgment to decide how best to carry out this design move." 代替 V10.2 的全部行为规定与复杂度治理原则；仅扩展算子出现，Init 无此句。
- **Prompt View 级代码注释剥离（`strip_comments_for_prompt`）**：在为 LLM 构建提示词视图时剥离普通行内注释（通过 tokenize 词法分析），避免历史注释锚定思维；而存储与评测的代码保持原样。
- **最老代优先裁剪（Oldest-generation-first Truncation）**：轨迹超长时从最老的 generation 开始裁剪，保留最贴近当前状态的演变证据。
- **基于选择惩罚的父代分配（Boltzmann Selection Penalty）**：在调度端对被高频访问的父节点施加温度衰减惩罚，平衡深度利用与多样性。

## 4. 真实完整的提示词模板

### 算法判断原文（Algorithmic Judgment）
````text
Use your algorithmic judgment to decide how best to carry out this design move.
````

### Improvement Operator 区块组装示例（以 Refine 为例）
````text
# Improvement Operator
Operator: Refine

Use your algorithmic judgment to decide how best to carry out this design move.

Instruction:
Continue developing the current algorithmic direction. Use the historical
trajectory to understand how this idea has evolved, and pursue the improvement
you judge most promising.
````

### 算子指令原文（Operator Instructions）
- **`Refine`**:
````text
Continue developing the current algorithmic direction. Use the historical
trajectory to understand how this idea has evolved, and pursue the improvement
you judge most promising.
````
- **`Pivot`**:
````text
Explore a promising algorithmic direction with a different primary mechanism
from the current one. Use the current algorithm and its trajectory as context
for developing the new direction.
````
- **`Fuse`**:
````text
Create a stronger coherent algorithm by developing the current algorithm with
complementary ideas or mechanisms from the reference algorithm.
````
- **`Init`**:
````text
Design a novel algorithm for this task from scratch. Propose one clear design
idea and implement it as a complete function that satisfies the Task Contract.
````

### 任务契约模板（Task Contract Template）
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
````

### 输出契约原文（Output Contract）
````text
# Output
Respond with exactly two parts and nothing else:
Latest Design Idea: <one sentence stating the actual algorithmic mechanism you introduce,
modify, or combine>
```python
<the complete function implementation>
```
````
