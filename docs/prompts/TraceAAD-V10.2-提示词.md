# TraceAAD-V10.2 提示词

## 1. 上下文组织逻辑

由标准化 Markdown 区块统一组装：
1. `# Task Contract`：任务契约与目标函数空接口。
2. `# Implementation Principle`：全局实现原则（优先质量、允许有用的昂贵计算、禁止机械堆叠重叠机制、通过代码而非注释表达算法）。
3. `# Current Algorithm`：当前算法的 Latest Design Idea、Fitness 以及通过 `strip_comments_for_prompt` 清理掉行内注释的可执行代码。
4. `# Historical Design Trajectory`：最近至多 display 代的形成路径；若超长，优先从最老的 generation 开始裁剪；每步标明相对前一步的趋势（`Improved` / `Degraded` / `Unchanged`）。
5. `# Reference Algorithm`（仅 Fuse 时呈现）：清理注释后的 Donor 代码、Fitness 与 Idea。
6. `# Improvement Operator`：当前算子指导指令（Refine / Pivot / Fuse 或 Init）。
7. `# Output`：严格的两段式输出契约。

## 2. 算子逻辑

- **`Refine` 算子**：保持核心机制，替换或精炼低效组件，优化决策精度。
- **`Pivot` 算子**：放弃当前机制核心瓶颈，换用正交新思想，开辟新解空间。
- **`Fuse` 算子**：机制替代型融合（禁止机械堆叠两套完整流程，提取一个具体互补组件替换当前薄弱环节）。

## 3. 特殊机制说明

- **Prompt View 级代码注释剥离（`strip_comments_for_prompt`）**：在为 LLM 构建提示词视图时剥离普通行内注释（通过 tokenize 词法分析），避免历史注释锚定思维；而存储与评测的代码保持原样。
- **机制替代型 Anti-bloat 准则**：在 Implementation Principle 中明确约束禁止机械堆叠两个算法的完整流程，只允许提取互补组件替换原有薄弱环节。
- **最老代优先裁剪（Oldest-generation-first Truncation）**：轨迹超长时从最老的 generation 开始裁剪，保留最贴近当前状态的演变证据。
- **基于选择惩罚的父代分配（Boltzmann Selection Penalty）**：在调度端对被高频访问的父节点施加温度衰减惩罚，平衡深度利用与多样性。

## 4. 真实完整的提示词模板

### 实现原则原文（Implementation Principle）
````text
# Implementation Principle

Prioritize algorithm quality. Complex computation is acceptable when it serves a distinct and useful algorithmic role. Do not remove effective mechanisms merely to shorten the implementation. At the same time, avoid redundant accumulation: when introducing a new mechanism, replace, simplify, or remove existing components that provide overlapping functionality or are superseded by the new design.

Express the algorithm through executable code rather than explanatory comments. Avoid verbose comments and do not preserve comments from previous implementations unless they are essential for correctness.
````

### 算子指令原文（Operator Instructions）
- **`Refine`**:
````text
Continue developing the current algorithmic direction. Preserve the core design
principle of the current algorithm. Use the historical trajectory as evidence to
understand how this direction has developed and what has already been tried,
then make a coherent improvement that better realizes or strengthens the current
idea. Simplification and removal of auxiliary mechanisms are valid improvements
when they better realize the same core design principle. The implementation may
change substantially if needed, but do not replace the core algorithmic principle
with a different one.
````
- **`Pivot`**:
````text
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
````
- **`Fuse`**:
````text
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
Keep the implementation concise. Do not include explanatory comments, reasoning notes,
design discussion, or commented-out alternatives. Use comments only when strictly
necessary to clarify non-obvious implementation constraints.
Do not narrate your reasoning inside the code.
````
