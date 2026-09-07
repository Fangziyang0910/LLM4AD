# TraceAAD-V9 提示词

## 1. 上下文组织逻辑

纯树简约基准，统一祖先链上下文与单阶段代码生成：
1. `[Task Description]`：任务说明与优化方向提示。
2. `[Current Program: target node to expand or refine]`：当前待扩展节点代码与适应度。
3. `[How This Program Was Reached]`：自根至当前节点的祖先形成路径（每步包含 Outcome、Fitness 演变、代码行变化与 Idea）。
4. `[What Has Already Been Tried From This Program]`（若有）：该节点已有的直接子分支尝试记录。
5. `[Reference Root Branch History]` 与 `[Reference Program: knowledge only, never a parent]`（可选）：参考分支的形成历史与代码。
6. `[Improvement Direction]`：注入四类语义算子之一的指令与约束。
7. `[Target Function Contract]`：目标函数空接口。
8. `[Instruction]`：输出格式规范。

## 2. 算子逻辑

在程序树骨架上继承正统四类语义算子：
- **`trace_ideate`**：探索全新算法方向，利用测试过的历史作为边界且不重复。
- **`trace_refine`**：对已有有效机制或薄弱点进行聚焦修正与深化。
- **`trace_synthesize`**：跨分支融合两个分支的优势原则，在当前程序中形成功能性互动。
- **`trace_transfer`**：跨分支借鉴参考根分支中的有效机制，保持当前程序主体结构不变。

## 3. 特殊机制说明

- **纯树简约基准**：删除固定深度限制与复杂的复合信用回传，统一祖先链三元组（Idea-Diff-Outcome）表示，恢复强劲纯树搜索性能。
- **模板与 V8 同源**：保持生成端协议的高度一致性，集中检验树搜索选择与扩展逻辑。

## 4. 真实完整的提示词模板

### 初始程序生成 Prompt（Initial Prompt）
````text
{task_description}
{fitness_direction_hint}

Generate a complete implementation for the target Python function. {diversity_hint}
Keep the function name, arguments, return type, and contract unchanged.
Imports from the task template remain available. You may add small top-level helper functions when they clarify the implementation.

Output format:
Idea: <one sentence describing the implemented algorithm, no more than 300 characters>
Code:
```python
{target_function_template}
```
````

### 算子约束原文（Operators）
- **`TraceIdeateOp` (`trace_ideate`)**:
````text
Propose a genuinely new algorithmic direction from the formation history. Use the previously tested direct branches as boundaries and do not repeat them.
````
- **`TraceRefineOp` (`trace_refine`)**:
````text
Make one focused correction to a mechanism that showed value or to a weakness exposed by the formation history and direct child attempts.
````
- **`TraceSynthesizeOp` (`trace_synthesize`)**:
````text
Identify one supported principle in each branch and make the two principles interact functionally in the current program. Do not concatenate implementations.
````
- **`TraceTransferOp` (`trace_transfer`)**:
````text
Keep the current program's core structure and adapt exactly one supported idea from the reference root branch to the current branch's tested history.
````

### 代码生成 Prompt（Code Generation Prompt）
````text
[Task]
{task_description}
{fitness_direction_hint}

{current_history}

[Current Program]
Current fitness: {current_fitness}
```python
{current_node.code}
```

[Reference Root Branch History]
{reference_history}

[Reference Program]
Reference fitness: {reference_fitness}
```python
{reference_node.code}
```

[Improvement Direction]
{operator_constraint}

[Target Function]
{target_function_template}

[Instruction]
Generate candidate {candidate_index} of {candidate_count} by directly improving the current program.
Use the histories as evidence and do not repeat tested changes.
Choose one concrete algorithmic change and implement it completely.
When a reference is shown, adapt it only according to the improvement direction.
Keep the target function signature and contract unchanged.
Return exactly one complete implementation.
Imports from the current program remain available; small top-level helpers are allowed.
Output only:
Idea: <one sentence describing the implemented change>
Code:
```python
<complete function implementation>
```
````
