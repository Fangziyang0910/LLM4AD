# TraceAAD-V4 提示词

## 1. 上下文组织逻辑

由动作提议与代码实现两阶段组装，首次引入实现期历史与约束暴露：

### 阶段一：动作提议（Action Proposal）
按顺序包含：
1. `[Task Description]`：任务描述。
2. `[MDP State History]`：将轨迹形式化为 MDP 设计状态历史（$s_t=(x_t, p_t, h_t, b_t)$）。
3. `[Algorithm Improvement History]`：局部改进历史记录。
4. `[Anchor Program]`：当前锚点程序代码。
5. `[Operator]`：算子名称与指令约束。
6. `[Target Function Contract]`：目标函数契约。
7. `[Instruction]`：动作提议指导指令。

### 阶段二：代码实现（Code Implementation）
按顺序包含：
1. `[Task Description]`：任务说明。
2. `[Current Program]`：当前基准程序代码与 Idea。
3. `[Requested Modification]`：批准的具体修改动作。
4. `[History Available During Implementation]`：在代码实现阶段首次呈现历史轨迹，使代码模型理解历史上下文。
5. `[Operator Constraint]`：代码实现约束。
6. `[Target Function Contract]`：目标函数签名。
7. `[Instruction]`：代码实现指令。

## 2. 算子逻辑

- **`trace_ideate` 算子**：负责全新机制探索。指导模型以退步和停滞作为已测试的边界，禁止重复已有失败方向，探索全新的算法机制。
- **`trace_refine` 算子**：负责局部利用与精炼。指导模型保留已证明有价值的核心思想，针对暴露出的薄弱环节或关键参数进行聚焦微调。

## 3. 特殊机制说明

- **MDP 设计状态历史表示**：将算法搜索形式化为马尔可夫决策过程状态历史，显式定义退步与停滞为探索边界。
- **代码实现期历史暴露（History Available During Implementation）**：打破传统“仅提议阶段看历史”的局限，在第二阶段代码实现 Prompt 中首次注入历史轨迹与算子约束，保证代码编写不偏离历史教训。

## 4. 真实完整的提示词模板

### 算子约束原文（Operator Constraints）
- **`trace_ideate`**:
````text
Propose one genuinely new algorithmic idea grounded in the full history. Use later regressions and plateaus as tested boundaries.
````
- **`trace_refine`**:
````text
Preserve one valuable idea already present in the history and make one focused mechanism or parameter refinement.
````

### 动作生成 Prompt（Action Proposal Prompt）
````text
[Task Description]
{task_description}

[MDP State History]
[Algorithm Improvement History]
The selected trajectory is the design state history. Improvements are useful evidence; plateaus and regressions are tested boundaries, not automatic deletion.
{fitness_direction_hint}
{history}

[Anchor Program]
Continue from Node p{base_node.id}. Selection reason: {base_reason}.
Idea: {base_node.idea}
Code:
```python
{base_node.code}
```

[Operator]
name={operator_name}
Constraint: {operator_constraint}

[Target Function Contract]
{target_function_template}

[Instruction]
Use the complete history as reasoning context for the next action.
Propose exactly {action_count} concrete next-step modifications.
Return only a numbered list of exactly {action_count} action lines, without code or rationale.
````

### 轨迹历史渲染格式（Trajectory History Format）
````text
Step {index}: p{parent.id} -> p{child.id} [selected anchor] [operator={edge.operator}]
  action: {edge.action}
  fitness: {parent_fitness} -> {child_fitness} (delta={edge.delta}, outcome={edge.outcome})
````

### 代码实现 Prompt（Code Implementation Prompt）
````text
[Task Description]
{task_description}

[Current Program]
Node p{current_node.id}
Idea: {current_node.idea}
Code:
```python
{current_node.code}
```

[Requested Modification]
{action}

[History Available During Implementation]
{history}

[Operator Constraint]
{operator_constraint}

[Target Function Contract]
{target_function_template}

[Instruction]
Implement the requested modification as a new complete implementation of the target function.
Keep the function name, arguments, return type, and output contract unchanged.
Include every required import and helper in the returned program.
Return only the new idea and complete code in this format:
Idea: <brief algorithm idea>
Code:
```python
<complete function implementation>
```
Do not include rationale, analysis, tests, or extra text.
````
