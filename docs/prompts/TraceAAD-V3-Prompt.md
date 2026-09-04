# TraceAAD-V3 提示词

## 1. 上下文组织逻辑

由动作生成与代码实现两阶段组装：

### 阶段一：动作提议（Action Proposal）
Prompt 按顺序包含以下核心区块：
1. `[Task Description]`：任务目标与接口说明。
2. `[Algorithm Improvement History]`：局部历史轨迹展开（最近 3–5 步动作、代码与适应度变化）。
3. `[Cross-Trajectory Action Evidence]`：跨轨迹动作证据池（显式汇总展示跨轨迹的成功动作 `Successful actions` 与失败动作 `Failed actions` 列表）。
4. `[Anchor Program]`：待修改基准程序（节点 ID、Idea 与完整代码）。
5. `[Operator]`：显式声明算子名称与算子约束（`name={operator_name} Constraint: {operator_constraint}`）。
6. `[Target Function Contract]`：目标函数契约模板。
7. `[Instruction]`：指导模型提出下一步具体改动动作。

### 阶段二：代码实现（Code Implementation）
Prompt 按顺序包含：
1. `[Task Description]`：任务描述。
2. `[Current Program]`：基准程序代码与 Idea。
3. `[Requested Modification]`：选定的修改动作。
4. `[Target Function Contract]`：目标函数契约。
5. `[Instruction]`：输出完整 Python 函数。

## 2. 算子逻辑

- **参数化算子约束**：在动作生成阶段通过 `[Operator]` 区块显式注入算子名称与定制约束（`operator_constraint`），如规定重点参考成功动作池、避开失败动作池、或强制跨机制跳跃。
- **动作提议算子**：结合局部历史与跨轨迹证据池，推导能够提升适应度的具体算法改动，输出清晰动作描述。
- **代码实现算子**：单向将自然语言动作落实为 Python 代码，严格遵守函数签名与契约。

## 3. 特殊机制说明

- **跨轨迹动作证据池（Cross-Trajectory Action Evidence）**：在动作提议阶段显式注入跨不同轨迹总结出的成功与失败动作对照列表，提供全局经验参照。
- **参数化算子约束注入**：在 Prompt 中直接设立 `[Operator]` 独立区块传递算子意图与边界约束。

## 4. 真实完整的提示词模板

### 动作生成 Prompt（Action Proposal Prompt）
````text
[Task Description]
{task_description}

[Algorithm Improvement History]
The selected trajectory records the modifications that led to the current program.
{fitness_direction_hint}
{trajectory_history}

[Cross-Trajectory Action Evidence]
{experience_block}

[Operator]
name={operator_name}
Constraint: {operator_constraint}

[Base Program To Modify]
Continue from Node p{base_node.id}. Selection reason: {base_reason}.
Idea: {base_node.idea}
Code:
```python
{base_node.code}
```

[Target Function Contract]
Only evolve:
```python
{target_function_template}
```

[Instruction]
Use the selected trajectory as the main account of how the current program was formed.
Use cross-trajectory actions only as supporting evidence of what worked or failed.
Propose exactly {action_count} concrete next-step modifications.
Each modification must change one main algorithmic idea and follow the operator constraint.
Do not output code or rationale.
Return only a numbered list of exactly {action_count} ideas, one per line.
````

### 跨轨迹动作证据模板（Cross-Trajectory Action Evidence Template）
````text
Successful actions:
- [operator={example.operator}] action={action} delta={delta}
- (none)
Failed actions:
- [operator={example.operator}] action={action} delta={delta}
- (none)
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

[Target Function Contract]
{target_function_template}

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
