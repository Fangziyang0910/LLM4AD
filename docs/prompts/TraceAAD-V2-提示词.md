# TraceAAD-V2 提示词

## 1. 上下文组织逻辑

采用动作提议与代码实现两阶段调用：

### 阶段一：动作提议（Action Proposal）
Prompt 按顺序组装多层上下文信息：
1. `[Task Description]`：任务描述文本。
2. `[Algorithm Improvement Context]`：优化方向提示与因果叙事（`_causal_narrative`），渲染最近步数的因果细节，每步标明 `[op=... mech=...]`、`action`、适应度增量 Delta、`outcome` 以及泛化信号。
3. `[Distilled Patterns]`：由模式记忆库抽取的跨轨迹重复出现的机制（Recurring mechanisms）以及带有 `ANTI` / `OK` 标签的经验教训（Recent lessons）。
4. `[Contrast Feedback]`：对比反馈，展现近期效果最好（best）与最差（worst）的机制标签、fitness 与 Idea 概要。
5. `[Operator]`：显式展示当前算子名称、角色定义与约束（`name={operator_name} role={operator_role}` 及 `Constraint: {operator_constraint}`）。
6. `[Base Program To Modify]`：待修改的基础程序（Anchor 节点），包含节点 ID、选择原因、Idea 及代码。
7. `[Target Function Contract]`：目标函数模板契约。
8. `[Instruction]`：指示模型综合轨迹、模式与对比证据，提出 `action_count` 个机制修改动作。

### 阶段二：代码实现（Code Implementation）
Prompt 组装实现上下文：
1. `[Task Description]`：任务描述。
2. `[Current Program]`：基准程序（包含节点 ID、Idea 与完整代码）。
3. `[Requested Modification]`：批准的具体修改动作描述。
4. `[Target Function Contract]`：目标函数签名与模板。
5. `[Instruction]`：指导模型将请求动作落实为完整函数代码。

## 2. 算子逻辑

- **`Explore` 算子**：角色定义为探索全新机制，约束为禁止在已有机制上做局部微调，强制跳出局部最优。
- **`Exploit` 算子**：角色定义为在当前表现优异的机制上做深化利用，约束为保持主干逻辑不变，优化参数或局部子过程。
- **模式感知动作生成**：指导模型吸收模式记忆库中的正面经验（`OK` 标签），严格避开负面经验（`ANTI` 标签）。

## 3. 特殊机制说明

- **模式记忆库（Pattern Memory）**：在线归纳高频重复出现的机制模式，打上正负经验标签并在 Prompt 中显式呈现。
- **对比反馈（Contrast Feedback）**：直接在提示词中对比呈现近期最成功与最失败的算法机制。
- **细粒度因果叙事**：记录每一步算子类型、机制标签、适应度增量以及跨实例的泛化表现。

## 4. 真实完整的提示词模板

### 初始程序生成 Prompt（Initial Prompt）
````text
{task_description}

Generate a complete implementation for the target Python function. {diversity_hint}
Keep the function name, arguments, return type, and output contract unchanged.

Output format:
Idea: <brief algorithm idea>
Code:
```python
{target_function_template}
```
````

### 动作生成 Prompt（Action Proposal Prompt）
````text
[Task Description]
{task_description}

[Algorithm Improvement Context]
The selected trajectory records attempted modifications and observed outcomes.
{fitness_direction_hint}
{causal_narrative}

[Distilled Patterns]
{patterns_block}

[Contrast Feedback]
{contrast_block}

[Operator]
name={operator_name} role={operator_role}
Constraint: {operator_constraint}

[Base Program To Modify]
Continue from Node p{base_node.id}. Selection reason: {base_reason}.
Idea: {base_node.idea}
Code:
```python
{base_node.code}
```

[Target Function Contract]
{target_function_template}

[Instruction]
Use the trajectory, patterns, and contrast as a record of what worked and what did not.
Propose {action_count} next-step modifications for the base program above:
- each modification must change one main algorithmic mechanism only;
- follow the operator constraint;
- avoid repeating directions that regressed or stayed unchanged.
Each modification must be concrete and implementable. Do not output code or rationale.
Return only a numbered list of exactly {action_count} ideas, one per line.
````

### 因果叙事与模式渲染模板（Narrative & Patterns Template）
````text
# 因果叙事步渲染
Step {i}: p{parent.id} -> p{child.id}  [op={edge.operator} mech={edge.mechanism_tag}]
  action: {edge.action}
  fitness: {parent_fitness} -> {child_fitness} (Δ={delta}, outcome={edge.outcome}, generalization={edge.generalization_signal})

# 蒸馏模式渲染
Recurring mechanisms (cross-trajectory evidence):
  - {mechanism_tag}: generalization={score} support={support_count}
Recent lessons:
  - ({ANTI/OK}, {mechanism_tag}) {lesson_text}

# 对比反馈渲染
Recent best: mech={best.mechanism_tag} fitness={best.fitness} idea='{best.idea}'
Recent worst: mech={worst.mechanism_tag} fitness={worst.fitness} idea='{worst.idea}'
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
