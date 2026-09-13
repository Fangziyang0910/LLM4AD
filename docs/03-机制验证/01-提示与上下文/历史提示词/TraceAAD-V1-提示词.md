# TraceAAD-V1 提示词

## 1. 上下文组织逻辑

搜索由两个独立的 Prompt 阶段组成：

### 阶段一：动作提议（Action Proposal）
Prompt 由动作生成模块组装，按顺序包含以下区块：
1. `[Task Description]`：任务描述与目标函数优化说明。
2. `[Algorithm Improvement Context]`：说明轨迹的作用（记录尝试过的修改与观测结果），提供适应度优化方向提示（`fitness_direction_hint`），并展开最近至多 5 步的历史轨迹记录。每步轨迹详细列出：`Step`、`Parent idea`、`Action tried`、`Child idea`、`Fitness` 变化量及 `Outcome` 判定。
3. `[Base Program To Modify]`：当前待修改的基准程序（选中的 Anchor 节点），包含节点 ID、选择原因、Idea、当前 fitness 以及完整 Python 代码。
4. `[Instruction]`：指导模型分析历史中改进、退步或停滞的方向，提出候选修改动作。

### 阶段二：代码实现（Code Implementation）
Prompt 由代码生成模块组装，按顺序包含以下区块：
1. `[Task Description]`：任务描述。
2. `[Current Program]`：当前被修改程序的完整代码与当前 fitness。
3. `[Requested Modification]`：阶段一批准的单个具体修改动作描述。
4. `[Target Function Contract]`：目标函数的空函数头（包含签名与参数）。
5. `[Instruction]`：指导模型将 Requested Modification 落实为完整的 Python 函数实现，保持签名不变。

## 2. 算子逻辑

- **`Action Proposal` 算子**：负责高层机制搜索。指示模型重点分析历史轨迹中哪些动作方向带来了改进、退步或停滞，针对基准程序提出 `action_count` 个互相独立且具体的单机制修改动作（Idea），禁止重复已验证无效的方向，仅输出动作编号列表。
- **`Code Implementation` 算子**：负责确定性代码实现。单向接受批准的修改动作，将其转化为合法的 Python 函数，严格保持函数签名、输入参数和返回类型契约不变。

## 3. 特殊机制说明

- **两阶段调用解耦**：将“动作提议”与“代码实现”严格分为两次独立的 LLM 请求。
- **有界显式轨迹**：历史轨迹严格截断为最近至多 5 步展开，显式展示每步动作对应的适应度数值增减与胜负结果。

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
{formatted_trajectory}

[Base Program To Modify]
Continue from Node p{base_node.id}. Selection reason: {reason}.
Node p{base_node.id}
Idea: {base_node.idea}
Fitness: {formatted_fitness}
Code:
```python
{base_node.code}
```

[Instruction]
Use the trajectory as a record of attempted modifications and outcomes.
Focus on which action directions improved, regressed, or stopped changing fitness.
Propose {action_count} next-step modifications for the base program above:
- each modification must change one main algorithmic mechanism only;
- avoid repeating directions that regressed or stayed unchanged;
- a modification may continue a direction that improved, or try a different direction after saturation.
Each modification must be concrete and implementable. Do not output code or rationale.
Return only a numbered list of exactly {action_count} ideas, one per line.
````

### 轨迹步格式化模板（Trajectory Step Template）
````text
Step {step}: p{node.id} -> p{next_node.id}
Parent idea: {node.idea}
Action tried:
{edge.action}
Child idea: {next_node.idea}
Fitness: {parent_fitness} -> {child_fitness}
Fitness change: {fitness_change}
Outcome: {outcome}
````

### 代码实现 Prompt（Code Implementation Prompt）
````text
[Task Description]
{task_description}

[Current Program]
Node p{current_node.id}
Idea: {current_node.idea}
Fitness: {formatted_fitness}
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
