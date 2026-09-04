# TraceAAD-V9.1 提示词

## 1. 上下文组织逻辑

由任务上下文、分支历史与多语义算子组装：
1. `[Task]`：任务描述与优化方向提示。
2. `{current_history}`：当前分支形成历史与直接尝试记录。
3. `[Current Program]`：当前待扩展程序的完整代码与适应度。
4. `[Reference Root Branch History]` 与 `[Reference Program]`（可选）：参考根分支的形成历史与完整代码。
5. `[Improvement Direction]`：注入四类算子约束 `{operator_constraint}`。
6. `[Target Function]`：目标函数模板契约 `{target_function_template}`。
7. `[Instruction]`：生成指导说明（含候选批次指示与严格单代码块契约）。

## 2. 算子逻辑

- **`Ideate` 算子**：提出全新算法思想，跳出现有解结构。
- **`Refine` 算子**：在当前算法基础上进行参数微调与局部提炼。
- **`Synthesize` 算子**：综合多重思想，重构核心计算流。
- **`Transfer` 算子**：跨分支优势机制迁移，将参考分支中的有效组件嫁接到当前程序中。

## 3. 特殊机制说明

- **剥离传统交叉算子**：无损剥离传统交叉机制，全面转向由四大纯语义意图驱动生成。
- **批次候选生成指令与迁移边界**：指令中显式下发 `Generate candidate {candidate_index} of {candidate_count}`；且明确规定参考分支仅作为知识迁移边界，绝非第二结构父代。

## 4. 真实完整的提示词模板

### 四类算子约束原文（Operators）
- **`TraceIdeateOp`**:
````text
Propose a genuinely new algorithmic direction from the formation history. Use the previously tested direct branches as boundaries and do not repeat them.
````
- **`TraceRefineOp`**:
````text
Make one focused correction to a mechanism that showed value or to a weakness exposed by the formation history and direct child attempts.
````
- **`TraceSynthesizeOp`**:
````text
Identify one supported principle in each branch and make the two principles interact functionally in the current program. Do not concatenate implementations.
````
- **`TraceTransferOp`**:
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
