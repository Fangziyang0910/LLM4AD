# TraceAAD-V8 提示词

## 1. 上下文组织逻辑

确立以完整程序树为骨架的单阶段生成上下文，按顺序包含：
1. `[Task Description]`：任务描述与优化方向提示。
2. `[How This Program Was Reached]`：自根至当前节点的祖先形成路径（记录每步的 Outcome、Fitness 演变、LOC 代码行数变化与 Idea）。
3. `[What Has Already Been Tried From This Program]`（若有）：该锚点已有的直接子分支尝试记录与结果。
4. `[Current Program: target node to expand or refine]`：当前待扩展锚点节点的完整代码与适应度。
5. `[Reference Root Branch History]` 与 `[Reference Program: knowledge only, never a parent]`（仅迁移/合成算子可见）：提供参考根分支的形成路径与代码。
6. `[Requested Modification]` 与 `[Operator Constraint]`：注入四类算子之一的详细指令与约束。
7. `[Target Function Contract]`：目标函数空接口模板。
8. `[Instruction]`：输出格式规范。

## 2. 算子逻辑

在程序树上完整继承并应用四类语义算子：
- **`trace_ideate`**：从形成历史出发提出全新算法方向，将已测试的直接子分支作为探索边界，避免重复。
- **`trace_refine`**：对形成历史与直接子分支尝试中暴露的薄弱点或有效机制，进行一次聚焦修正。
- **`trace_synthesize`**：跨分支合成算子。在当前程序与参考分支中各识别一个核心原则，在当前程序中实现功能性融合，禁止机械拼接。
- **`trace_transfer`**：跨分支迁移算子。保持当前程序核心结构，将参考根分支中验证有效的恰好一个思路适配到当前程序。

## 3. 特殊机制说明

- **首个完整树搜索实现**：在树结构上直接进行单阶段生成（一次请求同时输出 Idea 与完整 Python 代码）。
- **来时路与直接尝试分离呈现**：在上下文中分别独立设立 `[How This Program Was Reached]`（自顶向下形成路径）与 `[What Has Already Been Tried From This Program]`（已有下游子分支尝试），为算子提供清晰的探索与利用边界。

## 4. 真实完整的提示词模板

### 四类算子约束原文（Operators）
- **`trace_ideate`**:
````text
Propose a genuinely new algorithmic direction from the formation history. Use the previously tested direct branches as boundaries and do not repeat them.
````
- **`trace_refine`**:
````text
Make one focused correction to a mechanism that showed value or to a weakness exposed by the formation history and direct child attempts.
````
- **`trace_synthesize`**:
````text
Identify one supported principle in each branch and make the two principles interact functionally in the current program. Do not concatenate implementations.
````
- **`trace_transfer`**:
````text
Keep the current program's core structure and adapt exactly one supported idea from the reference root branch to the current branch's tested history.
````

### 形成路径与子分支证据渲染模板（Formation & Child Attempts Format）
````text
# 来时路历史
[How This Program Was Reached]
Step {position}: {edge.outcome}; fitness {parent_fitness} -> {child_fitness}; global breakthrough={yes/no}
  Implemented change: {edge.implemented_idea}
  Code change: {edge.code_change_ratio}; LOC {parent_loc} -> {child_loc}

# 直接子分支尝试历史
[What Has Already Been Tried From This Program]
Branch {position}: {edge.outcome}; fitness {parent_fitness} -> {child_fitness}
  Attempted change: {edge.implemented_idea}
````

### 改进代码生成 Prompt（Improvement Code Prompt）
````text
[Task Description]
{task_description}
{fitness_direction_hint}

{current_history}

[Current Program: target node to expand or refine]
Current fitness: {current_fitness}
```python
{current_node.code}
```

[Reference Root Branch History]
{reference_history}

[Reference Program: knowledge only, never a parent]
Reference fitness: {reference_fitness}
```python
{reference_node.code}
```

[Requested Modification]
{action}

[Operator Constraint]
{operator_constraint}

[Target Function Contract]
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
