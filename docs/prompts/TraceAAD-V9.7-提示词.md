# TraceAAD-V9.7 提示词

## 1. 上下文组织逻辑

由标准区块组装：
1. `[Task]`：任务描述与优化方向。
2. `[Current Algorithm]`：当前算法代码与 Fitness。
3. `history_text`：仅展示父代形成路径（每步包含 Idea、Change 压缩行、Result 与 Fitness 变化），严格剔除直接子代尝试记录。
4. `[Instruction]`：根据下发意图注入明确的 Refine 或 Explore 具体指示。
5. `_output_contract`：严格限制输出格式为单行 Idea 与单个完整代码块。

## 2. 算子逻辑

- **`Refine` 算子**：保持当前算法的核心原则与基本框架，消除低效计算，深化启发式逻辑。
- **`Explore` 算子**：改变核心决策原则与数据结构，开辟正交的替代算法方向。

## 3. 特殊机制说明

- **正统来时路规范**：彻底移除直接子代尝试（Direct Attempts），上下文仅保留自根至父的形成路径。

## 4. 真实完整的提示词模板

### 两类核心意图指令原文（Intent Instructions）
- **`Intent.REFINE`**:
````text
Continue improving the current algorithm within its existing design. Make one focused modification based on the current algorithm and its improvement history.
````
- **`Intent.EXPLORE`**:
````text
Seek a materially different way to improve the current algorithm. Do not merely tune parameters or make a small local modification. You may replace or substantially restructure an important part of the current design.
````

### 形成路径渲染模板（Parent Path History Format）
````text
[Recent Algorithm Improvement History]

[History {index}] Formation step
Idea: {attempt.idea}
Change: {compact_change}
Result: {attempt.outcome.value}
Fitness: {parent_fitness} -> {child_fitness}
````

### 初始程序生成 Prompt（Root Prompt）
````text
[Task]
{task_description}
{fitness_direction_hint}

[Target Function]
{target_function_template}
Keep the function name, arguments, return type, and contract unchanged.
Include every required import and helper in the returned program.

[Instruction]
Create one complete, valid, and competitive initial algorithm.
Avoid a placeholder or trivial baseline.
Output only one optional short Idea and one mandatory full Code block:
Idea: <optional semantic label, at most 300 characters>
Code:
```python
<complete executable implementation>
```
Do not output reasoning, evidence analysis, an operator label, or a patch.
````

### 改进生成 Prompt（Generation Prompt）
````text
[Task]
{task_description}
{fitness_direction_hint}

[Current Algorithm]
Fitness: {format_fitness(fitness)}
```python
{code}
```

{history_text}

[Instruction]
{INTENT_INSTRUCTIONS[intent]}
Keep the target function signature and contract unchanged.
Return one complete, self-contained implementation.
Output only one optional short Idea and one mandatory full Code block:
Idea: <optional semantic label, at most 300 characters>
Code:
```python
<complete executable implementation>
```
Do not output reasoning, evidence analysis, an operator label, or a patch.
````
