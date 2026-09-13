# TraceAAD-V9.14 提示词

## 1. 上下文组织逻辑

由标准纯树区块组装：
1. `[Task]`：任务描述与优化方向说明。
2. `[Current Algorithm]`：当前算法的评估 Fitness 与完整代码。
3. `history_text`：来时路形成路径（渲染父链上最近至多 8 步的 Idea、Diff 代码行增删与结果）。
4. `[Instruction]`：Refine 或 Explore 意图指令。
5. `_output_contract`：严格 Markdown 单代码块契约。

## 2. 算子逻辑

- **`Refine` 算子**：保持核心启发式不变，优化内部局部计算与控制参数。
- **`Explore` 算子**：重构主流程，探索全新算法机理。

## 3. 特殊机制说明

- **纯树极简架构**：树上节点只保留算法代码与其真实评估结果，剥离一切派生代理状态与复杂调度池。

## 4. 真实完整的提示词模板

### 意图指令原文（Intent Instructions）
- **`Intent.REFINE`**:
````text
Continue improving the current algorithm within its existing design. Make one focused modification based on the current algorithm and its improvement history.
````
- **`Intent.EXPLORE`**:
````text
Seek a materially different way to improve the current algorithm. Do not merely tune parameters or make a small local modification. You may replace or substantially restructure an important part of the current design.
````

### 初始程序生成 Prompt（Root Prompt）
````text
[Task]
{task_description}
{fitness_direction_hint}

[Target Function]
{template_function_without_body}
Keep the function name, arguments, return type, and contract unchanged.
Include every required import and helper in the returned program.

[Instruction]
Create one complete, valid, and competitive initial algorithm.
Avoid a placeholder or trivial baseline.
Output one concise Idea and one complete Python program:
Idea: <one sentence, within 500 characters>
Code:
```python
<complete executable implementation>
```
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
Output one concise Idea and one complete Python program:
Idea: <one sentence, within 500 characters>
Code:
```python
<complete executable implementation>
```
````
