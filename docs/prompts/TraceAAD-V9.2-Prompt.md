# TraceAAD-V9.2 提示词

## 1. 上下文组织逻辑

规范局部历史窗口组装：
1. `[Task]`：任务描述与优化方向提示。
2. `[Initial Route Strategy]`（可选）：初始分配给该路线的高层宏观策略。
3. `[Local Trajectory Context]`：规范局部历史窗口文本，包含来时路形成步与该锚点已有的下游尝试。
4. `[Current Executable Anchor]`：当前锚点的 Fitness 与可执行代码。
5. `[Instruction]`：要求提出一个机制级别的 Idea 并实现为完整代码，允许用实质不同的实现重试已测思路。
6. 代码表示与输出契约：严格去除代码注释（纯可执行实现），输出 `Idea: ...` 与 `Code: ...`。

## 2. 算子逻辑

- **锚点自适应意图算子**：指导模型权衡父代形成链与该锚点已有子代尝试的胜负证据，决定是沿既有成功方向继续深挖，还是转向新逻辑。

## 3. 特殊机制说明

- **规范局部历史窗口**：锚点作为选择与信用单位，每个锚点拥有确定性的局部历史窗口（父链形成事件 + 下游直接尝试）。

## 4. 真实完整的提示词模板

### 初始根程序生成 Prompt（Root Prompt）
````text
[Task]
{task_description}
{fitness_direction_hint}

[Target Function]
{target_function_template}
Keep the function name, arguments, return type, and contract unchanged.
Include every required import and helper in the returned program.

[Assigned Initial Strategy]
{strategy}

[Instruction]
Implement the assigned strategy as one complete, valid, and competitive initial algorithm.
Avoid a placeholder or trivial baseline.
Keep the target function signature and contract unchanged.
Keep the code free of internal docstrings, comments, explanation, or trailing analysis; keep only executable source.
Output only one Idea and one Code block:
Idea: <one concise sentence describing the implemented algorithm, at most 300 characters>
Code:
```python
<complete executable implementation>
```
````

### 锚点生成 Prompt（Generation Prompt）
````text
[Task]
{task_description}
{fitness_direction_hint}

[Initial Route Strategy]
{initial_strategy}

{window_text}

[Current Executable Anchor]
Anchor fitness: {anchor_fitness}
```python
{anchor.code}
```

[Instruction]
Generate exactly one coherent next Idea + Code from the current executable anchor, using the local trajectory evidence as guidance.
This is one complete search decision: propose and implement only the single next change shown in your output.
Treat each recorded result as evidence about that specific implementation; it does not by itself prove that the broader idea is good or bad.
If revisiting a tested idea, implement a materially different realization and state the difference in the Idea.
Keep the current target function signature and contract unchanged.
Return one complete, self-contained implementation.
Keep the code free of internal docstrings, comments, explanation, or trailing analysis; keep only executable source.
Output only one Idea and one Code block:
Idea: <one concise sentence describing the implemented algorithm, at most 300 characters>
Code:
```python
<complete executable implementation>
```
````
