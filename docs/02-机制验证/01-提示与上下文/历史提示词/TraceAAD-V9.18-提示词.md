# TraceAAD-V9.18 提示词

## 1. 上下文组织逻辑

由标准区块组装：
1. `[Task]`：任务说明。
2. `[Current Algorithm]`：当前代码与 Fitness。
3. `history_text`：来时路形成路径。
4. `global_facts`（可选）：极短全局数值事实摘要（如全局已知最佳水平与关键性能指标）。
5. `[Instruction]`：Refine 或 Explore 指令。
6. `_reliability_contract`：可靠性准则。
7. `_output_contract`：输出格式规范。

## 2. 算子逻辑

- **`Refine` / `Explore` 算子**：在全局已知极短数值事实参考下生成新算法。

## 3. 特殊机制说明

- **极短全局数值事实（Global-Facts-Lite）**：可选注入至多三条数值事实，让模型了解当前解与全局最优的差距。
- **衰减机会评分（Opportunity Prior）**：调度端为新节点提供衰减式再探索保护。

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

### 改进生成 Prompt（含可选极短全局数值事实）
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
{global_facts}

[Instruction]
{INTENT_INSTRUCTIONS[intent]}
Keep the target function signature and contract unchanged.
Return one complete, self-contained implementation.
Reliability constraints:
Keep execution bounded; do not use unbounded loops or recursion.
Return a value that satisfies the target function contract on every call.
Do not mutate input objects unless the task explicitly permits it.
Output one concise Idea and one complete Python program:
Idea: <one sentence, within 500 characters>
Code:
```python
<complete executable implementation>
```
````
