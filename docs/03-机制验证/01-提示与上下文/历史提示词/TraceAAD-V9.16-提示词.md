# TraceAAD-V9.16 提示词

## 1. 上下文组织逻辑

由标准区块与可靠性契约组装：
1. `[Task]`：任务描述。
2. `[Current Algorithm]`：当前算法的真实评测值与完整代码。
3. `history_text`：来时路形成历史。
4. `[Instruction]`：Refine 或 Explore 核心指令。
5. `_reliability_contract`：可靠性契约（严格限制输入不变性、有界时间复杂度和异常防范）。
6. `_output_contract`：输出格式规范。

## 2. 算子逻辑

- **`Refine` 算子**：在可靠性规则约束下进行局部优化。
- **`Explore` 算子**：在可靠性规则约束下进行机制大改。

## 3. 特殊机制说明

- **显式可靠性契约（Reliability Constraints）**：在 Prompt 中明确规定输入不变性与健壮性契约，从生成源头消灭非法变异。

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

### 异常修复 Prompt（Repair Prompt）
````text
[Task]
{task_description}
{fitness_direction_hint}

[Evaluated Parent]
Fitness: {format_fitness(parent_fitness)}
```python
{parent_code}
```

[Failed Candidate]
```python
{failed_code}
```
Failure during {intent_text}: {error}

[Instruction]
Repair the failed candidate with the smallest change that addresses the reported failure.
Preserve the intended algorithmic idea and the target function signature.
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
