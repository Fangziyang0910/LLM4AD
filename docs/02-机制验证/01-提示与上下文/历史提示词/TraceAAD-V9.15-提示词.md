# TraceAAD-V9.15 提示词

## 1. 上下文组织逻辑

包含标准算法生成与异常代码修复两种上下文模式：

### 阶段一：标准生成 Prompt（Generation Prompt）
由任务描述、当前算法、形成历史、意图指令与输出契约构成：
1. `[Task]`：任务描述与优化方向提示。
2. `[Current Algorithm]`：当前算法的评估 Fitness 与完整代码。
3. `history_text`：来时路形成历史。
4. `[Instruction]`：注入 Refine 或 Explore 意图指令。
5. `_output_contract`：输出格式规范（500 字符限制）。

### 阶段二：异常代码修复 Prompt（Repair Prompt）
当候选代码在语法预检或评测时发生异常报错时触发：
1. `[Task]`：任务描述与优化方向提示。
2. `[Evaluated Parent]`（若有）：当前候选代码所基于的父代已评估代码。
3. `[Failed Candidate]`：引发崩溃或报错的候选代码全文，并标明 `Failure during {intent_text}: {error}`（附带经过滤的异常栈帧）。
4. `[Instruction]`：修复指令（要求以最小改动修复报错，同时严格保留原意图 Idea 与目标签名）。
5. 可靠性与输出契约。

## 2. 算子逻辑

- **`Refine` 算子**：在现有设计框架内进行局部聚焦改进。
- **`Explore` 算子**：寻求实质不同的改进方式，重构或替换当前设计的关键组件。
- **`Repair`（异常修复算子）**：在保留原始设计思路与函数签名的前提下，针对具体的语法或运行时报错实施最小补丁修复。

## 3. 特殊机制说明

- **有界代码修复机制（Bounded Repair / EH）**：对执行崩溃或报错的代码，在保留原意图的前提下给出一轮针对性修复机会，有效挽救因微小工程失误导致的新颖机制。
- **轻量语法与目标预检**：在进入昂贵评测前进行 AST 语法解析与顶层函数签名预检，加速错误捕获。

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
