# TraceAAD-V9.19 提示词

## 1. 上下文组织逻辑

由行为表征与算子指令组装：
1. `[Task]`：任务说明。
2. `[Current Algorithm]`：当前算法代码、Fitness 与行为表征（Behavior Tag）。
3. `[Crossover Reference Algorithm]`（仅 CROSSOVER）：参考算法的代码、Fitness、行为标签及与当前算法的行为空间距离。
4. `[Formation History]`：形成历史与行为景观摘要。
5. `[Instruction]`：对应动作指令（DEVELOP、EXPLORE 或 CROSSOVER）。
6. 可靠性与超时引导约束。
7. 输出格式约束。

## 2. 算子逻辑

- **`DEVELOP` 算子**：在现有行为簇内深挖细化。
- **`EXPLORE` 算子**：强制探索新行为空间，打破现有行为模式。
- **`CROSSOVER` 算子**：跨分支融合两个具有显著行为距离的互补算法。

## 3. 特殊机制说明

- **在线行为景观控制（BehaveSim）**：在提示词中注入行为表征标签与行为空间距离。
- **代码注释干扰消融对比**：验证了代码注释累积对模型认知的负面锚定效应。

## 4. 真实完整的提示词模板

### 动作指令原文（Action Instructions）
- **`Action.DEVELOP`**:
````text
Continue developing the current algorithm. Preserve its main framework and make one coherent modification with a clear performance rationale. Use the formation path to identify what has already worked, what has been revisited, and which recent direction deserves refinement. You may improve a local rule or one substantive mechanism, but avoid redesigning unrelated parts.
````
- **`Action.EXPLORE`**:
````text
Propose a materially different algorithmic direction for the task. Change the main decision logic rather than making a cosmetic variation. Use the formation path to avoid repeating behavior that has already been revisited without improvement, while keeping the new design coherent and executable.
````
- **`Action.CROSSOVER`**:
````text
Combine the current algorithm with the provided reference algorithm. Identify one useful mechanism or decision rule in the reference and integrate it coherently into the current framework. Preserve the strong parts of the current algorithm, avoid copying weaknesses from the reference, and do not concatenate two complete codes mechanically.
````

### 改进生成 Prompt（Generation Prompt）
````text
[Task]
{task_description}
Fitness is this task's score and higher is better.

[Current Algorithm]
Fitness: {format_fitness(fitness)}
Behavior tag: {behavior_tag}
```python
{code}
```

[Crossover Reference Algorithm]
Fitness: {reference_fitness}
Behavior tag: {reference_behavior}
Behavior distance from current algorithm: {reference_distance}
```python
{reference_code}
```

[Formation History]
{history_text}

[Instruction]
{ACTION_INSTRUCTIONS[action]}
Keep the target function signature unchanged.
Return one complete, self-contained implementation.
Reliability constraints:
Keep execution bounded; do not use unbounded loops or recursion.
Return a value that satisfies the target function contract on every call.
Do not mutate input objects unless the task explicitly permits it.
Output only one optional short Idea and one mandatory full Code block:
Idea: <optional semantic label, at most 500 characters>
Code:
```python
<complete executable implementation>
```
Do not output reasoning, evidence analysis, an operator label, or a patch.
````
