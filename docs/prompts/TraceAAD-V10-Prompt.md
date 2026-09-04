# TraceAAD-V10 提示词

## 1. 上下文组织逻辑

由标准模块组装：
1. `[Task]`：任务说明与优化方向。
2. `[Current Algorithm]`：当前算法 Fitness 与完整代码（除 Restart 以外）。
3. `[Formation Path]`：来时路形成路径（至多 8 步）。
4. `[Reference Algorithm]`（仅 Transfer 算子可见）：供借鉴的 Donor 代码、Fitness、Idea 与参考形成路径。
5. `[Verified Improvement Cards]`（仅 Restart 算子可见）：历史验证过的优质改进卡片（仅文字与质量，不给代码）。
6. `[Semantic Inconsistency / Failure]`（仅 SemanticRepair 可见）：评判器诊断出的具体失败或不一致。
7. `[Instruction]`：注入五类算子对应的指导说明。
8. 可靠性与输出契约：严格限制输出为精简 Idea 与代码。

## 2. 算子逻辑

- **`Develop` 算子**：深化当前算法主机制，优化局部子过程与控制参数。
- **`Pivot` 算子**：在当前谱系内更换核心计算范式，打破瓶颈。
- **`Transfer` 算子**：跨谱系借鉴 Donor 算法的优势机制。
- **`Restart` 算子**：基于历史验证卡片重新构思算法。
- **`SemanticRepair` 算子**：针对评判器反馈的语义逻辑矛盾进行专项修复。

## 3. 特殊机制说明

- **生成与评判隔离**：生成模型与评判模型隔离，生成端严格看不见评判端的内部状态。
- **五类语义算子体系**：覆盖深化、转向、迁移、重启与语义修复完整动作谱系。

## 4. 真实完整的提示词模板

### 五类算子指令原文（Operator Instructions）
- **`develop`**:
````text
**Develop.** Preserve the current algorithm's core design hypothesis. Improve it coherently. You may tune parameters, strengthen a local or deep mechanism, simplify harmful details, or restructure supporting logic, but do not replace the main algorithmic idea.
````
- **`pivot`**:
````text
**Pivot.** Use the current algorithm as a starting point, but do not assume its core design hypothesis is correct. Replace or substantially redesign one central decision mechanism and create a coherent alternative direction.
````
- **`transfer`**:
````text
**Transfer.** Preserve the useful structure of the source algorithm. Identify one mechanism in the donor that is supported by its evaluator history and integrate that mechanism coherently. Do not copy the entire donor blindly.
````
- **`restart`**:
````text
**Restart.** Design a novel algorithm for the task from scratch. Use the verified improvement cards as evidence of what mechanisms have shown value, but do not copy any single predecessor implementation. Produce a fresh, self-contained design.
````
- **`semantic_repair`**:
````text
**Semantic Repair.** The previous attempt failed to evaluate or contained an identified behavioral inconsistency. Repair the implementation so that it runs reliably and matches its intended design idea. Preserve the core mechanism and keep the change as focused as possible.
````

### 改进生成 Prompt（Generation Prompt）
````text
[Task]
{task_description}
Fitness is this task's score and higher is better.

[Current Algorithm]
Fitness: {format_fitness(fitness)}
```python
{code}
```

[Formation Path]
{formation_lines}

[Instruction]
{OPERATOR_INSTRUCTIONS[operator]}
Keep the target function signature unchanged.
Return one complete, self-contained implementation.
Reliability constraints:
Keep execution bounded; do not use unbounded loops or recursion.
Return a value that satisfies the target function contract on every call.
Do not mutate input objects unless the task explicitly permits it.
Output only:
Idea: <concise algorithm idea, at most 500 characters>
Code:
```python
<complete executable implementation>
```
Do not output docstrings, comments, reasoning, or trailing analysis.
````
