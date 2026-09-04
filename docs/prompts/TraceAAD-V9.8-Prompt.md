# TraceAAD-V9.8 提示词

## 1. 上下文组织逻辑

由标准区块组装：
1. `[Task]`：任务说明与优化方向。
2. `[Current Algorithm]`：当前算法代码与 Fitness。
3. `history_text`：当前分支的来时路（形成路径，至多 8 步）。
4. `[Instruction]`：根据算子意图下发 Refine 或 Explore 专用指令。
5. `_output_contract`：输出单句 Idea 与单一代码块。

## 2. 算子逻辑

- **`Refine` 算子**：深化保持核心原则，在形成路径证据下深挖。
- **`Explore` 算子**：切换核心决策原则，提供可供后续步骤继续细化的高质量替代方向。

## 3. 特殊机制说明

- **Hypothesis 轨迹分段与衰减宽限**：在调度层引入宽限期保护新产生的 Explore 分支，提示词层严格区分 Refine 与 Explore 意图。

## 4. 真实完整的提示词模板

### 意图指令原文（Intent Instructions）
- **`Intent.REFINE`**:
````text
Develop the current algorithmic direction. Preserve its central design principle and make one focused change that improves, completes, or repairs its implementation, using the recorded formation path as evidence.
````
- **`Intent.EXPLORE`**:
````text
Propose one coherent alternative algorithmic direction. Change the central decision principle rather than tuning parameters or adding cosmetic complexity. Return one complete valid implementation that later steps could refine.
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
