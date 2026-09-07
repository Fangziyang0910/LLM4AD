# TraceAAD-V9.10 提示词

## 1. 上下文组织逻辑

由标准区块组装：
1. `[Task]`：任务描述。
2. `[Current Algorithm]`：当前算法代码与 Fitness。
3. `history_text`：形成历史。
4. `[Instruction]`：联合臂采样的 Refine 或 Explore 意图指令。
5. `_output_contract`：输出格式约束。

## 2. 算子逻辑

- **`Refine` 算子**：在联合臂高确定性区域深化利用。
- **`Explore` 算子**：在潜力区域执行结构重构。

## 3. 特殊机制说明

- **联合臂后验反馈调度**：维护“锚点 x 意图”的 Beta 后验分布，提示词层保持标准干净的“当前代码 + 形成历史 + Refine/Explore 意图”。

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
