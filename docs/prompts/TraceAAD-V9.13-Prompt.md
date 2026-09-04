# TraceAAD-V9.13 提示词

## 1. 上下文组织逻辑

由标准区块组装：
1. `[Task]`：任务说明。
2. `[Current Algorithm]`：当前算法代码与 Fitness。
3. `history_text`：来时路形成路径。
4. `global_context`（仅在 Explore 时注入）：已访问代理区域前沿质量表（`[Searched Proxy Regions]`），展示各宏簇的最佳适应度与边界特征。
5. `[Instruction]`：根据算子意图下发对应的指令。
6. `_output_contract`：输出契约。

## 2. 算子逻辑

- **`Refine` 算子**：在当前代理区域内进行深度优化。
- **条件化 `Explore` 算子**：结合已访问代理区域的前沿质量，禁止在已知低质量区域盲目重建，强制开辟全新的高质量机制区域。

## 3. 特殊机制说明

- **代理区域前沿宏簇上下文**：在 Explore 提示词中显式注入全局宏观特征分布表。

## 4. 真实完整的提示词模板

### 已搜索区域前沿表模板（Searched Proxy Regions Template）
````text
[Searched Proxy Regions]
Earlier in this search the following proxy mechanism regions were already implemented and evaluated. A candidate that merely rebuilds a region below its recorded level wastes budget.
{frontier_rows}
````

### 意图指令原文（Intent Instructions）
- **`Intent.REFINE`**:
````text
Develop the current algorithmic direction. Preserve its central design principle and make one focused change that improves, completes, or repairs its implementation, using the recorded formation path as evidence.
````
- **`Intent.EXPLORE`**:
````text
Propose one coherent alternative algorithmic direction. Change the central decision principle rather than tuning parameters or adding cosmetic complexity. Return one complete valid implementation that later steps could refine.
````

### 带有全局前沿表的生成 Prompt（Generation Prompt with Global Context）
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

[Searched Proxy Regions]
Earlier in this search the following proxy mechanism regions were already implemented and evaluated. A candidate that merely rebuilds a region below its recorded level wastes budget.
{frontier_rows}

[Instruction]
Propose one coherent alternative algorithmic direction. Change the central decision principle rather than tuning parameters or adding cosmetic complexity. Return one complete valid implementation that later steps could refine.
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
