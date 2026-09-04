# TraceAAD-V9.6 提示词

## 1. 上下文组织逻辑

由标准区块组装：
1. `[Task]`：任务描述与优化方向提示。
2. `[Current Algorithm]`：当前算法代码与 Fitness。
3. `history_text`：当前算法形成历史事件（最近形成步的 Idea、Diff 摘要、结果与 Fitness 变化）。
4. `[Instruction]`：统一改进指令（保留有用机制、参考先前尝试与结果、以不同实现重试失败方向）。
5. `_output_contract`：输出单句 Idea 与单个可执行 Code 块。

## 2. 算子逻辑

- **轻量局部演变算子**：根据最近形成步的差分信息，针对性微调或重构当前实现。

## 3. 特殊机制说明

- **轻量差分来时路**：精简历史记录，仅展示最邻近的核心演变差分，降低冗余信息干扰。

## 4. 真实完整的提示词模板

### 改进生成 Prompt（Generation Prompt）
````text
[Task]
{task_description}
{fitness_direction_hint}

[Current Algorithm]
Fitness: {anchor_fitness}
```python
{anchor.evaluator_input_code}
```

{history_text}

[Instruction]
Improve the current algorithm using the provided search history. Preserve useful mechanisms, consider previously tested modifications and their outcomes, and propose one coherent modification.
Historical outcomes are evidence rather than strict prohibitions; previously unsuccessful ideas may be revisited with a materially different implementation.
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
