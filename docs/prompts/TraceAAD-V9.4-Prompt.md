# TraceAAD-V9.4 提示词

## 1. 上下文组织逻辑

统一组装单步生成上下文：
1. `[Task]`：任务描述与优化方向提示。
2. `[Initial Route Strategy]`（可选）：初始路线策略。
3. `window_text`：局部历史轨迹窗口。
4. `failure_memory_text`（若有）：显式失败记忆区块（记录 evaluator 报错或致命缺陷，告诫模型避免同样错误）。
5. `[Current Executable Anchor]`：当前锚点代码与 Fitness。
6. `[Instruction]`：指导模型将评测反馈视作实现风险提示，严禁将错误信息复制进代码，严格输出一行 Idea 与一个代码块。

## 2. 算子逻辑

- **单步生成算子**：在局部轨迹窗口和失败记忆支持下，直接生成单个高质量的 Idea 与可执行代码，要求定向规避已发现的报错。

## 3. 特殊机制说明

- **显式失败记忆（Failure Memory）**：直接在提示词中注入该锚点近期遇到的运行时异常与崩溃信息，防止局部重复踩坑。

## 4. 真实完整的提示词模板

### 锚点单步生成 Prompt（Generation Prompt）
````text
[Task]
{task_description}
{fitness_direction_hint}

[Initial Route Strategy]
{initial_strategy}

{window_text}

{failure_memory_text}

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
When evaluator feedback is present, use it only to correct the relevant implementation risk; do not copy error text into Code.
Keep the current target function signature and contract unchanged.
Return one complete, self-contained implementation.
Keep the code free of internal docstrings, comments, explanation, or trailing analysis; keep only executable source.
Output only one non-empty Idea line and one Code block:
Idea: <one concise sentence describing the implemented algorithm, at most 300 characters>
Code:
```python
<complete executable implementation>
```
````
