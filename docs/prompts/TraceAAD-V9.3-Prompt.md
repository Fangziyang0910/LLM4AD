# TraceAAD-V9.3 提示词

## 1. 上下文组织逻辑

包含三段独立的 Prompt 流程：

### 阶段一：策略规划生成（Strategy Plan Prompt）
- 提出 `strategy_count` 个互补的计算机制策略，禁止写代码、禁止评估 fitness，仅输出编号列表 `Strategy i: ...`。

### 阶段二：轨迹单步决策（Trajectory Decision Prompt）
- 包含 `[Task]`、`[Initial Route Strategy]`、局部轨迹窗口 `window_text`、`[Current Executable Anchor]` 以及决策指导。
- 仅输出一行 `Idea: <one sentence>`，严禁输出代码。

### 阶段三：代码独立实现（Code Implementation Prompt）
- 包含 `[Task]`、`[Current Executable Anchor]`、`[Approved Next Idea]` 以及实现指令，输出完整 Python 代码。

## 2. 算子逻辑

- **宏观策略规划算子**：提出高层机制框架列表。
- **自然语言决策算子**：在轨迹上下文约束下做出单步 Idea 决断。
- **代码独立实现算子**：将自然语言决策落实为严谨的可执行函数。

## 3. 特殊机制说明

- **三阶段独立提示调用契约（Three-Stage Call Protocol）**：在交互层将单次算法生成任务严格解耦为三轮独立 LLM 调用——阶段一提出互补机制策略规划列表，阶段二做出单句自然语言 Idea 决断，阶段三由代码模型独立实现完整可执行程序。
- **调度层三步 Rollout 绑定**：在搜索调度器端尝试每次选择锚点后连续绑定 3 步生成（中途退步子代继续充当临时锚点，最终由三步中最佳结果结算信用）。注意：三步 Rollout 是调度器的分配机制，三阶段调用是交互层的提示词契约。

## 4. 真实完整的提示词模板

### 阶段一：策略规划 Prompt（Strategy Plan Prompt）
````text
[Task]
{task_description}
{fitness_direction_hint}

[Target Function]
{target_function_template}

[Instruction]
Propose exactly {strategy_count} complementary, task-grounded algorithmic strategies for implementing the target function.
Each strategy must state a distinct computational mechanism, not a cosmetic rewrite or isolated parameter value.
Do not write code, estimate fitness, rank the strategies, or discuss search.
Output only one non-empty line per strategy in this exact form:
Strategy 1: <one sentence>
Strategy 2: <one sentence>
...
````

### 阶段二：轨迹决策 Prompt（Trajectory Decision Prompt）
````text
[Task]
{task_description}
{fitness_direction_hint}

[Initial Route Strategy]
{initial_strategy}

{window_text}

[Current Executable Anchor]
Anchor fitness: {anchor_fitness}
```python
{anchor.code}
```

[Decision]
Decide exactly one coherent next algorithmic Idea for the current executable anchor, using the local trajectory evidence.
Treat each recorded result as evidence about that specific implementation; it does not by itself prove that the broader idea is good or bad.
If revisiting a tested idea, require a materially different realization and state that difference.
The Idea must be specific enough to implement from the anchor code alone.
Do not write code, estimate fitness, summarize the whole history, or discuss search.
Output only:
Idea: <one sentence, no more than 300 characters>
````

### 阶段三：代码实现 Prompt（Code Implementation Prompt）
````text
[Task]
{task_description}
{fitness_direction_hint}

[Current Executable Anchor]
Anchor fitness: {anchor_fitness}
```python
{anchor.code}
```

[Approved Next Idea]
{idea}

[Implementation]
Implement exactly the approved Idea from the current anchor.
Keep the current target function signature and contract unchanged.
Return one complete, self-contained implementation.
Keep the code free of internal docstrings, comments, explanation, or trailing analysis; keep only executable source.
Output only:
Code:
```python
<complete function implementation>
```
````
