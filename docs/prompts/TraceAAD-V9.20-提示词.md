# TraceAAD-V9.20 提示词

## 1. 上下文组织逻辑

由标准区块组装：
1. `[Task]`：任务说明。
2. `[Current Algorithm]`：当前算法代码与 Fitness。
3. `[Crossover Reference Algorithm]`（若执行 CROSSOVER）。
4. `[Direct Outcome Ledger]`（若提供）：直接下游尝试的结果账本。
5. 模式自适应的 `[Formation History]`：
   - 若 `context_mode == "explore"`：截断长父链，仅保留当前代码与账本失败规避提示。
   - 若 `context_mode == "develop"`：完整呈现来时路。
6. `[Instruction]`：动作指令。
7. 可靠性与输出契约。

## 2. 算子逻辑

- **`DEVELOP` 算子**：在现有策略框架内进行局部利用与深化，充分利用完整来时路经验。
- **`EXPLORE` 算子**：解耦长历史，防止模型被长成功祖先链上的过往思路锚定，结合直接结果账本规避已失败尝试。
- **`CROSSOVER` 算子**：跨分支融合算子。结合参考算法的代码与历史经验，实现互补重组。

## 3. 特殊机制说明

- **探索上下文解耦与直接结果账本（Direct Outcome Ledger）**：在 Explore 模式下截断冗长成功父链，仅提供简要当前代码与直接结果账本，避免新方向被旧实现的思维定势束缚。
- **模式自适应历史暴露**：根据算子类型动态切换历史视图（Explore 时极简化，Develop 时保留完整链路）。

## 4. 真实完整的提示词模板

### 三类动作指令原文（Action Instructions）
- **`Action.DEVELOP`**:
````text
Develop the current algorithm by improving its existing algorithmic strategy. Keep the core strategy intact, but refine the heuristic rules, scoring function, or execution logic based on the formation history.
````
- **`Action.EXPLORE`**:
````text
Propose a materially different algorithmic idea from the current algorithm. Change the primary decision rule, scoring logic, or heuristic mechanism. Do not make cosmetic variations or repeat failed modifications from the direct outcome ledger.
````
- **`Action.CROSSOVER`**:
````text
Synthesize the complementary strengths of the current algorithm and the reference algorithm into one coherent new algorithm. Identify what makes each effective from their descriptions and histories, and combine their core mechanisms. Do not simply concatenate their code.
````

### 统一生成 Prompt（含自适应历史与账本）
````text
[Task]
{task_description}
{fitness_direction_hint}

[Current Algorithm]
Fitness: {format_fitness(current.fitness)}
{current_behavior_section}
```python
{current.code}
```

[Crossover Reference Algorithm]
Fitness: {format_fitness(reference.fitness)}
{reference_behavior_section}
```python
{reference.code}
```

[Direct Outcome Ledger]
{direct_outcome_ledger}

[Formation History]
{formation_history_or_explore_notice}

[Instruction]
{ACTION_INSTRUCTIONS[action]}
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
