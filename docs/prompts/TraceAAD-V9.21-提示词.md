# TraceAAD-V9.21 提示词

## 1. 上下文组织逻辑

包含假说构思与独立实现两阶段调用：

### 阶段一：假说构思 Prompt（Idea Prompt）
1. `[Task]`：任务描述。
2. `[Stable Scaffold]`：当前稳定脚手架算法的代码与 Fitness。
3. `[Working Implementation]`（可选）：当前正在推进的工作代码。
4. `[Entry Hypothesis]`：当前分支的入口假说。
5. 证据区：形成历史、账本以及可选的 `[One Public Experiment Card]`（跨分支公开卡片，仅提供自然语言 Idea 与增益，不给代码）。
6. `[Instruction]`：根据分支目标提出具体假说，输出仅一行 `Idea: ...`。

### 阶段二：独立实现 Prompt（Realization Prompt）
1. `[Idea Under Test]`：待测试的具体假说。
2. `[Base Implementation]`：基础实现代码。
3. 证据区。
4. `[Instruction]`：将测试假说落地为完整 Python 函数。

## 2. 算子逻辑

- **假说构思算子**：以自然语言构想算法核心假说，不受代码实现的细节束缚。
- **假说落实算子**：单向将批准的假说翻译为严谨代码。

## 3. 特殊机制说明

- **假说与实现解耦**：第一阶段仅构思假说，第二阶段独立落实代码。
- **公共实验卡片（Public Cards）**：支持在构思阶段跨分支共享高层自然语言经验，但严格禁止共享代码以防污染。

## 4. 真实完整的提示词模板

### 假说生成 Prompt（Idea Prompt）
````text
[Task]
{task_description}
Fitness is this task's score and higher is better.

[Stable Scaffold]
Fitness: {base_fitness}
```python
{base_code}
```

[Entry Hypothesis]
{entry_idea}

[Formation History]
{formation_history}

[Direct Outcome Ledger]
{ledger}

[One Public Experiment Card]
{public_card}

[Instruction]
{instruction}
Write only one Idea line, at most 500 characters; do not write code.
Idea: <one concise algorithmic hypothesis>
````

### 假说指令原文（Proposal Instructions）
- **`continue`**:
````text
State the same algorithmic hypothesis in a more precise form, choosing one repair, reimplementation, or refinement justified by the evidence. Do not introduce an unrelated hypothesis.
````
- **`branch`**:
````text
Propose one materially different algorithmic hypothesis for this task. Use the public experiment only as a possible source of a compatible mechanism; do not copy it blindly.
````

### 代码落地实现 Prompt（Realization Prompt）
````text
[Task]
{task_description}
Fitness is this task's score and higher is better.

[Idea Under Test]
{idea}

[Base Implementation]
Fitness: {base_fitness}
```python
{base_code}
```

[Instruction]
Implement the Idea under test as one coherent, executable change. This is an independent realization: do not refer to another response and do not assume it exists.
Keep the target function signature unchanged.
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
