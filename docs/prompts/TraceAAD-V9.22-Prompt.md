# TraceAAD-V9.22 提示词

## 1. 上下文组织逻辑

继承 V9.21 的清晰两阶段契约，上下文由冻结的批次脚手架与独立分支上下文构成：
1. 阶段一（假说构思）：提供任务描述、冻结脚手架代码、当前分支代码与入口假说，输出单行 Idea。
2. 阶段二（代码实现）：提供批准假说与基础代码，输出完整 Python 实现。

## 2. 算子逻辑

- **独立分支假说构思算子**：在严格隔离的分支内部构想假说。
- **独立分支代码实现算子**：将假说实现为可执行代码。

## 3. 特殊机制说明

- **冻结批次上下文（Frozen Batch Context）与分支强隔离**：禁用跨分支公开卡片盲目迁移，强化分支独立性。

## 4. 真实完整的提示词模板

### 提议指令原文（Proposal Instructions）
- **`continue`**:
````text
State the same algorithmic hypothesis in a more precise form, choosing one repair, reimplementation, or refinement justified by the evidence. Do not introduce an unrelated hypothesis.
````
- **`branch`**:
````text
Propose one materially different algorithmic hypothesis for this task. Use the public experiment only as a possible source of a compatible mechanism; do not copy it blindly.
````

### 阶段一：假说构思 Prompt（Idea Prompt）
````text
[Task]
{task_description}
Fitness is the task score; higher is better.

[Stable Scaffold]
Fitness: {base_fitness}
```python
{base_code}
```

[Current Working Implementation]
Fitness: {working_fitness}
```python
{working_code}
```

[Entry Hypothesis]
{entry_idea}

{formation_history}

[Implementation Evidence]
{ledger}

[One Public Experiment Card]
{public_card}

[Instruction]
{instruction}
Write only one Idea line, at most 500 characters; do not write code.
Idea: <one concise algorithmic hypothesis>
````

### 阶段二：独立实现 Prompt（Realization Prompt）
````text
[Task]
{task_description}
Fitness is the task score; higher is better.

[Idea Under Test]
{idea}

[Base Implementation]
Fitness: {base_fitness}
```python
{base_code}
```

[Current Working Implementation]
Fitness: {working_fitness}
```python
{working_code}
```

{formation_history}

[Implementation Evidence]
{ledger}

[Instruction]
Implement the tested idea as a complete Python program, changing only what the idea requires.
Preserve the target function signature and interface contract.
Return a complete Python program in one fenced block:
Code:
```python
<complete program>
```
Do not write a module or function docstring, and do not write comments.
````
