# TraceAAD V10.4 提示词

## Idea Prompt

````text
# Task Contract
{task_description_and_target_contract}

# Current Algorithm
Idea: {current.idea}
Fitness: {current.fitness}

```python
{current.code}
```

# Recent Algorithm Improvement History

[History 1] Formation step
Idea: {formation_idea}
Result: {improve|regress|plateau} (Fitness: {parent} -> {child})

...

# Reference Algorithm
Idea: {donor.idea}
Fitness: {donor.fitness}

```python
{donor.code}
```

# Design Direction
{refine|pivot|fuse instruction}

# Proposed Algorithm Design
Describe the algorithmic idea you propose. Make the design clear enough to guide its implementation. The Python implementation will be produced separately after this design. Respond with the design, not the Python implementation.
````

`Reference Algorithm` 只在 Fuse 中出现。Init 没有 Current、History 或 Reference。

## 设计方向

### Refine

```text
Improve the current algorithm by continuing to develop its algorithmic idea. Study the current algorithm and how it was formed, then propose the development you judge most promising for making the algorithm better.
```

### Pivot

```text
Study the current algorithm and how it was formed. Identify an important limitation or unrealized opportunity in the current design, and develop a stronger algorithmic idea from that insight.
```

### Fuse

```text
Study the current and reference algorithms and identify the strengths of their designs. Develop a better algorithmic idea that brings their useful insights together as one coherent design.
```

### Init

```text
Design a competitive algorithm for this task from scratch. Develop a clear algorithmic idea that can be implemented as the target function.
```

Idea 调用的输出上限为 1024 tokens，Prompt 不规定 Idea 的句数或字符数。去除可选的 `<think>` 区块后，整个非空响应作为节点 Idea。

## Code Prompt

````text
# Task Contract
{task_description_and_target_contract}

# Current Algorithm
Idea: {current.idea}
Fitness: {current.fitness}

```python
{current.code}
```

# Reference Algorithm
{fuse_donor_if_present}

# Proposed Algorithm Design
{complete_idea_response}

# Implementation
Implement the proposed algorithm design as a complete executable implementation of the target function, using the current implementation as the starting program when one is provided.

Return only the complete Python implementation in a python fenced code block.
````

Code Prompt 不包含形成轨迹、算子名称和算子指令。Code 调用的输出上限保持 16384 tokens。
