# TraceAAD-V5 提示词

## 1. 上下文组织逻辑

采用 Primary 与 Reference 双轨清晰解耦的单阶段生成上下文：
1. `[Task Description]`：任务目标与接口规范。
2. `[Primary Program: the only structural parent]`：主程序（唯一的结构父代），提供当前程序的 Idea 声明、适应度与完整代码。
3. `[Primary Trajectory Context]`：主程序的形成历史轨迹。
4. `[Reference Program: knowledge only, never a parent]`（可选）：参考程序（仅作设计知识借鉴，绝非结构父代），提供参考程序的 Idea、适应度与完整代码。
5. `[Reference Trajectory Context]`（可选）：参考程序的演变历史。
6. `[Requested Modification]`：请求的修改动作（支持结构化或自然语言描述）。
7. `[Operator Constraint]`：注入四大语义算子之一的详细约束。
8. `[Target Function Contract]`：目标函数空接口。
9. `[Instruction]`：实现指令与单代码块输出契约。

## 2. 算子逻辑

包含四大经典语义算子：
- **`trace_ideate`**：从主程序形成历史出发，探索全新算法方向，将已测试分支作为边界且不重复。
- **`trace_refine`**：针对主程序在历史中展现出的优势机制或暴露的弱点，进行聚焦修正与深化。
- **`trace_synthesize`**：双轨合成算子。识别主程序与参考程序中各自主张的核心原则，在主程序结构中使其功能性互动，禁止机械拼接代码。
- **`trace_transfer`**：跨轨迁移算子。保持主程序的核心结构，从参考程序中借鉴恰好一个有效机制，适配到当前主程序中。

## 3. 特殊机制说明

- **主结构父代与参考知识解耦**：在 Prompt 中明确界定 Primary 为唯一的结构父代，Reference 仅提供机制知识参考，彻底杜绝多亲本合并时的结构混乱。
- **双轨迹上下文支持**：支持同时呈现主程序轨迹与参考程序轨迹，为跨分支借鉴提供完整的来龙去脉。
- **结构化动作尝试（StructuredAction）**：早期尝试在动作中解耦 `relation`、`change` 与 `novel_difference` 三要素，并在代码实现期强制约束。

## 4. 真实完整的提示词模板

### 四类语义算子指令原文（Semantic Operators）
- **`trace_ideate`**:
````text
Propose one genuinely new algorithmic idea grounded in the full history. Use later regressions and plateaus as tested boundaries.
````
- **`trace_refine`**:
````text
Make one evidence-grounded refinement. You may deepen, repair, replace, delete, merge, or simplify existing logic; do not default to adding branches.
````
- **`trace_synthesize`**:
````text
Extract one supported principle from each trajectory and make them interact functionally in the primary program. Do not concatenate two implementations.
````
- **`trace_transfer`**:
````text
Keep the primary program's core structure and adapt exactly one supported idea from the reference trajectory.
````

### 代码生成 Prompt（Code Generation Prompt）
````text
[Task Description]
{task_description}

[Primary Program: the only structural parent]
Node p{current_node.id}
Idea claim: {current_node.idea}
```python
{current_node.code}
```

[Primary Trajectory Context]
{primary_history}

[Reference Program: knowledge only, never a parent]
Node p{reference_node.id}
Idea claim: {reference_node.idea}
```python
{reference_node.code}
```

[Reference Trajectory Context]
{reference_history}

[Requested Modification]
relation={action.relation}
change={action.change}
novel_difference={action.novel_difference}

[Operator Constraint]
{operator_constraint}

[Target Function Contract]
{target_function_template}

[Instruction]
Implement exactly the requested change in the primary program.
Do not merely cite an edge or global experience.
Return one complete implementation with the unchanged target contract.
Output only:
Idea: <brief implementation claim>
Code:
```python
<complete function implementation>
```
````
