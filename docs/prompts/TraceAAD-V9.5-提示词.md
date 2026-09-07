# TraceAAD-V9.5 提示词

## 1. 上下文组织逻辑

统一组装标准纯树生成上下文：
1. `[Task]`：任务描述与优化方向提示。
2. `[Current Executable Anchor]`：当前可执行锚点代码与 Fitness。
3. `evidence_text`：由历史渲染模块生成的标准父代形成历史事件表（展示每步的 Idea、Diff 代码行增删以及 Result 判定：IMPROVED / REGRESSED / EQUAL）。
4. `[Instruction]`：指导大模型利用历史证据改进算法，强调历史是参考依据而非绝对禁令，允许以不同实现重试先前失败的方向。
5. `_output_contract`：强制输出一行可选 Idea（<= 300 字符）与一个完整的代码块，禁止输出推理分析、算子标签或 patch。

## 2. 算子逻辑

- **统一改进算子**：基于祖先搜索历史与当前代码，提出明确的机制演进思路并直接输出完整函数，兼具利用与探索。

## 3. 特殊机制说明

- **首创严格单代码块输出契约（Strict Markdown Contract）**：使用四反引号严格规范模型输出格式，彻底消灭 Markdown 解析截断与格式崩溃。
- **标准父代形成历史事件表**：建立标准化 Idea-Diff-Result 三元组证据表示。

## 4. 真实完整的提示词模板

### 初始根程序 Prompt（Root Prompt）
````text
[Task]
{task_description}
{fitness_direction_hint}

[Target Function]
{target_function_template}
Keep the function name, arguments, return type, and contract unchanged.
Include every required import and helper in the returned program.

[Instruction]
Create one complete, valid, and competitive initial algorithm.
Avoid a placeholder or trivial baseline.
Output only one optional short Idea and one mandatory full Code block:
Idea: <optional semantic label, at most 300 characters>
Code:
```python
<complete executable implementation>
```
Do not output reasoning, evidence analysis, an operator label, or a patch.
````

### 锚点改进生成 Prompt（Generation Prompt）
````text
[Task]
{task_description}
{fitness_direction_hint}

[Current Executable Anchor]
Anchor fitness: {anchor_fitness}
```python
{anchor.evaluator_input_code}
```

{evidence_text}

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
