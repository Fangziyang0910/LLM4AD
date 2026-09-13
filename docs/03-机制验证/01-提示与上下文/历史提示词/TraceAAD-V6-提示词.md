# TraceAAD-V6 提示词

## 1. 上下文组织逻辑

由紧凑历史与自然语言动作驱动的单阶段生成上下文：
1. `[Task]`：任务描述与优化方向提示。
2. `[Current Program History]`：当前程序的演进历史（说明过去尝试过的修改与结果）。
3. `[Current Program]`：当前待改进程序的完整代码。
4. `[Reference Program History]`（可选）：参考程序的演进历史。
5. `[Reference Program]`（可选）：参考程序的完整代码。
6. `[Requested Modification]`：由搜索器传入的自然语言修改动作。
7. `[Target Function]`：目标函数接口与签名契约。
8. `[Instruction]`：生成指导说明与严格单代码块契约。

## 2. 算子逻辑

- **自然语言动作驱动生成**：废除 V5 复杂的 StructuredAction 结构化字段，全面回归自然语言 `[Requested Modification]`，赋予模型更灵活的算法表达空间。
- **单亲或双轨自适应**：在无参考程序时专注于基于自身历史演化；在提供参考程序时，要求保持当前程序主体，仅有机借鉴参考程序中的机制。

## 3. 特殊机制说明

- **自然语言动作与契约规范化**：输出契约明确要求 `Idea` 控制在单行且不超过 300 字符，并附带单个完整 Python 代码块。
- **工程容错与辅助函数支持**：Prompt 中显式明确“simple, complete, valid”，允许模型引入任务模板所需的标准 import 和顶层辅助函数（helper functions）。

## 4. 真实完整的提示词模板

### 初始程序生成 Prompt（Initial Prompt）
````text
{task_description}

Generate a simple, complete, and valid implementation for the target Python function.
The function name, arguments, return type, and contract should remain unchanged.
Imports from the task template remain available. You may add small top-level helper functions when they clarify the implementation.

Output format:
Idea: <one sentence describing the implemented algorithm, no more than 300 characters>
Code:
```python
{target_function_template}
```
````

### 改进代码生成 Prompt（Improvement Code Prompt）
````text
[Task]
{task_description}

[Current Program History]
{history}

[Current Program]
```python
{current_node.code}
```

[Reference Program History]
{reference_history}

[Reference Program]
```python
{reference_node.code}
```

[Requested Modification]
{action}

[Target Function]
{target_function_template}

[Instruction]
The histories describe what has been tried; they may help explain the requested modification from the primary program.
When a reference program is present, the requested modification can adapt an idea from it.
The target function signature and contract remain unchanged.
Return one complete implementation.
Imports from the task template remain available. Include any additional imports and top-level helper functions required by this implementation.
Output only:
Idea: <one sentence describing the implemented change, no more than 300 characters>
Code:
```python
<complete function implementation>
```
````
