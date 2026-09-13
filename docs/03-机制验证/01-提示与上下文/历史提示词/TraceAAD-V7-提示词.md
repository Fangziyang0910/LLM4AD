# TraceAAD-V7 提示词

## 1. 上下文组织逻辑

在 Prompt 上下文层面与 V6 保持完全同构：
1. `[Task]`：任务描述与优化方向提示。
2. `[Current Program History]`：当前算法历史。
3. `[Current Program]`：当前算法代码。
4. `[Reference Program History]`（可选）：参考算法历史。
5. `[Reference Program]`（可选）：参考算法代码。
6. `[Requested Modification]`：自然语言修改动作。
7. `[Target Function]`：目标函数空接口。
8. `[Instruction]`：实现指令与单代码块输出契约。

## 2. 算子逻辑

- **自然语言动作驱动**：保持与 V6 相同的自然语言动作落实逻辑，支持单亲本局部改进与可选参考程序机制嫁接。

## 3. 特殊机制说明

- **提示词保持稳定基准**：V7 继承了 V6 的全部提示词模板与解析协议，不改变文本契约，以便隔离调度器改动的因果影响。
- **调度层代码去重与衰减探索**：V7 的核心创新发生在搜索调度层——引入唯一可执行状态与代码哈希去重（重复生成的代码直接复用已有评价结果，不浪费真实评价预算），并引入随搜索预算消耗而逐渐衰减探索强度的 UCB 机制。

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
