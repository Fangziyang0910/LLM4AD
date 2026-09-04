# TraceAAD-V8.3 提示词

## 1. 上下文组织逻辑

采用“代码搜索生成（Search Prompt）+ 事实描述提取（Description Prompt）”的双调用系统：

### 阶段一：算法搜索生成 Prompt（Search Prompt）
1. `[Task Description]`：任务描述。
2. `[Local Exploration Context]`：局部探索脉络摘要。
3. `[Current Algorithm]`：当前算法的适应度、客观事实描述（Description）与完整代码。
4. `[Reference Algorithm]`（仅交叉/参考时呈现）：参考算法的适应度、事实描述与完整代码。
5. `[Operator]`：注入五类精细算子之一的名称与详细中文指导指令。
6. `[Target Function Contract]`：目标函数接口。
7. `[Generation Requirements]`：输出契约（严格输出 `Design Idea:` 与单代码块）。

### 阶段二：独立事实描述 Prompt（Description Prompt）
在候选代码生成后独立调用，提取无主观偏见的行为事实：
1. `[Task Description]`：任务描述。
2. `[Target Function Contract]`：函数接口契约。
3. `[Algorithm Implementation]`：待描述的完整算法代码。
4. `[Instruction]`：指导模型客观陈述代码实际采取的计算步骤、决策逻辑与复杂度，禁止主观夸大。

## 2. 算子逻辑

包含完整的五类精细算子体系（全中文详细引导）：
- **`REFINE`（精细改进）**：在现有策略框架内对核心启发式逻辑、决策条件或优先级函数进行深化完善。
- **`TUNE`（参数与细节调整）**：保持整体逻辑完全不变，仅微调关键参数、权重比例或边界阈值。
- **`SIMPLIFY`（化简与去冗余）**：删除冗余分支与无用计算，提高执行效率与算法鲁棒性。
- **`INNOVATE`（机制创新与跳跃）**：打破当前框架，引入全新的启发式准则、评分机制或搜索视角。
- **`CROSSOVER`（跨算法机制重组）**：提取当前算法与参考算法各自的核心优势机制进行有机结合。

## 3. 特殊机制说明

- **双调用体系与行为事实描述解耦**：设立独立的 Description Prompt，由模型对生成代码进行客观事实描述，将算法的真实代码行为与搜索生成时的意图声称解耦，避免意图虚假或名不副实对后续搜索造成误导。
- **五类全功能算子细分**：首次系统确立包含改进、调参、化简、创新与交叉的完备算子集合。

## 4. 真实完整的提示词模板

### 五类算子指令原文（Operator Instructions）
- **`REFINE`**:
````text
聚焦于继续发展或修复当前机制。根据来时路中已经形成的有效思路、当前算法的薄弱点以及已尝试分支暴露的问题，选择一个最值得继续开发的方向，提出聚焦且完整的下一步改进。
````
- **`TUNE`**:
````text
聚焦于校准当前机制的参数、阈值、尺度、触发条件或控制细节。判断哪些细节限制了当前算法，并为校准它们做出必要的配套修改；配套修改可以包含状态统计、归一化、自适应控制或局部结构调整。
````
- **`SIMPLIFY`**:
````text
聚焦于降低当前算法不必要的机制和实现复杂度。判断哪些部分可以删除、合并、重组，或用更简单的机制取代，同时保留当前算法的关键功能。简化应当减少真实概念或代码复杂度，不是只改名、压缩排版或隐藏相同逻辑。
````
- **`INNOVATE`**:
````text
聚焦于从当前节点探索一个明显不同的核心思路。利用局部探索脉络识别已经停滞或反复失败的方向，提出具有实质机制差异的新路线。可以保留当前算法中仍然有价值的组件，但不应将普通的局部调整当作换新方向。
````
- **`CROSSOVER`**:
````text
以当前算法为主体，从参考算法中识别一项与当前机制互补的思想，将它选择性地适配并融入当前代码。判断两者的功能关系和冲突，完成必要的配套调整。不要机械拼接两份完整代码，也不要无选择地复制整个参考算法。
````

### 搜索代码生成 Prompt（Search Prompt）
````text
[Task]
{task_description}
Fitness direction: {higher is better / lower is better}.

[Current Algorithm]
Fitness: {current_fitness}
Description: {current_description}
Code:
```python
{current_code}
```

[Local Exploration Context]
{history}

[Operator]
{operator_name}: {operator_instruction}

[Reference Algorithm]
Fitness: {ref_fitness}
Description: {ref_description}
Code:
```python
{ref_code}
```

[Generation Requirements]
Use the current algorithm as the direct basis for one meaningful change.
Use the local exploration context to identify effective ideas and already tested directions; avoid repeating a known attempt without a reason.
Allow necessary parameter, condition, data-flow, or local structural changes needed to implement the main intention.
Keep the target function signature and task contract unchanged.
Return one complete executable implementation with no placeholders.
Make the design idea concise, actionable, and consistent with the code.
Output exactly:
Design Idea: <one concise sentence>
Code:
```python
<complete implementation>
```

[Target Function]
{target_function_template}
````

### 事实描述独立 Prompt（Description Prompt）
````text
[Task]
{task_description}

[Generated Design Idea]
{design_idea}

[Generated Code]
```python
{code}
```

Describe the actual behavior of this code factually and concisely.
Do not propose changes and do not restate an intention that the code does not implement.
Output exactly:
Description: <concise factual description>
````
