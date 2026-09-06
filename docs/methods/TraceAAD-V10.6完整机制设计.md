# TraceAAD V10.6 完整机制设计

日期：2026-09-06。状态：已实现，87项测试及五任务真实联调通过，15路正式实验已启动。本稿取代[V10.6讨论稿](TraceAAD-V10.6机制讨论稿.md)的首发建议。预算范围采用已确认的“保留V10.5分配，不上线个体潜力预测”。实现入口见[运行说明](../../experiments/traceaad_v10_6/README.md)，实际进度和已发现的摘要语义反例见[启动记录](../../experiments/traceaad_v10_6/launch_20260906.md)。

摘要与算子修订：依据用户进一步要求，摘要目标约500字，单条Prompt上限1024 tokens，历史最多8条、8192 tokens；借鉴BehaveSim的实现与行为案例，对Refine/Pivot/Fuse作有限的决策语义澄清。下文为修订后的唯一推荐配置。

提示词修订：发给生成模型的指令统一采用正向行动表述，每个信息块承担一个职责。任务调用、算子方向和输出格式分别表达；研究诊断与失败案例保留为本文设计依据。

生成视角修订：每次调用都是一个自洽的算法设计任务。算子用“沿当前思路改进、尝试不同思路、结合两份算法的有用思路”表达；行为分析术语留在研究解释和确有需要的任务调用说明中。

## 1. 核心决定与目标

**V10.6采用一次调用先输出完整Code、再输出简短实现摘要；以真实父子关系和评价构成形成历史；预算保留V10.5的质量与次数基座、算子条件分布和逐评价重选。** 为满足先选父后定算子的顺序，用等价联合分布重排抽样，不改变相同可扩展集合和基础权重下的目标概率。新Prompt会改变哪些长程序能够容纳，因此不把整个版本的实际搜索路径称为与V10.5等同。

优化目标是在固定真实评价预算下获得更强的最终算法，并确认未知实例与跨规模表现。轨迹首先帮助生成器理解当前实现的形成过程。摘要准确率、局部改善率、谱系数量都不是最终优化目标。

本版撤下个体EI/Beta估计及20%机会混合通道。相近fitness、出生算子和回撤量能否支持增益迁移尚未得到验证；不能为了凑出可计算的评分，用这些特征假装识别了潜力。群体统计并非原则上不能用于个体，但需要相关性、条件可迁移性和不确定性校验，目前不具备这种证据。

本版也不使用活跃代表、候补池、固定多步保活、行为聚类、LLM价值critic或在线算子成功率调整。退步后的多步潜力仍未解决；保留旧分配是当前行动选择，不是宣称旧分配最优。

## 2. 已有证据如何约束设计

- V10.5前750次评价的15路中，5847次Refine涉及2762个父节点，2001个只获得一至两次Refine（72.4%）。这一政策分布下的数据不足以支撑逐节点独立潜力曲线。[覆盖核查](../analysis/机制分析/TraceAAD-V10.6分配反馈覆盖核查-20260906.json)
- CVRP node363的depot偏置在零值上相乘，文字声称的机制未生效，而实际程序获得较好fitness。应修正历史描述，不因意图未实现就自动修改或淘汰程序。[V10系列核查](../analysis/机制分析/TraceAAD-V10系列实验核查-20260906.md)
- V10.4的计划先行、长文本历史与实现上下文问题，不等同于代码后描述。MCTS-AHD在代码生成后另调模型提取至多三句描述；本稿只借鉴其方向，单次调用的效果不能由其结果保证。[原文§3.1、Appendix E.2](https://arxiv.org/html/2501.08603v3)
- 已有形成来时路实验支持保留与当前节点匹配的近期路径；加入所有直接子代尝试没有稳定额外收益。[生成上下文经验](../knowledge/生成上下文经验.md)
- BehaveSim Figure 5/§4.2指出，Code 6因bug使future_potentials全为零，Code 8因距离项权重占主导，两者都表现为最近邻；Figure 2展示argmin/argmax的微小替换可以改变选择行为。本版由此完善实现摘要，未引入其行为度量，也不声称论文验证了本版摘要方案。[BehaveSim原文](https://arxiv.org/html/2603.02787v1)
- 过去的landing、历史贡献分及固定开发块没有形成跨任务稳定优势；不把“没开发过”写成“开发价值高”。[预算分配经验](../knowledge/预算分配经验.md)

## 3. 搜索对象、初始化与入档

沿用单父程序树。节点继续保存`id, code, idea, fitness, evaluation_id, parent_id, operator, donor_id`。V10.6的`idea`字段存代码后实现摘要，沿用字段只是兼容存储，不再表示事前计划；渲染标签统一为`Implementation Summary`。

初始化从头生成8个有效根，Init也采用Code→Summary。初始化正式评价计入总预算。生成无效不获得根名额；预算用完仍不足8根时如实报告初始化未完成，不额外补预算。正式比较不从历史赢家热启动。

每次非初始化迭代只产生一个子代。有限有效fitness的候选全部入档，包括退步、持平和重复代码。Fuse仍以选中的parent作为唯一父节点，donor只记录引用。摘要缺失不影响入档；低分、代码长和“看起来不像Pivot”都不成为额外语义门槛。

最终输出全档案中fitness最大的程序；当前不能装入生成上下文的程序仍可成为最终输出。

## 4. 预算分配：概率不增加新的潜力假设

### 4.1 基础分布

令A为有限有效fitness、最低完整父代Prompt能够放入上下文的节点集合，N=|A|，q(n)为越大越好的原始fitness。沿用既有ESS校准：

\[
E=\min(N,\max(0.1N,2)),\qquad
w(n)=\exp[\beta(q(n)-q_{max})].
\]

β使质量归一化权重的ESS尽可能达到E；若并列最好节点数超过E，沿用可达下界处理。令c(n)为该node ID已经被选为父代的次数：

\[
p_0(n)=\frac{w(n)/\sqrt{c(n)+1}}{\sum_{j\in A}w(j)/\sqrt{c(j)+1}}.
\]

次数只作为沿用的机会分散规则，不解释为成功率或潜力。一次候选选定后加一；同一pending请求的传输重试、恢复和token计数不重复增加。质量ESS、次数修正后的ESS分别记录，不声称前者控制了后者。

### 4.2 先选父，再定算子

V10.5目标联合分布为：R/F用p0选父；P用0.5p0+0.5/N；请求算子比例为0.50/0.15/0.35。V10.6按下面完全等价的方式执行：

\[
m(n)=0.925p_0(n)+0.075/N.
\]

先按m选择父节点n，再按以下条件概率选择请求算子：

\[
P(R\mid n)=\frac{0.50p_0(n)}{m(n)},\quad
P(P\mid n)=\frac{0.075p_0(n)+0.075/N}{m(n)},\quad
P(F\mid n)=\frac{0.35p_0(n)}{m(n)}.
\]

三者之和为1，边际请求算子比例仍为50/15/35；这是抽样次序变换，不是节点条件价值学习。它保持相同A与p0下的联合概率，不保证同一随机seed产生与旧版相同的实现路径。parent日志记实际边际m(n)、条件算子概率和`parent_route=joint_marginal`，不虚构旧版quality/uniform隐变量。

### 4.3 Fuse donor与回退

请求Fuse后，在档案中排除parent自身、祖先、后代，再按完整Fuse最低上下文能否容纳过滤。从合法候选中取fitness最高5个（不足5个则全部，平分按ID），均匀选donor。该规则只是沿用谱系约束，不宣称保证实际机制互补。

没有合法donor时，将这次Fuse执行为Refine，parent不变，不重抽、不生成第二个候选；记录requested=Fuse、executed=Refine。此时实际R/F比例由可用donor决定；不能声称实际执行比例始终严格50/15/35，也不把缺失Fuse质量转给Pivot。

## 5. 任务描述：把接口作用说清楚

保留任务定义、完整函数接口和原有评价器；V10.6的`Execution Context`简述调用时机、可用输入、输出如何进入决策及框架负责的状态更新。说明取自实际evaluator，采用直接陈述可执行规则的写法。

| 任务 | 必须明确的实际调用语义 |
| --- | --- |
| TSP construct | 每个构造步骤接收current、destination、未访问候选和距离矩阵，返回候选中的节点ID；当前evaluator给出的候选按距离组织，最后剩余节点由框架接上并计算闭环路长 |
| CVRP ACO | 每实例利用距离、坐标、需求和总容量计算一次静态边prior；ACO后续以pheromone^alpha × prior^beta × 访问/容量mask形成转移权重，并负责路线及实时容量更新；prior按现有框架作floor处理 |
| OP ACO | 每实例利用收益、距离和总预算maxlen计算静态边prior；ACO负责访问、已行程和回仓预算可行性的动态更新；目标是收益最大 |
| OBP | 每件物品到来时，函数接收当前物品大小与可行箱的剩余容量，返回优先级向量，框架按argmax放置；严格单调分数变换保留该次排序 |
| VRPTW | 每步提供已经过容量、时间窗及回仓可行性检查的客户集合，函数选择其中一个客户；当前节点为客户时，也可返回depot结束路线；时间、服务过程与容量由框架更新 |

运行限制从对应run的真实配置填入，不新增代码行数、组件数或“必须NumPy单公式”等算法限制。当前模板已经包含部分上述内容，应消除重复叙述；不是把这张表机械叠加到旧描述后。

核对来源：`llm4ad/task/optimization/{tsp_construct,cvrp_aco,op_aco,online_bin_packing,vrptw_construct}/evaluation.py`及各自`template.py`。不改evaluator来迎合新Prompt。

任务公共指令采用以下正向版本，替换旧公共块中的否定式提醒；各任务的描述和接口docstring使用同样表述原则，同时保留真实接口语义：

````text
# Task Contract
Design an algorithm for the task below by implementing the provided Python function.

<task objective and concise Execution Context>

Target interface:
```python
<template program>
```
Objective: maximize evaluator fitness (higher is better).
Use the information supplied through this interface and produce its specified
output within the stated runtime limit.
````

## 6. 生成协议与算子

### 6.1 唯一默认：一次调用Code→Summary

不增加独立计划或摘要调用，不要求模型输出事前Idea。完整上下文共同用于生成代码和后置摘要。代码输出在先，使摘要可以条件于已经完成的实现；这是可检验的生成假设，不能称为语义保证。

正式输出协议：

````text
Return one complete Python implementation followed by its summary in this format:

```python
<complete target implementation, including all required imports and helpers>
```
Summary: <implementation summary>

Write approximately 500 words in 2–3 paragraphs, scaled to the implementation's
complexity. Explain how the algorithm implemented above works, including its main
idea and the important formulas, parameter values, and steps in the code.
Explain when its key rules apply.
````

提供了Current Algorithm时，在上述摘要要求后追加一句：

```text
Describe the main changes relative to the current algorithm.
```

初始化直接使用公共摘要要求。生成模型每次收到与实际输入对应的说明；Init/Refine/Pivot/Fuse等调度标记保留在方法实现和日志中，Prompt以本次算法设计要求作为标题和正文。

Summary同时包含当前决策机制、关键实现逻辑与本次主要修改。按用户要求以约500字为目标：沿用英文摘要时Prompt写约500 words，中文摘要则约500汉字，分别计量，不将字、词、token按固定比例换算，也不为这一长度修改额外切换生成语言。简单实现可以更短，不凑长度。不新增多个必填输出字段，仍保存为一段Summary文本，允许2–3个自然段。

摘要依次说明以下内容，不要求每个程序硬填同样的检查清单：

1. **决策规则**：可用输入经过什么计算，怎样通过排序、argmin/argmax、采样或分支产生动作或prior。
2. **决定行为的实现细节**：关键系数及作用、归一化尺度、条件方向、mask、clip/floor、初始化与更新顺序、tie/fallback；只写理解这份实现必要的部分。
3. **本次修改及其作用条件**：具体改了哪些表达式或逻辑，在什么输入条件下可能改变决策。Fuse交代实际采用的两方元素；Init不作父代比较。

可从代码确定的零值传播、代数抵消、严格单调排序，可以明确描述；依赖未知输入范围的权重支配或分支触发频率必须作条件表述。摘要不是正式行为测量，也不是逐节点语义认证。增加篇幅用于保留必要细节，不扩张为规划、逐行讲解或成功归因。

单条摘要进入后续Prompt的硬上限放宽为1024实际tokens（1K，`summary_tokens=1024`）。长度目标与上下文上限之间留出余量，以减少截断；是否超限以实际tokenizer为准。超过时只保留不超过上限的完整前缀段落，预留省略标记空间；不跳过中间段落拼出新的解释，不截断句子或公式。若第一个完整段落就放不下，则该次Prompt省略此摘要并标记原因。原始完整摘要和response仍存档，不追加压缩调用、不因摘要超长丢弃可评价代码。该上限也适用于根与donor摘要，统一计算一次并缓存。

### 6.2 四种生成行为

保留Init/Refine/Pivot/Fuse及中等强度的生成自由度。生成模型负责根据本次提供的任务、算法和历史，设计并实现一个算法。算子说明它与输入算法之间的设计关系。BehaveSim帮助我们认识到实现细节的重要性；这一认识通过完整代码、必要的调用说明和实现摘要进入上下文，通用算子使用直接的算法设计语言。

| 算子 | 正向行动指引 |
| --- | --- |
| Init | 为给定任务设计一个有竞争力的算法，并实现提供的函数接口 |
| Refine | 沿当前算法的主要思路设计更好的版本，自主选择最有希望提升表现的实现修改 |
| Pivot | 为同一任务设计采用不同主要思路的算法，将当前算法作为参考 |
| Fuse | 结合当前与参考算法中的有用思路，选择并调整适合共同使用的部分，争取超过两份输入 |

具有形成历史时采用下列说明。质量目标与接口/运行限制由Task Contract表达，此块说明提供的历史是什么，以及它与本次设计算法的关系：

```text
The development history describes how the current algorithm was built and how
previous versions performed. Use it as context for this algorithm design task.
```

四条算子指令如下。每次Prompt装入被选中的一条，统一替换旧指令；Init仍承担原有职责，措辞一并收敛：

```text
Init:
Design a competitive algorithm for this task and implement it using the provided
function interface.

Refine:
Build on the current algorithm's main idea to design an improved version for
this task. Choose the implementation changes most likely to improve its performance.

Pivot:
Design an algorithm for this task using a different main idea from the current
algorithm. Use the current algorithm as a reference for developing a promising
new approach.

Fuse:
Design an improved algorithm for this task by combining useful ideas from the
current and reference algorithms. Choose and adapt the parts that work well
together, aiming to outperform both algorithms.
```

### 6.3 Prompt组装与信息职责

正式Prompt由任务与调用说明、Current Algorithm的完整代码与评价分数、需要时的Reference Algorithm、匹配的发展历史、本次算法设计要求、输出格式组成。历史解释紧邻实际历史展示，历史为空时自然省去。设计要求块使用`# Algorithm Design Task`标题和所选指令正文。Init使用任务、初始设计要求和公共输出契约。各块按既有顺序组装，每种指令出现一次。

每次调用的材料足以回答：解决什么问题；函数提供哪些信息、输出怎样被使用；已有算法是什么、表现怎样；这次沿用、替换还是结合哪些算法思路；返回什么。模型通过这些材料完成当前设计。种群、预算分配、节点价值、行为距离及算子调度由外部搜索程序负责；其名称与数值保存在控制器日志中。

写作要求是“目标＋行动＋必要条件”，并从生成模型收到的这一次任务出发检查用语。共同目标放在Task Contract，算子定义本次算法设计方向，Summary描述完成代码中的算法思路与实现。BehaveSim的案例、历史失败清单及本文设计论证供科研与实现人员查阅；生成模板以本节标明的正式指令块为准。长摘要的容量用于具体算法内容。

长度计数、代码完整性、接口合法性、评价和恢复由现有程序设施校验。接口中的数学条件、真实代码和历史数据按事实保留。此次指令调整遵循用户的提示设计偏好，效果仍由实际生成与搜索结果检验。

研究解释上，OBP对同一分数作严格单调变换，argmax通常不变；ACO中即使边权排序相同，调整相对权重仍可能改变归一化转移概率。这解释了为什么实现细节需要保留。确有必要的argmax或ACO转移公式放在对应任务的调用说明中，说明候选函数的输出如何被使用。通用算子采用第6.2节的算法设计表述，不把“概率变化、激活条件、尺度调整”设成所有任务共同的生成目标。

Refine不要求行为完全不变，Pivot也不要求修改大量代码。二者仍可能产生重叠的候选，区分的是本轮生成方向，而不是事后证明其属于哪个语义类别。发现一个未生效项不触发自动修复、回滚或免费重生成；代码完成后Summary如实描述，正式fitness决定结果。

生成器看到当前代码、真实质量与匹配的形成历史；Fuse另见donor代码、摘要与质量。沿用“历史失败在限制被处理后可重访”的原则；提示只说明整体修改与评价相伴，不下组件因果结论。

算子日志表示实际使用了哪条指令，不表示已通过语义分类器验证。上述改变不新增执行probe、行为阈值、算子验收器或新颖奖励，也不改变预算分配与算子比例；模型未充分遵循Pivot或Fuse时仍按真实代码评价。这保留了算子自由度，也避免把尚不可观察的行为差异硬编码成拒绝门槛。

## 7. 历史组织与上下文裁剪

沿用最近最多8条形成边，包含当前节点自己的形成事件，按时间从旧到新展示；历史区上限为8192实际tokens（`history_tokens=8192`）。每条按真实父子关系取出修改前后质量和child的代码后摘要；Fuse形成事件额外显示当时参考算法的fitness。给模型展示“先前版本、修改后版本、参考算法”，父子ID、donor ID和算子代码保留在存档及`history_ids`等追踪字段中。

示意（数字为假设，不是实验结果）：

```text
Step 3
Previous version fitness: -9.3
Reference algorithm fitness: -8.9
Resulting version fitness: -9.0
Implementation Summary: <该次完成代码的实现摘要>
```

参考算法的历史补充仅使用当时的评价分数；完整谱系和历史代码保留在档案中。当前非根摘要只在最后形成事件展示一次；根与当前donor的摘要放在各自代码块旁。历史区被裁掉时，当前完整代码仍足以作为生成输入，不通过额外复制摘要绕过预算。

精确计数沿用模型服务tokenizer，满足`prompt + 16384 output + 256 margin <= 32768`，即完整输入最多16128 tokens。8192是历史上限，不是硬预留；通常按未触及单条上限的摘要组织8条，最终以实际计数为准。8条摘要若都达到1024，再加事件头就会超出8192，按规则移除最旧事件，不截断最新摘要。历史用满8192时，任务、当前代码、donor及其他输入最多还可用7936 tokens，长代码仍可能迫使移除旧事件，不保证每次保留8条。处理顺序固定：

1. 所有摘要先按第6节1024-token上限生成完整段落视图；组装历史仍超8192时，逐条移除最旧事件，不把每条重新压回一句话。
2. 完整请求超限，继续移除最旧历史事件。
3. 无历史仍超限，依次移除donor、根当前节点的辅助摘要（若有），原始摘要留在档案。
4. 不截断current/donor代码、任务接口或输出契约。最低Prompt的可容纳性按无摘要、无历史的完整代码判定。
5. donor最低Prompt过长则跳过该donor；parent最低Prompt也过长则只暂时不可扩展，仍留在档案；全无可扩展节点则报告未完成。

Prompt代码视图沿用现有普通注释过滤，原始候选模块完整保存和评价。摘要缺失时只显示`[Summary unavailable; refer to implementation.]`，不得拿旧Idea或自动猜测的机制填充。

记录原始/实际展示的摘要tokens、超长省略原因、历史tokens与保留事件数，用来检查实际上下文是否满足设计，不作为在线奖励。

## 8. 解析、预算与恢复：摘要失败不等于代码失败

### 8.1 完整代码判定

响应应以唯一Python代码块开始，代码围栏必须闭合。沿用目标函数名、参数接口、AST与compile检查；完整模块按原样存档并交给统一SecureEvaluator，不通过函数重构丢掉装饰器、辅助函数、类或模块级语句。存在多个代码块造成候选歧义时不挑一个猜测执行。除已有thinking标记清理外，不将代码前的计划文本兼容为新协议。

解析代码和摘要分开处理。摘要非空并符合后置`Summary:`格式记为present，只说明格式存在，不意味语义验证。缺失或格式不合法记unavailable，不修改代码、不追加LLM修复调用。原始response完整落盘。

### 8.2 finish_reason的明确变化

| 状态 | 处理 |
| --- | --- |
| 正常结束，完整代码＋摘要 | 正式评价代码，保存摘要 |
| 正常结束，完整代码但无合格摘要 | 正式评价，摘要标unavailable |
| length，代码块已经闭合且通过语法和接口检查，截断位于代码块之后 | 正式评价完整代码；整段后置摘要不用，标truncated |
| length，代码块未闭合，或代码/接口不合法 | 不评价，不补围栏、不自动补代码，重新分配下一次生成 |
| provider状态unknown | 不伪造stop；按同样完整代码检查，日志保留unknown |
| 拒绝/内容过滤/传输失败且没有正常可用响应 | 沿用明确失败处理，不作为length情形抢救 |

V10.5是在解析前拒绝所有length；V10.6必须连同这一分支修改，不能只换正则。这里接受的是模型已经明确关闭的完整代码块，不推断被截断的代码本应是什么。通过静态检查仍不保证运行正确，运行结果由正式评价决定。

### 8.3 记账与恢复

- 一次真实evaluator调用占一个slot，包括初始化、失败、非法运行输出和超时；有效有限fitness才能入树。LLM失败和静态解析失败单独计成本，不虚构评价。
- 摘要提取不发生第二次模型调用。仅摘要不完整不增加代码无效连续计数；连续50次没有可评价代码沿用停止规则。
- 沿用pending选择、原始响应、evaluation_started和评价收据的持久化。已有响应不重生成，已有评价收据不重评价；同一候选最多发生一次确认的正式评价。
- 只有提交而无可确认收据时，沿用unknown reservation并停止该路核实，不免费重评，也不宣称完整预算已完成。
- V10.6使用独立版本号106、协议/hash和目录；不能把V10.5旧Idea原样解释成新摘要，不能在正式旧run上直接恢复为V10.6。

## 9. 完整流程

```text
initialize independent RNG, empty single-parent tree and evaluation journal
while budget remains:
    if valid root count < 8:
        select Init, no parent or donor
    else:
        A = nodes whose minimum complete parent prompt fits
        p0 = existing ESS quality weights corrected by selection counts
        select parent n from m(n) = .925*p0(n) + .075/len(A)
        select requested R/P/F from the conditional probabilities in section 4
        increment n's selection count once
        if requested Fuse:
            select a fitting top-5 cross-lineage donor, or execute Refine

    build task + current code + optional donor + recent formation events
    checkpoint this selection, exact prompt and post-selection RNG state
    make one model generation call: complete Code, then Summary
    durably save response and finish metadata
    extract complete code and independently classify summary availability
    if code is not evaluable:
        log generation failure; checkpoint; reselect
    else:
        submit one formal evaluation using the complete archived module
        persist evaluation receipt; charge one slot
        if fitness is finite and valid:
            store a root or one child, with optional post-code summary
        log outcome; checkpoint; reselect
return best valid program in the full archive
```

## 10. 唯一首发配置

| 项目 | 配置 |
| --- | --- |
| 主比较 | 五任务，每任务3次独立运行，seed=0/1/2；每路1000次真实评价，初始化计入 |
| 初始化 | 8个有效根，从头生成 |
| 模型与采样 | 同V10.5正式后端的Qwen3.8-27B配置；temperature=1，top_p=.95，top_k=20，thinking关闭；固定并记录真实权重/量化/服务信息，不仅凭模型别名宣称完全一致 |
| 输出 | 一次调用Code→Summary；约500字的目标（英文约500 words、中文约500字，分别计量），简单实现可更短；Prompt摘要硬上限1024实际tokens |
| 分配 | 第4节的等价父代先行分布；请求R/P/F=.50/.15/.35 |
| ESS/次数 | fraction=.10，minimum=2，保留并列处理；1/sqrt(c+1) |
| donor | 合法且可容纳的top-5均匀选；缺失则Fuse转Refine |
| 历史 | 最近最多8条真实形成边，8192 tokens；Fuse事件展示参考算法quality，IDs与算子代码在日志中追踪 |
| 上下文 | 32768总上限；16384输出预留；256余量；整事件裁剪 |
| 评价器 | 原有五任务定义、训练数据、求解设置和各自超时；不改变评价标准 |
| 额外控制器 | 无新增潜力模型、代表集合或保护预算；无独立summary/critic调用 |

这些沿用参数和工程限额不是由新数据识别的最优参数。本版不同时引入调参网格。

## 11. 实现范围与必要验证

实现时建立独立`traceaad_v10_6`入口，复用V10.5的树、ESS、统一评价、计数、日志和恢复设施。主要变更点明确为：

1. Prompt：补任务调用语义，调整输出顺序、摘要标签与形成事件donor事实；采用第6.2节面向单次算法设计的指令，按当前材料组装历史说明与摘要比较句。
2. 解析/候选推进：Code→Summary解析；解除摘要对可评价代码的绑定；按第8节处理length。
3. 调度：父先行的等价联合抽样；日志记录实际边际和条件概率。
4. 上下文：1024-token摘要完整段落视图、8192-token历史区；辅助摘要可移除，最低可容纳性只依赖完整代码和必要协议。
5. 版本/运行：106 checkpoint、机制指纹、独立run与批次入口。不能仅改METHOD字符串而继承硬编码105的加载校验。

实现采用独立版本目录，复用现有评价器与基础设施，保持正在运行的旧版源指纹稳定。设计阶段结束后，按用户授权依次完成实现核对、必要测试、真实服务联调和15路正式实验启动。

落地后的必要验证按真实风险组织，复用现有测试设施：

- 正常代码后摘要、无摘要、摘要过长、代码前Idea、多代码块、错误接口；确认存档模块与实际评价模块相同。
- length在代码中与摘要中两种位置，前者不执行，后者只执行一次完整代码且不使用残缺摘要。
- 父子事件、根、Fuse donor及上下文裁剪；覆盖1024-token边界、首段就超长、完整前缀段落与省略标记、8192历史及16128总输入边界；不重复当前摘要，不截断代码，不把较强donor贡献隐藏成单亲突破。
- 枚举不同p0（包括零权重、并列与单节点），核对父先行联合概率与V10.5目标完全相同；检查Fuse回退不改变Pivot请求质量。
- pending响应与评价收据恢复，证明不重复生成/评价/增加选择计数；旧版恢复行为不回归。

单元验证和真实服务少量冒烟只确认实现协议与成本，不先开展一整套轨迹消融。然后运行一个完整V10.6配方，在共同真实评价前缀及1000终局比较训练best、运行失败与实际生成成本；冻结后测试未知实例和跨规模。三重复不能支撑强统计断言，不能用最好单路替代稳定SOTA结论。

## 12. 当前仍需诚实保留的边界

单次代码后摘要可能仍然描述错误；这是本版要尝试改善的生成行为，未升级为行为验证。质量分配仍可能不给低分结构足够的后续机会；本版没有解决多步潜力识别。ESS随档案增长、节点ID次数重置也仍有局限。

选择这些边界，是为了先实现一条明确的改进链：**完成实现 → 描述实际代码 → 保存匹配的形成历史 → 辅助下一次改进 → 用完整搜索终局决定保留。** 如果用户要求本版同时解决退步节点的开发机会，需要先讨论接受哪种明确的探索规则，不能偷偷用不受支持的潜力估计补齐。
