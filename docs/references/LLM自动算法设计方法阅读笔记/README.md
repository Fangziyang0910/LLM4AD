# LLM 自动算法设计方法阅读笔记

本目录逐篇记录 60 篇 LLM 自动算法设计、广义程序发现与机制分析论文，覆盖 59 个方法和 1 篇纯进化算子分析。每篇笔记只依据本地论文正文或附录，分开记录论文宣称、实验事实和本文分析。

## 证据口径

对每项机制使用以下证据等级：

- **直接支持**：有针对该机制的消融或受控对照，且结果方向与机制主张一致；
- **部分支持**：有相关消融或过程指标，但仍同时改变其他因素，或只在部分任务成立；
- **间接支持**：整体方法优于基线，但没有隔离该机制；
- **未验证**：论文提出该机制，却没有能够判断其独立作用的实验；
- **反向或混合证据**：部分实验不支持该机制，或不同任务、模型、预算下方向不一致。

整体方法胜出只能支持联合系统有效，不能自动证明其中每个组件有效。跨方法结果若同时改变 LLM、评价预算、提示、任务或嵌入框架，不用于单独归因机制。

## 单篇笔记结构

每篇笔记统一回答：

1. 论文解决什么问题，最终设计对象是什么；
2. 论文宣称了哪些机制贡献；
3. 主结果、消融和过程分析分别支持了什么；
4. 哪些机制只有整体结果、缺少独立验证；
5. 机制可能有效的底层逻辑是什么；
6. LLM4AD / TraceAAD 可以学习什么、需要什么前提、怎样做最小验证；
7. 结论在任务、模型、预算和 evaluator 上有什么边界。

每个事实性结论应给出本地论文中的章节、表、图或附录定位。底层逻辑和可学习之处属于阅读分析，不写成论文已经证明的结论。

## 阅读索引

### 语义进化与种群搜索

1. [Evolution through Large Models](01-ELM.md)
2. [Language Model Crossover](02-Language-Model-Crossover.md)
3. [Evolving Code with a Large Language Model](03-Evolving-Code.md)
4. [FunSearch](04-FunSearch.md)
5. [EoH](05-EoH.md)
6. [MEoH](06-MEoH.md)
7. [HSEvo](07-HSEvo.md)
8. [QUBE](08-QUBE.md)
9. [CDEoH](09-CDEoH.md)
10. [EoH-S](10-EoH-S.md)
11. [Controlling the Mutation in LLMs](11-Controlling-Mutation.md)
12. [AlphaEvolve](12-AlphaEvolve.md)
13. [ShinkaEvolve](13-ShinkaEvolve.md)
14. [Evolutionary Discovery of RL Algorithms](14-Evolutionary-RL-Algorithms.md)

### 反思、记忆与历史上下文

15. [ReEvo](15-ReEvo.md)
16. [HiFo-Prompt](16-HiFo-Prompt.md)
17. [Experience-Guided Reflective Co-Evolution](17-Experience-Guided-CoEvolution.md)
18. [MeLA](18-MeLA.md)
19. [MeEvo](19-MeEvo.md)
20. [DeltaEvolve](20-DeltaEvolve.md)
21. [PhyloEvolve](21-PhyloEvolve.md)

### 树搜索、规划与预算分配

22. [MCTS-AHD](22-MCTS-AHD.md)
23. [Planning of Heuristics](23-Planning-of-Heuristics.md)
24. [PathWise](24-PathWise.md)
25. [Clade-AHD](25-Clade-AHD.md)
26. [CogMCTS](26-CogMCTS.md)
27. [RefineEvo](27-RefineEvo.md)
28. [Compute Allocation / BaSE](28-Compute-Allocation-BaSE.md)

### 表示扩展与系统级合成

29. [RedAHD](29-RedAHD.md)
30. [Hercules](30-Hercules.md)
31. [MoH](31-MoH.md)
32. [InstSpecHH](32-InstSpecHH.md)
33. [A2DEPT](33-A2DEPT.md)
34. [BEAM](34-BEAM.md)

### 模型学习、智能体与协作

35. [CALM](35-CALM.md)
36. [EvoTune](36-EvoTune.md)
37. [Fine-tuning LLM for AAD](37-Fine-tuning-LLM-for-AAD.md)
38. [AHD Agent](38-AHD-Agent.md)
39. [RoCo](39-RoCo.md)
40. [CORAL](40-CORAL.md)
41. [Beyond Inference-Time Search](41-Beyond-Inference-Time-Search.md)

### 模型训练与学习型搜索增强

42. [Self-Developing](42-Self-Developing.md)
43. [AlgoPilot](43-AlgoPilot.md)
44. [LLaMEA-SAGE](44-LLaMEA-SAGE.md)
46. [Latent Heuristic Search](46-Latent-Heuristic-Search.md)
50. [GAE](50-GAE.md)
57. [Teacher-Aware Evolution](57-Teacher-Aware-Evolution.md)

### 知识、轨迹与结构反馈

45. [Back to the Beginning of Heuristic Design](45-Back-to-the-Beginning.md)
47. [MEMOIR](47-MEMOIR.md)
52. [AutoSND](52-AutoSND.md)
56. [DyACE](56-DyACE.md)
59. [GeoEvolve](59-GeoEvolve.md)

### 种群组织、信用与预算机制

48. [SMCEvolve](48-SMCEvolve.md)
49. [Mutation Without Variation](49-Mutation-Without-Variation.md)
51. [DGA²D](51-DGA2D.md)
53. [RelayEvolve](53-RelayEvolve.md)
54. [BehaveSim](54-BehaveSim.md)
55. [EvoStage](55-EvoStage.md)
58. [TurboEvolve](58-TurboEvolve.md)
60. [PartEvo](60-PartEvo.md)

## 跨论文认识

60 篇笔记共记录 246 项“主张—证据”判断：60 项直接支持、72 项部分支持、83 项间接支持、25 项未验证、6 项反向或混合证据。这里统计的是判断条目而不是论文篇数；同一论文可以同时包含直接消融证据和未经验证的机制解释。

### 1. 最稳定的共同基础是可执行外部反馈

[ELM](01-ELM.md)、[Evolving Code](03-Evolving-Code.md) 和 [FunSearch](04-FunSearch.md) 说明，LLM 的程序先验只有进入执行—评价—选择闭环后，才会转化为可积累的算法改进。底层逻辑是 evaluator 把语言上合理的候选筛成满足任务目标的可执行程序。可学习之处是先保证 evaluator 的可行性、方向、超时和测试边界可靠；否则更强的搜索只会更快利用评价漏洞。

### 2. LLM 的优势在语义变化，但变化幅度必须受控

[Language Model Crossover](02-Language-Model-Crossover.md) 支持 LLM 从多个父代产生可继承的语义组合，[Controlling Mutation](11-Controlling-Mutation.md) 则直接显示提示控制会改变变异比例和代码差异。底层逻辑是 LLM 可以调用预训练得到的程序修改先验，跨越随机字符串或语法树变异难以协调的修改。可学习之处是记录“父代—修改意图—实际 diff—评价结果”，并在固定预算下验证修改是否带来行为差异，而不是只统计文本新颖性。

### 3. 历史有用的关键是可追溯，不是摘要越多越好

[DeltaEvolve](20-DeltaEvolve.md) 的受控比较支持代码上下文比只有标量分数更有信息；[ReEvo](15-ReEvo.md)、[HiFo-Prompt](16-HiFo-Prompt.md) 和 [PathWise](24-PathWise.md) 分别探索反思、前后视提示和路径反馈，但独立证据强度并不相同。底层逻辑是历史改变下一次生成的条件分布；只有历史与真实父子边、操作和结果绑定，模型才可能区分可延续变化与失败。可学习之处是优先保留结构化 action–result 事实，再检验摘要或记忆是否在相同 token 预算下产生额外增益。

### 4. 树和规划首先是预算分配机制

[Planning of Heuristics](23-Planning-of-Heuristics.md) 对搜索策略进行了受控比较，[PathWise](24-PathWise.md) 对 critic 和提示多样性给出消融，[Compute Allocation / BaSE](28-Compute-Allocation-BaSE.md) 直接研究深度与广度的调用分配；[MCTS-AHD](22-MCTS-AHD.md) 的整法结果则不能自动证明 UCT、回传和渐进扩展各自有效。底层逻辑是搜索结构决定有限预算落在哪些候选和路线，而不替代 LLM 单步生成。可学习之处是分开评价起点选择、下一步生成和最终输出，并同时报告 evaluator 次数、LLM 调用和 token。

### 5. 多样性只有转化为有效路线覆盖才有价值

MEoH、HSEvo、QUBE、CDEoH、EoH-S、ShinkaEvolve 和 RoCo 分别用 Pareto、和声搜索、不确定性、类别、启发式集合、过滤或角色维持多样性，但多数证据为部分或间接支持。底层逻辑是多样性可能降低搜索路线的相关性，增加至少一条路线突破的机会；代价是稀释有限评价预算。可学习之处是测量算法行为、路线贡献和 held-out 结果，不把代码差异、角色名称或 embedding 距离直接当成有效多样性。

### 6. 扩大设计对象会增加表达力，也会扩大无效空间

[RedAHD](29-RedAHD.md)、[Hercules](30-Hercules.md)、[A2DEPT](33-A2DEPT.md) 和 [BEAM](34-BEAM.md) 将设计对象扩展到问题规约、搜索方向、程序组件或完整求解器。底层逻辑是固定单函数模板可能排除真正需要的结构变化；同时，更大的表示空间会增加不可执行候选、错误归因和 evaluator exploit。可学习之处是每次只扩大一个可验证层次，并报告合法率、有效候选率、复杂度和独立测试表现。

### 7. 把搜索经验写入模型，需要与在线搜索分开证明

[CALM](35-CALM.md)、[EvoTune](36-EvoTune.md)、[Fine-tuning LLM for AAD](37-Fine-tuning-LLM-for-AAD.md)、[AHD Agent](38-AHD-Agent.md) 和 [Beyond Inference-Time Search](41-Beyond-Inference-Time-Search.md) 分别用 GRPO、搜索反馈训练、DPO、agentic RL 或 solver synthesis 内化设计能力。底层逻辑是参数学习可以摊销未来搜索成本，但也可能只记住任务模板、可执行格式或 evaluator 偏好。可学习之处是将训练成本与在线搜索成本分开，并用权重固定的模型、固定代码和未见实例检验能力是否真正转移。

### 8. 历史应按作用范围分层，且不是越长越好

[MEMOIR](47-MEMOIR.md) 的受控消融支持分离 branch-local 调试事实与 global 跨路线知识，并显示失败记录本身有价值；[DGA²D](51-DGA2D.md) 则显示 first-order transition credit 优于 zero-order，但 second-order 和 full-path 更差。底层逻辑是历史粒度存在偏差—方差权衡：过短会混淆上下文，过长会切碎样本并注入噪声。对 TraceAAD 更直接的学习点是保留完整 lineage 作为可重放真相源，但只把与当前决定相关的局部事实或终止路线摘要送入生成上下文。

### 9. 算子本身会收敛，多样性要测行为而不只测代码

[Mutation Without Variation](49-Mutation-Without-Variation.md) 在移除选择压力后仍观察到结构 attractor、自环和短周期；[BehaveSim](54-BehaveSim.md) 与 [PartEvo](60-PartEvo.md) 分别说明执行轨迹和有意义特征构造的 niche 比随机或静态表面差异更能组织搜索。可学习之处是同时监测 lineage 重访、程序骨架和 held-out 行为距离，并验证多样路线是否真的贡献突破。

### 10. 外部知识只有与当前失败证据对齐时才稳定有用

[GeoEvolve](59-GeoEvolve.md) 让代码诊断触发领域文献检索；其静态知识 prompt 在部分指标退化，而动态检索完整系统更好。[AutoSND](52-AutoSND.md) 进一步把执行证据编译成结构政策。可学习之处是保存“失败状态—查询—证据块—代码修改—结果”的闭环 provenance，不把增加上下文长度本身当成知识机制。

### 当前最可靠的研究顺序

这些论文共同支持的不是某个复杂控制器必然最优，而是一条较稳妥的验证顺序：先建立可靠 evaluator 和可重放 lineage，再验证上下文是否改善单步生成，然后在相同评价与调用预算下验证路线选择，最后才考虑共享记忆、系统级表示扩展或模型参数训练。联合方法的最终胜出只能说明完整配方有效；具体机制仍需最小、受控消融。
