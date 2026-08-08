# LLM 自动算法设计方法汇总

本文记录论文库中由 LLM 参与自动算法设计（AAD）、自动启发式设计（AHD）以及更广义程序与算法发现的方法。纳入标准是：方法的产物为可复用的启发式、求解器程序或算法，而不是只为单个实例输出一个解。

当前共收录 **41 个方法**：31 个直接 AAD/AHD 方法，10 个更广义的程序与算法发现方法。综述、benchmark、机制分析、端到端实例求解、奖励函数设计和一般黑盒优化不在本文范围内。完整论文元数据见[论文库文献汇总](论文库文献汇总.md)，实验任务与预算见[AAD 方法实验设置汇总](AAD方法实验设置汇总.md)。

每篇论文的机制主张、实验支持、证据边界和可学习之处，见[逐篇阅读笔记](LLM自动算法设计方法阅读笔记/README.md)。其中真正把算法设计反馈用于模型参数更新的方法，另见[自动算法设计与模型训练结合的方法](自动算法设计与模型训练方法.md)；研究 LLM 作为变异、交叉或局部改进算子时搜索行为的论文，见[LLM 作为进化算子的机制分析](LLM作为进化算子的机制分析论文.md)。

2026-08-08 对 arXiv、OpenAlex、OpenReview、NeurIPS Proceedings 和 ACL Anthology 的补充检索，发现了一批尚未进入本地论文库的轨迹／训练型工作，见[全网补充检索：轨迹与训练型 LLM 自动算法设计](全网补充检索-轨迹与训练型自动算法设计.md)。该清单是待补收候选，不计入本文 41 个已完成逐篇核对的方法。

## 机制分类

分类依据是方法控制持续设计过程的主导机制，而不是应用任务或发表时间。同一方法通常同时包含种群、反思、记忆或多智能体等组件；为避免重复，本文只按其主要贡献归入一类。

| 主导机制 | 解决的主要问题 | 直接 AAD/AHD | 广义算法发现 | 合计 |
| --- | --- | ---: | ---: | ---: |
| 语义进化与种群搜索 | 如何生成、组合和保留候选程序 | 8 | 6 | 14 |
| 反思、记忆与历史上下文 | 如何把评价结果和改进历史用于下一次生成 | 5 | 2 | 7 |
| 树搜索、规划与预算分配 | 如何选择下一条路线并分配有限评价预算 | 7 | 0 | 7 |
| 表示扩展与系统级合成 | 如何从单函数设计扩展到集合、组件或完整求解器 | 6 | 0 | 6 |
| 模型学习、智能体与协作 | 如何让模型或智能体系统随设计过程共同改进 | 5 | 2 | 7 |
| **合计** |  | **31** | **10** | **41** |

## 一、语义进化与种群搜索

这类方法把 LLM 作为理解和修改程序的语义算子，以种群、岛屿或程序库保存候选，再由 evaluator 结果驱动选择。主要差异在于候选表示、生成算子、多样性维护和选择信号。

| 方法 | 范围 | 主导机制 |
| --- | --- | --- |
| [Evolution through Large Models](../../../papers/Evolution_through_Large_Models/) | 广义算法发现 | 以代码模型作为智能变异算子，并与 MAP-Elites 结合生成多样程序。 |
| [Language Model Crossover](../../../papers/Language_Model_Crossover_Variation_through_Few_Shot_Prompting/) | 广义算法发现 | 在 few-shot 上下文中提供多个父代，让 LLM 完成语义层面的交叉与变异。 |
| [Evolving Code with a Large Language Model](../../../papers/Evolving_Code_with_A_Large_Language_Model/) | 广义算法发现 | 将提示驱动的 LLM 程序修改形式化为遗传编程算子。 |
| [FunSearch](../../../papers/Mathematical_discoveries_from_program_search_with_large_language_models/) | 直接 AAD/AHD | 以岛屿种群、优秀程序提示和自动 evaluator 形成程序生成—评价—选择闭环。 |
| [EoH](../../../papers/Evolution_of_Heuristics_Towards_Efficient_Automatic_Algorithm_Design_Using_Large_Languag/) | 直接 AAD/AHD | 以“算法思想 + 代码”为个体，通过多类生成操作和种群选择演化启发式。 |
| [MEoH](../../../papers/MEoH/) | 直接 AAD/AHD | 用 Pareto 选择同时优化启发式性能及其他目标。 |
| [HSEvo](../../../papers/HSEvo/) | 直接 AAD/AHD | 结合和声搜索、遗传操作与多样性维护，扩展种群探索。 |
| [QUBE](../../../papers/QUBE_Enhancing_Automatic_Heuristic_Design_via_Quality_Uncertainty_Balanced_Evolution/) | 直接 AAD/AHD | 在选择中联合候选质量与不确定性，平衡利用和探索。 |
| [CDEoH](../../../papers/CDEoH_Category_Driven_Automatic_Algorithm_Design_With_Large_Language_Models/) | 直接 AAD/AHD | 用算法类别组织多样性，并联合演化思想和代码。 |
| [EoH-S](../../../papers/EoH_S_Evolution_of_Heuristic_Set_using_LLMs_for_Automated_Heuristic_Design/) | 直接 AAD/AHD | 将设计对象从单个启发式扩展为互补启发式集合。 |
| [Controlling the Mutation in LLMs](../../../papers/Controlling_the_Mutation_in_LLMs_for_Efficient_Evolution_of_Algorithms/) | 直接 AAD/AHD | 显式控制 LLM 变异的概率与修改范围，调节搜索步幅。 |
| [AlphaEvolve](../../../papers/AlphaEvolve/) | 广义算法发现 | 以多模型编码智能体、程序数据库和自动评价持续演化可执行程序。 |
| [ShinkaEvolve](../../../papers/ShinkaEvolve/) | 广义算法发现 | 用程序库、搜索组、历史 patch 和经验摘要组织开放式程序进化。 |
| [Evolutionary Discovery of RL Algorithms via LLMs](../../../papers/Evolutionary_Discovery_of_RL_Algorithms_via_LLMs/) | 广义算法发现 | 直接演化可执行的强化学习更新规则与训练流程。 |

## 二、反思、记忆与历史上下文

这类方法的重点是保存“做过什么、结果如何”，再把这些信息压缩或组织为下一次生成的上下文。它们主要改变 LLM 接收到的历史，而不只改变候选选择公式。

| 方法 | 范围 | 主导机制 |
| --- | --- | --- |
| [ReEvo](../../../papers/ReEvo/) | 直接 AAD/AHD | 用候选对比产生短期反思，并递归更新长期反思以指导后续生成。 |
| [HiFo-Prompt](../../../papers/HiFo_Prompt_Prompting_with_Hindsight_and_Foresight_for_LLM_based_Automatic_Heuristic_Des/) | 直接 AAD/AHD | Hindsight 沉淀历史经验，Foresight 根据当前种群调整下一步探索与利用。 |
| [Experience-Guided Reflective Co-Evolution](../../../papers/Experience-Guided_Reflective_Co-Evolution_of_Prompts_and_Heuristics/) | 直接 AAD/AHD | 从历史搜索提炼经验，并让提示与启发式协同进化。 |
| [MeLA](../../../papers/MeLA_Metacognitive_LLM-Driven_Architecture_for_Automatic_Heuristic_Design/) | 直接 AAD/AHD | 进化指导 LLM 生成启发式的元认知提示，而非只进化启发式代码。 |
| [MeEvo](../../../papers/MeEvo_Metacognitive_Evolution_for_Automatic_Heuristic_Design/) | 直接 AAD/AHD | 联合自然进化与元认知进化，兼顾候选探索和设计知识继承。 |
| [DeltaEvolve](../../../papers/DeltaEvolve_Accelerating_Scientific_Discovery_through_Momentum_Driven_Evolution/) | 广义算法发现 | 用程序变化的语义增量及其结果构造动量式上下文，避免反复传入全量历史。 |
| [PhyloEvolve](../../../papers/PhyloEvolve_LLM-Powered_Evolutionary_Code_Optimization_on_a_Phylogenetic_Tree/) | 广义算法发现 | 用谱系树保存程序的完整演化关系，使后续生成利用祖先历史。 |

## 三、树搜索、规划与预算分配

这类方法把算法设计视为路线选择问题。树或关系图保存候选派生关系，搜索策略根据节点、分支或预测价值决定下一次扩展位置。

| 方法 | 范围 | 主导机制 |
| --- | --- | --- |
| [MCTS-AHD](../../../papers/MCTS-AHD/) | 直接 AAD/AHD | 以候选启发式为树节点，通过渐进扩展、价值回传和 UCT 分配评价预算。 |
| [Planning of Heuristics](../../../papers/Planning_of_Heuristics_Strategic_Planning_on_Large_Language_Models_with_Monte_Carlo_Tree/) | 直接 AAD/AHD | 将启发式改进建模为规划过程，用自反思与 MCTS 决定搜索方向。 |
| [PathWise](../../../papers/PathWise/) | 直接 AAD/AHD | 用策略模型、世界模型和评价模型在程序关系图上规划后续修改。 |
| [Clade-AHD](../../../papers/Clade-AHD_Clade-level_Selection_for_MCTS_in_Automatic_Heuristic_Design/) | 直接 AAD/AHD | 聚合进化枝的后代结果形成贝叶斯信念，并用 Thompson sampling 选择分支。 |
| [CogMCTS](../../../papers/CogMCTS_Cognitive-Guided_MCTS_for_Iterative_Heuristic_Evolution/) | 直接 AAD/AHD | 在 MCTS 中加入记忆与认知经验，指导节点扩展和路线切换。 |
| [RefineEvo](../../../papers/RefineEvo_Planning-Guided_Heuristic_Evolution_with_Bidirectional_Experience/) | 直接 AAD/AHD | 用规划确定改进方向，并同时积累成功与失败经验。 |
| [Compute Allocation / BaSE](../../../papers/Compute_Allocation_in_Evolutionary_Search_From_Depth_Breadth_to_Multi_Armed_Bandits/) | 直接 AAD/AHD | 将深挖与广搜视为多臂老虎机，动态分配 LLM 调用和评价预算。 |

## 四、表示扩展与系统级合成

这类方法主要扩展“设计什么”。设计对象不再局限于固定模板中的单个函数，而是启发式集合、问题规约、算法组件或完整求解器。

| 方法 | 范围 | 主导机制 |
| --- | --- | --- |
| [RedAHD](../../../papers/RedAHD_Reduction_Based_End_to_End_Automatic_Heuristic_Design_with_Large_Language_Models/) | 直接 AAD/AHD | 先把目标问题规约到已知模板，再端到端生成完整启发式算法。 |
| [Hercules](../../../papers/Efficient_Heuristics_Generation_for_Solving_Combinatorial_Optimization_Problems_Using_La/) | 直接 AAD/AHD | 先提出可执行的搜索方向，再高效生成和筛选对应启发式。 |
| [MoH](../../../papers/Generalizable_Heuristic_Generation_Through_LLMs_with_Meta_Optimization/) | 直接 AAD/AHD | 用元优化过程自动发现启发式及其改进方式，减少人工预定义搜索器。 |
| [InstSpecHH](../../../papers/LLM_Driven_Instance_Specific_Heuristic_Generation_and_Selection/) | 直接 AAD/AHD | 为不同实例子类生成专用启发式，并在运行前选择匹配的启发式。 |
| [A2DEPT](../../../papers/A2DEPT_Large_Language_Model_Driven_Automated_Algorithm_Design_via_Evolutionary_Program_T/) | 直接 AAD/AHD | 用进化程序树表示和重组算法组件，搜索超越固定单函数模板的结构。 |
| [BEAM](../../../papers/BEAM_Bi-level_Memory-adaptive_Algorithmic_Evolution/) | 直接 AAD/AHD | 在组件层和求解器层进行双层记忆自适应进化，逐步合成完整算法。 |

## 五、模型学习、智能体与协作

这类方法不把 LLM 固定为被动生成器，而是更新模型参数、学习设计策略，或让多个具有不同职责的智能体共同控制生成、评价和选择。

| 方法 | 范围 | 主导机制 |
| --- | --- | --- |
| [CALM](../../../papers/CALM/) | 直接 AAD/AHD | 在演化启发式的同时根据搜索反馈更新语言模型，使生成器与算法共同进化。 |
| [EvoTune](../../../papers/Algorithm_Discovery_With_LLMs_Evolutionary_Search_Meets_Reinforcement_Learning/) | 直接 AAD/AHD | 把进化搜索反馈转化为强化学习信号，训练模型学习如何继续搜索算法。 |
| [Fine-tuning LLM for AAD](../../../papers/Fine-tuning-LLM-Automated-Algorithm-Design/) | 直接 AAD/AHD | 用质量与多样性兼顾的数据采样和偏好优化，专门训练算法设计模型。 |
| [AHD Agent](../../../papers/AHD_Agent_Agentic_Reinforcement_Learning_for_Automatic_Heuristic_Design/) | 直接 AAD/AHD | 将生成、评价和选择建模为 agentic RL 动作，让模型学习控制完整 AHD 循环。 |
| [RoCo](../../../papers/RoCo_Role_Based_LLMs_Collaboration_for_Automatic_Heuristic_Design/) | 直接 AAD/AHD | 由 explorer、exploiter、critic 等角色协作提出、检查和改进启发式。 |
| [CORAL](../../../papers/CORAL_Towards_Autonomous_Multi_Agent_Evolution_for_Open_Ended_Discovery/) | 广义算法发现 | 多个长驻智能体通过共享持久记忆、反思和协作进行开放式程序发现。 |
| [Beyond Inference-Time Search](../../../papers/Beyond_Inference-Time_Search_RL_Synthesizes_Reusable_Solvers/) | 广义算法发现 | 用强化学习把在线搜索经验内化到代码模型，使其直接合成面向问题类的可复用求解器。 |

## 阅读这组方法时的统一问题

不同方法的表面结构差异很大，但可以用四个问题统一比较：

1. **设计对象是什么**：单个函数、启发式集合、算法组件还是完整求解器；
2. **保存什么历史**：当前种群、祖先路径、反思摘要、程序差分还是共享记忆；
3. **怎样分配预算**：精英选择、Pareto 选择、树搜索、bandit、规划模型还是智能体策略；
4. **LLM 怎样改进**：固定提示生成、反思后生成、提示自适应、模型参数更新还是多智能体协作。

上述分类描述各论文的主要机制，不表示同类方法具有相同效果，也不把联合系统的实验结果归因于单个组件。跨方法比较仍需统一 evaluator、评价预算、模型条件和独立测试协议。
