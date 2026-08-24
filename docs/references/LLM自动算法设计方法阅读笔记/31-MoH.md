# MoH

- 论文：*Generalizable Heuristic Generation Through LLMs with Meta Optimization*；本地来源：`../../../../papers/Generalizable_Heuristic_Generation_Through_LLMs_with_Meta_Optimization/main.tex`；设计对象：生成“生成/改进启发式的优化器”，而非仅一条启发式。

## 1. 核心问题与方法

MoH 把搜索升至两层：外层当前 meta-optimizer 一次生成 M 个 heuristic-optimizer；内层每个 optimizer 生成 K 条启发式，在 N 个下游任务中选出最好者，汇总 utility 后选出下一代 meta-optimizer。LLM 生成的实现可以递归或迭代地改进优化策略。目标是让训练出的产生器在未见实例、分布或任务上仍可生成有效启发式。

## 2. 论文宣称的机制贡献（逐项）

- 以跨下游任务 utility 评价“启发式生成器”，而非单程序 fitness。
- 外层元优化递归改进搜索策略。
- 通过跨问题/分布实验追求泛化。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|在 TSP、在线 BPP 上有竞争力|Tables `tsp_heu`、`bpp_table`，§Empirical Result（口径为三次独立运行中最好启发式在测试集上的平均，best-of-3 而非三次均值）|部分支持|支持报告设置下的端到端表现，不分离两层选择的作用。|
|跨分布/跨问题泛化|Appendix Tables `tab:crossprob`、`tab:clustertsp`、`tab:tsplib_small/large`|部分支持|`tab:tsplib_*` 采用逐实例从各尺度最优启发式中挑选的策略（instance-wise selection，所有方法同口径），评价的是 portfolio 而非单一启发式泛化；`tab:crossprob`（5 次运行均值±标准误）无任何基线对照，论文 FAQ 自认只是初步实验、不直接跨问题泛化单一启发式。|
|自然语言 ideas 有用|Table `ablation`（w/o ideas）、Fig. `fig:ablation_fig`（种群规模 1/5/10）|直接支持|该表只消融两件事：w/o ideas 与种群规模。|
|元优化本身有用|与固定 EC 基线（EoH/ReEvo/HSEvo/MCTS-AHD）的整系统比较|间接支持|没有"固定 seed optimizer、不做外层进化"的变体；未把外层候选数 M、内层 best-of-K 与总预算逐因素分离。|
|模型选择稳健|Table `LLM_table`|间接支持|只是一组 backbone 比较，不能推出机制与模型无交互。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

用多下游任务聚合分数，相当于把“能否继续产生有用变体”作为适应度，可能抑制只适合某一实例的投机规则。代价是外层选择面对更高方差：一个 optimizer 的 utility 同时受 K、N、下游任务难度和 best-of-K 极值效应影响。若训练任务相似，元优化也可只学到任务家族的提示偏好。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：把历史轨迹中的“下一步改法”与最终程序分开评估。前提：跨实例评估固定且可复核。风险：元层预算吞噬程序探索。最小验证：固定总 evaluator 次数，比较直接演化和一次元层控制。
- 可学习点：显式报告 ID/OOD 迁移。前提：训练与测试实例严格隔离。风险：把相近分布称为泛化。最小验证：先做一个已声明分布偏移的 held-out split。

## 6. 证据边界

主实验为 TSP、BPP，并补 CVRP/离线 BPP 和 TSPLIB；表格所报口径随任务改变（主表 best-of-3，跨问题表 5 次均值）。optimizer utility 按任务规模加权（$w_i\propto s_i$，Eq. 2）。论文有消融和收敛曲线，但未把外层候选数、内层 best-of-K 与搜索总预算（T=10、种群 10、1000 次启发式评估）逐因素分离；附录另有同预算与 Concorde/OR-Tools/LEHD/SIL/NeuOpt 的比较、八 LLM 全面比较与统计显著性分析。结论自认外层与多任务带来额外计算开销。
