# BaSE：Compute Allocation in Evolutionary Search

- 论文：Compute Allocation in Evolutionary Search: From Depth, Breadth to Multi-Armed Bandits；本地来源：[main.tex](../../../../papers/Compute_Allocation_in_Evolutionary_Search_From_Depth_Breadth_to_Multi_Armed_Bandits/main.tex)、[bandit.tex](../../../../papers/Compute_Allocation_in_Evolutionary_Search_From_Depth_Breadth_to_Multi_Armed_Bandits/bandit.tex)、[bandit_appendix.tex](../../../../papers/Compute_Allocation_in_Evolutionary_Search_From_Depth_Breadth_to_Multi_Armed_Bandits/bandit_appendix.tex)；设计对象是固定 LLM 调用预算下的自演化轨迹调度。

## 1. 核心问题与方法

BaSE 将一次独立自演化 run 当作 bandit arm。每个 arm 有当前深度和 fitness，调度器每次决定哪条轨迹获得下一次 refinement，而不预先固定所有 run 的深度。由于 arm 回报随 refinement 非平稳，论文用近期趋势预测下一步 reward，再比较 UCB、EXP3.P、Thompson 等分配；父代采样仍是被选轨迹内部的独立问题。

## 2. 论文宣称的机制贡献（逐项）

1. 在固定预算下刻画 breadth–depth 分配及任务相关 fitness surface。
2. 用非平稳 MAB 自适应把调用投向更有潜力的轨迹。
3. 在不改 base model、evaluator、prompt 的条件下改善最终 fitness 和达阈值效率。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|固定分配的 breadth/depth 权衡|`bandit.tex` 的 depth–breadth 实验与 fitness 表|直接支持|支持所测任务/模型/总调用预算中的经验表面，非普适定律。|
|BaSE 整体优于 *Evolve 基线|§Our Results，`tab:fitness_scores`、`tab:threshold_iteration_flops`|间接支持|是 arm 池、预测奖励、bandit 策略等联合系统效果。|
|arm pool 大小|§Our Results，`tab:bandit_ablation_arms`|直接支持|目标变量 K 的消融支持中等 K 的局部折衷。|
|与不同父代采样兼容|Appendix §Pairwise Fitness Comparisons，`tab:pairwise_fitness`|部分支持|固定同一 prompt generator 的配对比较支持若干组合；不是所有任务/采样策略必增益。|
|阈值效率|`tab:threshold_iteration_flops`|部分支持|采用 bootstrap 90% 达阈值，支持报告条件下的速度；阈值选择会影响结论。|

## 4. 机制的底层逻辑

阅读分析：BaSE 处理的是跨轨迹资源分配，不决定单轨迹怎样改程序。把历史平均 reward 直接用于 arm 选择会偏爱早熟路线，所以预测近期趋势；但趋势外推面对 LLM 突变和 evaluator 噪声可能失准。它不能替代 TraceAAD 的路径语义/生成指导，只是可与其并列的预算调度层。

## 5. 对 LLM4AD / TraceAAD 可学习之处

|可学习点|成立前提|主要风险|最小验证方式|
|---|---|---|---|
|将路线级预算分配与路线内生成分开|每条路线有独立状态与调用账本|误把调度收益归为生成机制|固定生成器，比较 equal split、随机、bandit 的同总调用结果。|
|用近期增益而非历史均分数估计潜力|评价尺度稳定|趋势追逐偶然噪声|在日志中检验预测增益与实际下一步增益的相关性。|

## 6. 证据边界

论文的 CI 说明位于 appendix：许多基线用 seed trajectory bootstrap，BaSE 则是在固定的 greedy pool 上运行 10 次 MAB 并 bootstrap；两者随机性层级不同，比较时须谨慎。主文主张总 LLM-call 固定，但 token、wall-clock、evaluator 代价仍可能异质；“最快达阈值”还依赖预选 $\tau$，不能与最终 best 混为同一证据。

## 7. 论文内定位

入口：[main.tex](../../../../papers/Compute_Allocation_in_Evolutionary_Search_From_Depth_Breadth_to_Multi_Armed_Bandits/main.tex)。主体：[bandit.tex](../../../../papers/Compute_Allocation_in_Evolutionary_Search_From_Depth_Breadth_to_Multi_Armed_Bandits/bandit.tex)，§Our Results、`tab:fitness_scores`、`tab:threshold_iteration_flops`、`tab:bandit_ablation_arms`；附录：[bandit_appendix.tex](../../../../papers/Compute_Allocation_in_Evolutionary_Search_From_Depth_Breadth_to_Multi_Armed_Bandits/bandit_appendix.tex)，§Pairwise Fitness Comparisons、`tab:pairwise_fitness`、§Full Threshold Comparison。
