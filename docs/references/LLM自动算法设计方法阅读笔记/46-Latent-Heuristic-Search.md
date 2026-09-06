# Latent Heuristic Search

- 论文：*Latent Heuristic Search: Continuous Optimization for Automated Algorithm Design*；本地来源：`../../../../papers/Latent_Heuristic_Search_Continuous_Optimization_for_Automated_Algorithm_Design/paper.pdf`；设计对象：由 latent soft prompt 解码的启发式程序。

## 1. 核心问题与方法

LHS 针对离散代码空间难以计算梯度的困难，提出潜空间连续优化：
1. **冻结组件**：预训练代码编码器（UniXcoder/Embedding 模型）与代码生成 LLM 保持冻结，不进行参数微调；
2. **训练组件**：仅联合训练 Normalizing Flow、Latent-to-Soft-Prompt Mapper 以及任务性能 Surrogate 回归器；
3. **两阶段优化**：
   - **潜空间连续梯度搜索（零 evaluator 成本）**：在高斯化潜空间中，直接利用可微 Surrogate 模型的梯度引导潜向量沿预测性能上升方向移动若干步（不调用昂贵真实 evaluator）；
   - **解码与真实评估**：将优化后的潜向量通过 Mapper 映射为连续 soft prompt，注入冻结 LLM 解码生成离散程序代码，最后提交外部真值 evaluator 进行真实性能评价。

## 2. 论文宣称的机制贡献（逐项）

- 把离散程序空间变换为可微连续 latent 空间进行基于梯度的搜索。
- Normalizing Flow 缓解预训练代码嵌入空间的各向异性与低密度空洞。
- Surrogate 梯度提供比纯几何插值（No-Grad）更有方向性的演变路径。

## 3. 实验究竟支持了什么

|机制主张|论文证据|证据等级|判断|
|---|---|---|---|
|完整 LHS 在 routing 问题上有竞争力|Tables 1–4|直接支持（整法有效性）|匹配协议下的整法比较直接支持 LHS 方案在所测 TSP/CVRP 路径问题上优于基线；不能顺带证明 Flow、Surrogate 或潜空间优化各自独立有效（组件消融见 Table 5），且在 Knapsack/OBP 等非路径问题上未展现统一领先。|
|Flow 提高有效解码率与解质量|§4.4、Table 5，LHS 对比 No-Flow|直接支持|在相同评估预算下，可运行率从 61% 提升至 74%，目标值显著改善。|
|Surrogate 梯度优于纯潜空间插值|§4.4、Table 5，LHS 对比 No-Grad|直接支持|LHS 目标值（6.61）优于纯插值（6.79），但有效解码率（74%）低于插值（86%），体现质量与语法稳定性的权衡。|
|各训练辅助组件各自独立必要|§3、Appendix A|部分支持|Flow 与梯度（Surrogate）有受控消融；Mapper 架构未见独立隔离消融。注：encoder 与 LLM 均为冻结模型，无训练证据。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

Flow 先把预训练 encoder 产生的不规则代码嵌入映射到更规整的高斯先验流形，Surrogate 才能在其上提供相对平滑的局部梯度方向。**关键机制区分**：必须严格区分内层潜空间梯度步骤（纯神经网络前向与反向，不耗费真值 evaluator 次数）与外层解码候选的真实评价（消耗 AAD 任务评估预算）。在远离数据覆盖的潜空间区域，Surrogate 极易发生假性上升（false ascent，预测性能极高但解码出的代码语法崩溃或表现恶化），因此连续可微代理并不意味着真实适应度景观变得简单。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- **代理模型与真实评价的分工**：利用可微 surrogate 或启发式评分提供密集的内层候选筛选是可行的，但不能把内层梯度步数当成 evaluator 次数，最终必须由真实执行评估把关。
- **避免混淆参数学习对象**：明确冻结基座模型、仅训练适配器/流模型/代理模型的边界；TraceAAD 若探索表示学习，应优先保持生成代码 LLM 固定，隔离训练变量。
