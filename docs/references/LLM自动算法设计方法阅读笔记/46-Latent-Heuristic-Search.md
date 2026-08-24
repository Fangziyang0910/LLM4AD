# Latent Heuristic Search

- 论文：*Latent Heuristic Search: Continuous Optimization for Automated Algorithm Design*；本地来源：`../../../../papers/Latent_Heuristic_Search_Continuous_Optimization_for_Automated_Algorithm_Design/paper.pdf`；设计对象：由 latent soft prompt 解码的启发式程序。

## 1. 核心问题与方法

LHS 训练 program encoder、normalizing flow、latent-to-soft-prompt mapper 和任务 surrogate。搜索时代码 LLM 权重固定，在高斯化 prior space 中沿 surrogate 梯度移动，再由 soft prompt 解码、执行和评价候选。

## 2. 论文宣称的机制贡献（逐项）

- 把离散代码搜索变为可微连续 latent 搜索。
- normalizing flow 缓解 encoder 空间各向异性与低密度空洞。
- surrogate 梯度提供比几何插值更明确的性能方向。

## 3. 实验究竟支持了什么

|机制主张|论文证据|证据等级|判断|
|---|---|---|---|
|完整 LHS 在 routing 有竞争力|Tables 1–4|间接支持|TSP/CVRP 最好，Knapsack/OBP 并非统一领先。|
|flow 提高有效解码率|§4.4、Table 5，LHS 对 No-Flow|直接支持|同预算下成功率 74% 对 61%，目标值也改善。|
|梯度优于仅插值的质量|§4.4、Table 5，LHS 对 No-Grad|直接支持|LHS 目标 6.61 优于 6.79，但有效率低于插值的 86%，是质量—稳定性权衡。|
|全部训练组件各自必要|§3、Appendix A|未验证|encoder、mapper 的独立消融未报告。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

flow 先把不规则程序表示映到较平滑先验，surrogate 才能用局部梯度导航；执行 evaluator 仍是最终真相源。离开训练分布后，surrogate 会产生 false ascent，因此连续可微不等于真实目标平滑。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：把历史程序映射为可优化表示，但每个 latent 步必须执行验证。最小验证：记录预测增益、真实增益、有效率随离 archive 距离的变化。

## 6. 证据边界

Table 5 消融只在 TSP；完整训练管线依赖跨任务程序库和多个学习模块。当前证据不能说明比同等训练成本下扩展离散搜索更便宜，也不能保证 surrogate 跨阶段校准。
