# EoH（Evolution of Heuristics）

- 论文：*Evolution of Heuristics Towards Efficient Automatic Algorithm Design Using Large Language Models*；本地来源：[`main.tex`](../../../../papers/Evolution_of_Heuristics_Towards_Efficient_Automatic_Algorithm_Design_Using_Large_Languag/main.tex)；设计对象：组合优化问题的启发式函数/代码。

## 1. 核心问题与方法

EoH 将 LLM 作为启发式代码的初始化与变异/交叉生成器，并用种群演化选择。个体是"自然语言思想 + 代码"二元组；五种生成策略按原文分组为 **Exploration（E1 相异重组、E2 共同思想扩展）与 Modification（M1 结构修改、M2 调参、M3 删冗余）**——论文以 Modification 命名后三者的修改功能，未用 exploitation 一词。父代按秩反比 $p_i\propto 1/(r_i+N)$ 软选择，种群按适应度截断 top-N；除 E1/E2 的指令多样性外无任何多样性维护结构。目标是以较少人工特征工程在多个优化任务得到可执行 heuristics。

## 2. 论文宣称的机制贡献（逐项）

- 用自然语言角色/指令把启发式知识注入初始化。
- 探索与开发算子并用，平衡新思路与高分代码修订。
- 代码级演化将 LLM 创造性与黑箱评价结合。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|EoH 在所选 CO 任务可产生竞争性启发式|§4 `tab:bin_pack`、`tab:tsp`、`tab:fssp`|间接支持|这是整法与基线的比较，支持论文任务、实例与 evaluator 配置下的联合系统表现。|
|探索—开发组合有效|§4.3 Ablation Study，`tab:versions`、`tab:bin_pack_ablation`|部分支持|增量式消融（EoC 仅代码+E1 → +thought → +E2 → +M1/M2/M3）显示 thought 表示 > E2 交叉 > M 组逐层累积正贡献；非全因子分解，限于 Weibull 设置（3 runs）。|
|LLM 初始化优于随机/人工起点|论文未报告随机初始化对照|未验证|不能将 EoH 主结果归因给初始化。专家启发式注入实验（FunSearch 启发式放入初始种群得 0.55 < EoH 0.66）支持精英知识可被继承继续进化。|
|每种 prompt 均有独立因果作用|未见完整全因子消融|未验证|不能凭最终 EoH 优势归因给单个提示。|

## 4. 机制的底层逻辑

阅读分析：EoH 将“算法想法”压缩为代码候选，evaluator 把语言的流畅性筛为任务目标。探索 prompt 对抗 LLM 在单一高分程序附近复述，开发 prompt 则让模型利用可见父代结构。两者能否真平衡，取决于种群更新、调用配额和实例泛化；若只在训练实例选优，开发会强化 evaluator exploit。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：明确区分探索和改良的生成意图。前提：实际采样预算也分开记录。风险：标签不同而输出分布无差异。最小验证：对输出 diff/行为距离和收益作分组统计。
- 可学习点：只让 evaluator 决定存活。前提：训练与测试实例隔离。风险：生成代码投机固定 evaluator。最小验证：每个新增 best 立即 held-out 重测。

## 6. 证据边界

论文的比较受任务集、基线实现、LLM 与调用预算约束；`fig:evolution` 明示 population=10、20 generations，但 `tab:versions` 不是全因子 prompt 消融，且正文未报告可用于所有比较的置信区间/显著性检验（主实验多数单 run，消融 3 runs）。代码有效率与最终质量也是不同结论。LLM 消融（GPT-3.5 0.66 < Gemini Pro 0.71 < CodeLlama 1.07 < Deepseek 1.41；任何 LLM 的 EoH 2000 查询均优于 GPT-3.5 随机采样 10000 次的 2.44）显示进化回路本身的收益大于换更强 LLM。

## 7. 论文内定位

入口 `main.tex`，依次 include [`3-method.tex`](../../../../papers/Evolution_of_Heuristics_Towards_Efficient_Automatic_Algorithm_Design_Using_Large_Languag/3-method.tex)、[`4-experiment.tex`](../../../../papers/Evolution_of_Heuristics_Towards_Efficient_Automatic_Algorithm_Design_Using_Large_Languag/4-experiment.tex)、[`7-appendix.tex`](../../../../papers/Evolution_of_Heuristics_Towards_Efficient_Automatic_Algorithm_Design_Using_Large_Languag/7-appendix.tex)。
