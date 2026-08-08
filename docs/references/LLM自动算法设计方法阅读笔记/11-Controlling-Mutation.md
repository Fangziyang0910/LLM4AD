# Controlling the Mutation in LLMs

- 论文：*Controlling the Mutation in LLMs for Efficient Evolution of Algorithms*；本地来源：[`samplepaper.tex`](../../../../papers/Controlling_the_Mutation_in_LLMs_for_Efficient_Evolution_of_Algorithms/samplepaper.tex)；设计对象：LLM 生成的算法代码变异。

## 1. 核心问题与方法

论文关注 LLM 变异幅度不可控：太小会重复父代，太大则破坏可执行结构。它通过变异指令/提示、候选差异度和质量反馈调节修改强度，在演化循环中寻找可用的变异尺度；目录中的 `ratio-mutation`、`code-diff-mutation` 等图表对应此问题。

## 2. 论文宣称的机制贡献（逐项）

- 可控制的变异幅度改善效率与有效候选率。
- 代码 diff/相似度能反馈变异是否过强或过弱。
- 不同 prompt 的变异倾向可被比较并利用。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|不同控制设置产生不同变异比例/代码差异|`ratio-mutation.tex`、`code-diff-mutation.tex`、`code_diff.tex`|直接支持|这些是目标量的测量，支持控制确实改变输出。|
|控制设置提高搜索质量|`score.tex`、`convergence.tex` 的主曲线|间接支持|整体曲线不能排除额外 prompt、采样分布等混杂。|
|存在最优变异强度|`mse.tex`、`aggregated_plots.tex` 的比较|部分支持|仅当在固定模型、预算、任务下系统扫描才支持该范围。|
|代码距离是有效的探索代理|差异图与分数关联|间接支持|文本 diff 与行为差异未必一致。|

## 4. 机制的底层逻辑

阅读分析：有效演化需要“保留可用因果结构”与“逃离当前盆地”同时发生，代码距离只是二者的廉价代理。控制 prompt 直接改变模型的条件分布，故它可能提升语法有效率；但若 evaluator 偏好微调，得到的最佳幅度会是 evaluator 特有的，而非普适规则。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：记录目标变异强度与实际 diff/行为变化。前提：代码版本可比较。风险：把长度当语义距离。最小验证：按 diff 桶报告有效率、训练/测试增益。
- 可学习点：根据轨迹阶段调节探索尺度。前提：阶段信号可预测收益。风险：过早收缩。最小验证：固定调度与自适应调度的等预算多种子对照。

## 6. 证据边界

各图所用任务、模型、prompt 数和评价次数须逐项读取 `samplepaper.tex`；图中趋势不等于跨基准统计结论。没有行为层的消融，不能把代码差异本身认定为机制变量。

## 7. 论文内定位

入口：[`samplepaper.tex`](../../../../papers/Controlling_the_Mutation_in_LLMs_for_Efficient_Evolution_of_Algorithms/samplepaper.tex)；图表源为 `ratio-mutation.tex`、`ratio-prompts.tex`、`code-diff-mutation.tex`、`score.tex`、`mse.tex`、`convergence.tex`、`aggregated_plots.tex`。
