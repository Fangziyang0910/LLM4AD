# TurboEvolve

- 论文：*TurboEvolve: Towards Fast and Robust LLM-Driven Program Evolution*；本地来源：`../../../../papers/TurboEvolve_Fast_and_Robust_LLM_Driven_Program_Evolution/paper.pdf`；设计对象：多岛程序进化及 warm-start seed pool。

## 1. 核心问题与方法

TurboEvolve 让一次 LLM 调用通过 Verbalized Sampling 产生 K 个带自报权重的互补 offspring（权重不用于选择）；每个岛按停滞程度在线调 K。已有 seed pool 经 embedding clustering 分配到不同岛，并用少量跨簇注入和 elite preservation 保持模式差异与可重组性。

## 2. 论文宣称的机制贡献（逐项）

- multi-offspring 提高单次上下文的信息产出。
- adaptive K 在停滞时拓宽探索、进展稳定时降低开销。
- cluster+mix+elite 的 seed injection 改善 warm start 稳健性。

## 3. 实验究竟支持了什么

|机制主张|论文证据|证据等级|判断|
|---|---|---|---|
|完整 TurboEvolve 提高样本效率与稳健性|主结果 Figures/Tables in §5|间接支持|完整 multi-island、VS、adaptive K 和 initialization 的联合结果。|
|seed allocation 优于随机|§5 RQ3、seed-pool allocation figure|部分支持|同 seed pool 比较 random、kmeans、kmeans+elite；收益随任务而变且幅度有限。|
|一次大 K 调用的头部候选更有用|§6.1 within-event top-m analysis|部分支持|控制在同一次调用内部，支持候选排序/互补现象；不是固定 K 因果消融。|
|adaptive K 本身优于 fixed K|§6|未验证|论文明确采用观察性 within-event 分析来规避在线选择偏差，没有 matched fixed-K ablation。|
|Verbalized Sampling 单独有效|Appendix A prompt diff、§6 profiles|未验证|没有只切换普通多样采样与 VS 的受控最终成绩。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

一次上下文共享父代与反馈，批量 offspring 可摊薄提示成本；岛级 K 应随局部停滞变化。seed clustering 的价值是初始路线去相关，但 embedding 簇未必对应行为簇。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：按路线停滞调 offspring 数，并报告 evaluated programs、tokens、价格三种预算。最小验证必须补 adaptive-vs-fixed K 及 VS-on/off。
