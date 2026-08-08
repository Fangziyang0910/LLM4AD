# PathWise

- 论文：PathWise；本地来源：[example_paper.tex](../../../../papers/PathWise/example_paper.tex)；设计对象为以改进路径为中心的 LLM 启发式搜索。

## 1. 核心问题与方法

PathWise 不把每个候选仅视为种群成员，而保存从起点到当前程序的一条改进路径，并据路径的演化信息选择下一步生成与延续。路径承载“连续修改如何导致分数变化”的上下文，目标是在有限评价中区分值得加深的路线与应另开分支的路线。

## 2. 论文宣称的机制贡献（逐项）

1. 路径级表示保留连续演化信息。
2. 路径选择把预算投入潜力更高的改进路线。
3. 历史路径为 LLM 提供比孤立父代更有用的生成指导。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|整体结果|§Overall Results，`tab:tsp_kp_step_by_step`、`tab:aco_general_framework_merged`；曲线 `fig:evolution_same_ne_exp`|间接支持|整法比较不能分别证明图状态、agent 分工或提示内容。|
|critic 反馈|§Ablation Study，`tab:ablation-ours`：固定其余组件，移除 policy/world-model critics|直接支持|目标 critic 机制的受控消融，限 TSP 构造、GPT-5-nano(low)、5 次平均。|
|prompt 多样性|§Ablation Study，`tab:ablation-diversity-v2`：固定其余组件，移除 prompt perturbation/state shuffling|直接支持|支持该任务上两个多样性处理的局部贡献。|
|路径可解释性|方法示例/路径图|间接支持|仅说明记录形式。|

## 4. 机制的底层逻辑

阅读分析：路径把候选价值从静态 $f(program)$ 扩展为“该路线过去如何响应修改”。这有利于估计延续价值，但如果路径重建不完整、父子边缺失或不同路径的评价条件不同，路径价值会被错误归因。

## 5. 对 LLM4AD / TraceAAD 可学习之处

|可学习点|成立前提|主要风险|最小验证方式|
|---|---|---|---|
|以 trajectory 为最小记忆单元|完整 lineage 与操作记录|把长路径全部塞入上下文|固定 token 上限，比较最近边、压缩路径、无路径。|
|把路线价值与终点分数分开|有可验证的后继收益|后见之明挑选成功路径|预先定义路线评分，按未来增益验证。|

## 6. 证据边界

路径长度、调用数、评价预算和测试协议必须与最终分数一起报告。没有父子 ID 时，任何后验路径重建均是近似，不能作为论文已证明的信用分配证据。

## 7. 论文内定位

入口：[example_paper.tex](../../../../papers/PathWise/example_paper.tex)。方法 `sec:method`、`sec:method-graph`、`sec:method-critics`、`sec:method-evolution`；主表 `tab:tsp_kp_step_by_step`、`tab:aco_general_framework_merged`；消融 `sec:exp-ablation`、`tab:ablation-ours`、`tab:ablation-diversity-v2`；扩展参数消融 `tab:ablation-ours-nanw`、`tab:ablation-ours-np`、`tab:ablation-ours-imax`。
