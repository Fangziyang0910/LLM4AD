# EvoTune

- 论文：*Algorithm Discovery With LLMs: Evolutionary Search Meets Reinforcement Learning*；本地来源：`../../../../papers/Algorithm_Discovery_With_LLMs_Evolutionary_Search_Meets_Reinforcement_Learning/colm2025_conference.tex`；设计对象：算法程序及生成它们的 LLM 策略。

## 1. 核心问题与方法

EvoTune 将 evaluator 驱动的进化搜索与 RL 微调结合：进化产生、筛选高质量程序，同时将搜索获得的信号转为对 LLM 生成策略的训练，使后续采样偏向有用算法。主文的 Method 图和 TSP、bin packing、flatpack 曲线描述该闭环；这不是纯 inference-time 搜索，也不是仅离线微调。

## 2. 论文宣称的机制贡献（逐项）

- 搜索发现与参数学习相互供给：进化提供候选，RL 改变生成分布。
- 用任务 reward/程序表现而非人工标签训练算法生成。
- 在多类组合任务检验算法发现。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|联合方法在任务上改进|§`sec:results`，Table `table:best_50` 与 Fig. `fig:results_horizontal`|间接支持|完整方法相对 evolution-only FunSearch 同时包含 RL 更新与其数据/训练配方，不能单独归因给 RL。|
|RL 使搜索更有效|Table `table:best_50`（三任务、三模型、10 seeds，FunSearch 不训练 LLM）|部分支持|这是 evolution-only 基线与联合方法的直接系统比较；并非“关闭 RL 但其他训练/数据路径完全相同”的匹配单组件消融。|
|程序分布发生有益变化|Fig. `fig:combined`(a)，Appendix Figs `appendix:fig-pdb-hist-bin/tsp/fp`|间接支持|分布变化不等于因果地带来最终质量。|
|RL 算法选择的影响|Appendix §`app:rest`、Fig. `fig:rest-em`|部分支持|Granite/bin-packing、三学习率下比较 DPO 与 ReST-EM；仅支持该 RL 更新选择。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

进化负责在当前分布下提供高 reward 样本，RL 把局部发现写入模型参数，理论上可摊销后续搜索。危险是 on-policy 数据高度选择性：模型可能记住可执行模板或评价器漏洞；若测试仍接近训练分布，参数改进不等于可迁移的算法知识。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：区分“轨迹内推理/检索改进”和“把经验写进模型参数”。前提：后者有严格 held-out 评估。风险：高训练成本和灾难性遗忘。最小验证：固定模型权重，仅比较搜索历史利用；再单独测试微调模型的 pass@k 和 OOD。
- 可学习点：保存产生训练信号的程序 lineage。前提：reward、代码与过滤规则可重放。风险：survivorship bias。最小验证：报告入池、过滤、训练三阶段的数量和质量分布。

## 6. 证据边界

主比较为 Llama3.2-1B、Phi3.5-mini、Granite3.1-2B，bin packing/TSP/flatpack，10 random seeds；每个预算为 9.6k、16k、22.4k sampled programs，且报告 validation、validation-perturbed、同分布 test（§`sec:results`、Table `table:best_50`）。论文没有 evolution-only 的受控“RL off”消融，因此不能将完整联合优势进一步分给 RL、进化数据库或 prompt 构造；曲线/分布只是过程证据。

## 7. 论文内定位

`colm2025_conference.tex`：Method 图 `images/Method.pdf`，实验节；过程图 `tsp_avg_reward.pdf`、`bin_avg_reward.pdf`、`flatpack_avg_reward.pdf`、`pdb_dist_example_bintspflatpack_3x2.pdf`、`Hashcode_and_LLMSR_results.pdf`。
