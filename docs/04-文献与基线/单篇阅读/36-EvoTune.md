# EvoTune

- 论文：*Algorithm Discovery With LLMs: Evolutionary Search Meets Reinforcement Learning*；本地来源：`../../../../papers/Algorithm_Discovery_With_LLMs_Evolutionary_Search_Meets_Reinforcement_Learning/colm2025_conference.tex`；设计对象：算法程序及生成它们的 LLM 策略。

## 1. 核心问题与方法

EvoTune 将 evaluator 驱动的进化搜索与 RL 微调结合：进化搜索是探索策略与数据收集器，RL 是策略优化器（作者引 Bitter Lesson："Search generates new data, while learning distills patterns from the data"）。数据库（6 岛，类 replay buffer）同时供 prompt 构造（in-context）与训练（in-weight）；每次 RL 更新从基模型 $\pi^0$ 重新出发，防止策略漂移累积。防探索侵蚀的机制群：forward-KL 正则的 DPO（mass-covering）、训练数据覆盖全搜索历史（非只用近期）、数据库采样百分位随时间退火（显式 explore→exploit 时间表）。偏好对来自同一 prompt 的 K=8 个输出按分数配对，正样本须超过动态 30 百分位阈值，无效程序与有效程序配成额外对（教模型避免无效输出）。

## 2. 论文宣称的机制贡献（逐项）

- 搜索发现与参数学习相互供给：进化提供候选，RL 改变生成分布。
- 用任务 reward/程序表现而非人工标签训练算法生成。
- 在多类组合任务检验算法发现。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|引入 RL 训练阶段带来搜索整体收益|Table `table:best_50`（三任务、三模型、10 seeds）与 Fig. `fig:results_horizontal`|直接支持|论文明确包含同构的无训练（FunSearch-style）进化基线，共享相同 prompt 构造、6 岛数据库与评估协议，仅关闭 RL-Update。无训练搜索基线支持加入训练阶段这一整包机制的收益；未必隔离训练阶段内部每个设计，且总训练算力需另计（模型训练消耗的额外 GPU 算力未合并核算）。|
|训练阶段内部各子设计各自独立必要|Appendix §`app:rest`、Fig. `fig:rest-em` 与正文描述|部分支持/未完全隔离|附录仅在 Granite/bin-packing 下比较了 DPO 与 ReST-EM（支持更新目标选择）；而全历史采样覆盖、百分位退火调度、每次从基模型 $\pi^0$ 重启等具体训练配方为打包引入，未报告逐项关停消融。|
|程序生成分布发生有益变化|Fig. `fig:combined`(a)，Appendix Figs `appendix:fig-pdb-hist-bin/tsp/fp`|直接支持|测量显示微调后模型的采样分布有效压缩了无效程序比例并提升了高分密度；但需注意分布变化不等于保证未知实例上的跨分布泛化。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

进化负责在当前分布下提供高 reward 样本，RL 把局部发现写入模型参数，理论上可摊销后续搜索。危险是 on-policy 数据高度选择性：模型可能记住可执行模板或评价器漏洞；若测试仍接近训练分布，参数改进不等于可迁移的算法知识。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：区分“轨迹内推理/检索改进”和“把经验写进模型参数”。前提：后者有严格 held-out 评估。风险：高训练成本和灾难性遗忘。最小验证：固定模型权重，仅比较搜索历史利用；再单独测试微调模型的 pass@k 和 OOD。
- 可学习点：保存产生训练信号的程序 lineage。前提：reward、代码与过滤规则可重放。风险：survivorship bias。最小验证：报告入池、过滤、训练三阶段的数量和质量分布。
