# Fine-tuning LLM for AAD

- 论文：*Fine-tuning LLM for Automated Algorithm Design*；本地来源：`../../../../papers/Fine-tuning-LLM-Automated-Algorithm-Design/iclr2026_conference.tex`；设计对象：为 AAD 生成候选程序的 LLM 参数。

## 1. 核心问题与方法

论文不只更换搜索器，而是用 AAD 产生的算法样本微调 LLM，再把微调模型接入随机采样、EoH 与 FunSearch 等搜索。实验考察 ASP、CVRP、TSP，比较 base/fine-tuned LLM 的候选质量、搜索收敛和 OOD；图表显示 top-1/top-10、top-5 曲线，部分结果三次独立运行取均值和标准差。

## 2. 论文宣称的机制贡献（逐项）

- 用算法设计数据把模型从通用代码先验适配到 AAD。
- 微调与 inference-time 搜索可叠加，而非二选一。
- 训练样本构造和采样方式影响泛化。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|微调改善 ASP 候选与搜索表现|§`sec:expt1`，Figs `expt1_2`、`admi_search`，Table `tab:admi_search`|直接支持|表中 top-1/top-10 以三次均值±标准差报告，支持该任务/训练配方。|
|微调改善 CVRP 的 EoH/FunSearch|§`sec:search_performance`，Fig. `cvrp_search`、Table `tab:cvrp_search`|直接支持|支持所列搜索器与预算；不等于所有 AHD 控制器。|
|能 OOD 泛化|Fig. `convergence_ood`、Table `table:ood`、§`sec:generalized_performance`|部分支持|支持论文定义的 CVRP-50 到 CVRP-100/TSP-50 等迁移；域和描述相近时仍可能有泄漏式相似性。|
|DAR sampling 优于 top-k sampling|§Sampling Method Study，Fig. `fig:expt1_1`：五种采样策略各构造 250 个 preference pairs，并比较相应微调模型生成的 1,000 个算法中 top-50 分布|部分支持|直接比较了采样配方，但训练数据内容随策略变化正是干预本身；证据限 ASP、该 DPO 配置与一次数据构造。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

微调可提高“先验采到可执行且有意义变体”的概率，令同一搜索预算更有效；搜索又保留了针对具体 evaluator 的后验纠错。它也可能缩窄生成多样性，使强基座模型在新问题上反而少探索。微调收益必须与训练算力、样本选择偏差和测试相似度分开核算。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：先把模型参数训练视为独立研究变量，不能混入控制器改动。前提：有 frozen-base 对照。风险：把样本数量收益当算法机制。最小验证：同 prompts、同 token 和同搜索预算下比较 base 与 fine-tuned。
- 可学习点：做 ID/OOD 两套报告。前提：样本来源可审计。风险：训练集程序与测试模板重叠。最小验证：以程序族去重后再构建 held-out。

## 6. 证据边界

论文明确部分主表为三次重复，但这种重复不自动覆盖所有 OOD、采样和离线结果；优化目标是对已知最优/最好算法的 gap，须注意 evaluator 口径。模型参数训练不提供“历史轨迹利用”本身有效的证据。
