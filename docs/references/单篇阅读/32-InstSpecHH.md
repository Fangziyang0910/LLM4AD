# InstSpecHH

- 论文：*LLM-Driven Instance-Specific Heuristic Generation and Selection*；本地来源：`../../../../papers/LLM_Driven_Instance_Specific_Heuristic_Generation_and_Selection/main.tex`（含 `section/framework.tex`、`section/experiments.tex`、`supp_content.tex`）；设计对象：按实例生成并选择启发式，而非一个全局规则。

## 1. 核心问题与方法

该文关注同一 COP 内实例结构不同，单一全局启发式会失配。框架为 offline 生成 / online 选择闭环：按实例子类（CVRP 675 = 472 intra + 203 inter；摘要写 365，与实验节矛盾）生成候选启发式（特征向量 + 自然语言双表示），运行前用欧氏距离预筛 + LLM 或两层 NN 分类器为新实例挑选。**Neighbor Search（NS）是关键机制**：对每个子类取 $k_n=20$ 个最近邻子类的已有启发式入池复评取优（$k_n$ 从 2→20 单调改善、20 附近饱和，Fig. `fig:ns_analysis`）；论文明确 RQ3 中胜过单实例 EoH 主要由 NS 驱动——w/o NS 时 EoH (Individual) 反超，CVRP intra 上 w/o NS（7.72）差于 EoH（5.11，Table `tab:intra_main`）。基线模型不对称：EoH/ReEvo 用 DeepSeek-V3，本方法用 DeepSeek-R1-Distill-Qwen-14B（控制逐子类生成成本）。

## 2. 论文宣称的机制贡献（逐项）

- 设计目标从全局 heuristic 扩展到 instance-specific portfolio。
- 把"生成"与"对当前实例的选择"组成闭环，并用 NS 在邻域子类间迁移已有启发式。
- 用实例特征/描述使 LLM 生成针对性规则。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|实例定制方案在 CVRP、OBPP 有效|§`sec:experiments` Tables `tab:intra_main`、`tab:inter_main`（两表含 w/o NS 消融行；inter 表并列 Random/Closest/LLM/Classifier 四种选择策略）|部分支持|NS 与选择器的贡献已被部分拆开：去 NS 后单实例 EoH 反超（RQ3 的优势主要由 NS 驱动）；仍不能拆的是"逐子类生成范式"整体。|
|选择策略与候选数的作用|§Sensitivity Analysis，Fig. `fig:topk`；inter-subclass 候选数为 LLM 选择 k_c=3（OBPP）/2（CVRP）、classifier 选择 k_c=5（OBPP）/2（CVRP）（Table `tab:inter_main` caption 与 `tab:setting`）|部分支持|候选数变大时 classifier 更稳、LLM 选择方差增大；LLM 选择平均仅约 1.01 次 query。|
|候选确有结构差异|`images/code_*` 的案例|间接支持|代码例子可说明可解释差异，不能证明差异是性能原因。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

阅读分析：实例专属的收益有两个来源且被 NS 混杂——"为该子类专门生成的规则"与"从邻域子类迁移来的已验证规则"，论文自己的消融显示后者（NS）是主要来源。选择器承担把昂贵代码搜索变成 portfolio 决策的工作；若同一实例既用于选规则又用于最终得分，容易形成选择偏差。TSPLIB 类跨分布表（MoH 同款口径）若用逐实例挑选策略，评价的是 portfolio 而非单一启发式泛化。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：记录轨迹节点适用的实例特征与失败条件。前提：特征和训练/测试拆分明确。风险：记录变成事后标签。最小验证：固定候选池，比较特征条件选择与随机选择。
- 可学习点：把“搜索到的程序”和“何时采用”分开。前提：选择器不访问测试结果。风险：每实例搜索掩盖在线成本。最小验证：报告生成、选择、求解三段成本。
