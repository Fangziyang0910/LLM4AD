# InstSpecHH

- 论文：*LLM-Driven Instance-Specific Heuristic Generation and Selection*；本地来源：`../../../../papers/LLM_Driven_Instance_Specific_Heuristic_Generation_and_Selection/main.tex`（含 `section/framework.tex`、`section/experiments.tex`、`supp_content.tex`）；设计对象：按实例生成并选择启发式，而非一个全局规则。

## 1. 核心问题与方法

该文关注同一 COP 内实例结构不同，单一全局启发式会失配。框架让 LLM 基于问题和实例特征产生候选启发式，再经 evaluator 获得表现，并学习/使用选择机制为新实例挑选候选；论文分别讨论 CVRP 与在线 bin packing，并以代码、时间和雷达图展示结果。

## 2. 论文宣称的机制贡献（逐项）

- 设计目标从全局 heuristic 扩展到 instance-specific portfolio。
- 把“生成”与“对当前实例的选择”组成闭环。
- 用实例特征/描述使 LLM 生成针对性规则。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|实例定制方案在 CVRP、OBPP 有效|§`sec:experiments` Tables `tab:intra_main`、`tab:inter_main`|间接支持|两表比较完整系统；不能拆为生成或选择单独功劳。|
|选择策略与候选数的作用|§Sensitivity Analysis，Fig. `fig:topk`；inter-subclass 固定候选数为 OBPP 3、CVRP 2（Table `tab:inter_main` caption）|部分支持|Fig. `fig:topk` 比较不同选择策略及不同 k_c，直接观测该联合敏感性；候选数同时改变，不能给单一选择器“直接支持”。|
|候选确有结构差异|`images/code_*` 的案例|间接支持|代码例子可说明可解释差异，不能证明差异是性能原因。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

实例专属的潜在收益来自把“哪个规则好”条件化到实例；选择器承担了把昂贵代码搜索变成 portfolio 决策的工作。关键混杂是：若同一实例既用于选规则又用于最终得分，容易形成选择偏差；若实例表征不足，LLM 可能只是采样更多变体。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：记录轨迹节点适用的实例特征与失败条件。前提：特征和训练/测试拆分明确。风险：记录变成事后标签。最小验证：固定候选池，比较特征条件选择与随机选择。
- 可学习点：把“搜索到的程序”和“何时采用”分开。前提：选择器不访问测试结果。风险：每实例搜索掩盖在线成本。最小验证：报告生成、选择、求解三段成本。

## 6. 证据边界

设置在 Table `tab:setting`；intra/inter subclass 的任务划分和 candidate 数在 Tables `tab:intra_main`、`tab:inter_main`。`tab:inv_analysis` 明确把单实例 EoH 与 InstSpecHH 的在线时间分开，Fig. `fig:time_analysis` 再估算离线＋在线总成本。论文的敏感性（`fig:scal`、`fig:ns_analysis`、`fig:topk`）不是随机种子置信区间，不能据此声称稳定泛化。

## 7. 论文内定位

`main.tex` 的 `\input` 入口；`section/framework.tex`（框架），`section/experiments.tex`（主结果/消融），`supp_content.tex`（补充）；图像入口 `images/code_cvrp.pdf`、`images/code_obpp.pdf`、`images/time_*`、`images/radar_*`。
