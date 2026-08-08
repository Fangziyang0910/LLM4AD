# EoH-S

- 论文：*EoH-S: Evolution of Heuristic Set using LLMs for Automated Heuristic Design*；本地来源：[`eohs.tex`](../../../../papers/EoH_S_Evolution_of_Heuristic_Set_using_LLMs_for_Automated_Heuristic_Design/eohs.tex)；设计对象：一组协作/组合使用的启发式，而非单一启发式。

## 1. 核心问题与方法

EoH-S 指出单一启发式通常只能覆盖问题的一部分状态或实例分布，因此演化对象扩展为 heuristic set：LLM 提议集合成员及其组合，evaluator 按集合的整体求解表现评分，演化同时处理成员质量与集合互补性。它把自动设计从“找一个 best rule”改成“找可分工的规则库”。

## 2. 论文宣称的机制贡献（逐项）

- 启发式集合可利用成员间互补性，应对单规则的偏置。
- LLM 生成使集合成员能带来不同策略思路。
- 结合/选择机制把成员质量转为集合级性能。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|EoH-S 在文中任务的集合性能|§Experimental Studies，`table:main_obp`、`table:main_tsp_cvrp`、`table:benchmark`|间接支持|整体法与基线比较，不能单独证明“集合互补”机制。|
|集合优于单一启发式|`table:main_obp`、`table:main_tsp_cvrp`（每法三次、报平均）|间接支持|这仍是集合系统的联合比较；表中没有令单启发式与集合严格同调用成本的机制隔离。|
|成员互补驱动增益|§Complementary Performance，`fig:comparison`；附录 `convergence_obp`、`convergence_cvrp_eohs`|部分支持|CPI 和实例级最佳成员雷达图直接测补充性表现，但不独立证明它造成主性能增益。|
|LLM 产生了真正不同的策略|代码案例|间接支持|文本和代码差异不等于行为分工。|

## 4. 机制的底层逻辑

阅读分析：集合的价值来自条件分工：不同启发式在不同状态/实例赢，组合器才可将局部优势转成全局收益。若 evaluator 只回传集合总分，成员信用分配会变得模糊，成员可能只是冗余备份；集合增益也可能来自更多候选调用，而非互补。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：把“多条历史路线”当作可条件选择的库。前提：有可靠的选择条件。风险：库只增加 token 和评估开销。最小验证：固定总生成数，比较单候选、随机集合与条件集合。
- 可学习点：为每个成员记录何时被采用。前提：运行时可观测。风险：只看总分无法信用分配。最小验证：成员 leave-one-out 及实例级胜率。

## 6. 证据边界

集合大小、成员生成成本、组合器、评估实例和种子都会影响结果；没有集合大小匹配的预算对照与成员消融时，不能把主表优势解释为互补性。集合对未见实例是否泛化还须独立测试。

## 7. 论文内定位

入口：[`eohs.tex`](../../../../papers/EoH_S_Evolution_of_Heuristic_Set_using_LLMs_for_Automated_Heuristic_Design/eohs.tex)。使用 §EoH-S 的 `fig:framework`、§Experimental Studies 的 `table:main_obp`、`table:main_tsp_cvrp`、`table:benchmark`、`fig:comparison`、`tab:ablation`，以及附录 `convergence_obp`、`convergence_cvrp_eohs`。
