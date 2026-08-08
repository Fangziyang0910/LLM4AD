# A2DEPT

- 论文：*A2DEPT: Large Language Model–Driven Automated Algorithm Design via Evolutionary Program Trees*；本地来源：`../../../../papers/A2DEPT_Large_Language_Model_Driven_Automated_Algorithm_Design_via_Evolutionary_Program_T/paper.tex`；设计对象：完整可执行 solver 程序。

## 1. 核心问题与方法

A2DEPT 从模板槽位设计转向程序空间搜索。树节点存可执行程序、分数、父代/算子历史和局部算子权重；每轮混合选择多个父代，LLM 以微调、结构编辑等程序级算子批量扩展。维护管线解析函数注册表，区分不可变定义和可变策略，补齐缺失依赖、从入口做可达性裁剪，再评估。选择将模拟退火式父子接受与 Boltzmann 抽样结合，避免只保留 top-k。

## 2. 论文宣称的机制贡献（逐项）

- 完整 solver 的 evolutionary program tree，而非固定框架组件搜索。
- 依赖修复与死代码裁剪使自由代码可执行。
- 多父代 frontier、混合选择和自适应算子调度支撑程序级探索。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|程序级 AAD 主结果优于代表性 AHD|§`sec:experiments`，Fig. `fig:framework` 后的标准/高约束 benchmark 表；摘要报告相对最强 AHD 的平均归一化 gap 降低 9.8%|间接支持|支持该完整配方；不同程序表示和维护成本共同变化，不能单独归因给程序树。|
|每个循环组件有贡献|Table `tab:ablation_unified`|直接支持|作者称移除任一组件至少在一个任务退化；这是任务条件下的组件证据，不是普适必需性。|
|维护提高可执行性|§Program Maintenance（依赖闭合/可达性裁剪）及 Appendix `app:prompts_repair`|未验证|该机制有明确实现描述，但本地论文没有把“只关闭维护、其他条件匹配”的表格标签定位为独立结果。|
|树历史优于其他控制器|收敛、预算、敏感性附录|间接支持|除非有匹配预算的控制器替换，否则只说明该控制器可用。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

程序树让“系统架构改动”有 lineage，而修复/裁剪把 LLM 的长程序不稳定性转成受限工程循环。它也改变搜索分布：不可执行、被修复和被裁剪的程序不再与原始自由生成可比。因而质量收益、可执行率收益与额外 LLM 修复预算必须分开计量。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：只有当设计对象确需跨函数控制流时才扩展到程序级。前提：入口、不可变契约和执行验证已固定。风险：复杂度压倒真实创新。最小验证：选一个小 solver，固定总调用数比较函数槽位与程序级编辑。
- 可学习点：将依赖闭合、裁剪视为评估前卫生，而非优化信用。前提：修复日志可追溯。风险：把修复能力误记作算法改进。最小验证：单列原始可执行率、修复后可执行率和修复成本。

## 6. 证据边界

论文包含六个 COP、额外 backbone/预算/OOD/资源分析，但全系统同时改变搜索对象、维护和控制器。摘要的 9.8% 是跨任务归一化 gap 聚合，不能当作每个任务或每个组件的效应；需按 §`sec:experiments` 与附录的具体 evaluator、约束、重复和 token 预算解释。

## 7. 论文内定位

`paper.tex`：§方法（约 275 行起）、Fig. `fig:framework`、§`sec:experiments`、Table `tab:ablation_unified`；附录 `app:extended_results`、`app:llm_backends`、`app:scaling_budget`、`app:ood_generalization`。
