# SMCEvolve

- 论文：*SMCEvolve: Principled Scientific Discovery via Sequential Monte Carlo Evolution*；本地来源：`../../../../papers/SMCEvolve_Principled_Scientific_Discovery_via_Sequential_Monte_Carlo_Evolution/paper.pdf`；设计对象：数学、算法效率、符号回归和 ML 研究程序。

## 1. 核心问题与方法

SMCEvolve 将程序进化解释为对 reward-tilted 分布的 SMC 近似：按递增 reward intensity 重采样父代，以多种 LLM mutation kernel 做 MH 风格接受/拒绝，再用有效样本量 ESS 自适应推进退火与停止。

## 2. 论文宣称的机制贡献（逐项）

- 从统一概率目标导出父代选择、变异与终止机制。
- 多 kernel 加自适应选择兼顾探索、有效性与混合。
- 给出有限样本复杂度界并自动判断收敛。

## 3. 实验究竟支持了什么

|机制主张|论文证据|证据等级|判断|
|---|---|---|---|
|完整系统跨域且调用较少|Tables 1–3、Figure 4|间接支持|完整配方与不同系统对比；调用、模型和后端设置仍需逐表读取。|
|自适应重采样优于 uniform/greedy|§4.2、Table 4|直接支持|固定 LLM 调用预算，两端替代均退化。|
|mutation mixture 与 Thompson 选择有益|Table 4|直接支持|单 kernel 和 uniform mixture 均低于默认。|
|宽度/深度需平衡|Table 4|直接支持|固定 N×K=16，N=8,K=2 优于两种极端。|
|理论保证精确覆盖实际 LLM kernel|§3、Appendix E、Table 5|部分支持|正式界依赖不变性/混合假设；实现用 reward-only 接受率近似，论文明确承认偏离精确 MH。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

退火把一次过强选择拆成一串相邻分布，ESS 决定何时增加选择压力；重采样负责复制有希望路线，链内 mutation 负责重新扩散。若 LLM proposal 的实际概率不可得，概率解释更多是设计原则而非完全严格采样器。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：用群体退化指标决定选择压力和停止，而非固定代数。最小验证：固定总调用比较 ESS 调度与固定温度/固定停止，并报告独特 lineage 数。

## 6. 证据边界

关键组件消融集中在 Circle Packing N=21、3 seeds；自动停止的独立质量—成本消融不足。理论所需 uniform ergodicity 等条件没有在真实代码空间实证验证。

## 7. 论文内定位

Figures 1–3；§2；Algorithm 1；§3 Theorem 3.1；Tables 1–4；Figure 4；§4.2；Appendix E、Table 5。
