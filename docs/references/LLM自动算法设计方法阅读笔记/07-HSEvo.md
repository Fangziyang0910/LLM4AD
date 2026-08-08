# HSEvo

- 论文：HSEvo；本地来源：[`papers/HSEvo/aaai25.tex`](../../../../papers/HSEvo/aaai25.tex) 与 [`appendix.tex`](../../../../papers/HSEvo/appendix.tex)；设计对象：组合优化启发式代码。

## 1. 核心问题与方法

HSEvo 将层级搜索用于 LLM 自动启发式设计：不是只在完整代码层面平铺采样，而是把启发式的高层思想/结构与具体实现的改写分层组织，经过 evaluator 选择后继续演化。其意图是在代码搜索中同时保留结构性创新与局部调优。

## 2. 论文宣称的机制贡献（逐项）

- 分层表示让不同抽象层的改动可被分别探索。
- LLM 生成结合演化选择，能产生任务相关启发式。
- 层级过程改善探索—开发权衡。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|方法在论文 CO 基准上达到其主比较结果|§Experiments，`tab:hsevo_results`|间接支持|这是 HSEvo 整体与比较方法的结果，支持披露的任务和预算。|
|Harmony search 与 flash reflection 的作用|§Ablation study，`tab:ablation_hs`、`tab:ablation_flash_reflection`|部分支持|两张表分别给出组件消融；它们不能隔离整个框架中所有层的作用。|
|不同层贡献可解释|候选示例与过程图|间接支持|示例是事后说明，不能量化因果贡献。|
|每层都不可缺少|未见完整移除各层的组合消融时|未验证|整体结果不能拆分为各层效应。|

## 4. 机制的底层逻辑

阅读分析：层级的价值在于缩短一次 LLM 生成所需的推理跨度：高层约束候选的算法方向，低层修改实现细节。但抽象层如果不能映射到可执行、可独立评估的行为，层级只会增加 prompt 和选择噪声；它还可能把搜索锁在预设结构中。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：区分“路线/思想变更”和“代码细化”。前提：两类事件有可核验的表示。风险：人为标签掩盖实际重复。最小验证：分别统计两类变更后的 held-out 增益与行为距离。
- 可学习点：在轨迹中保存抽象决策到实现的映射。前提：生成步骤可追踪。风险：上下文膨胀。最小验证：限制轨迹长度，对比检索后有效率。

## 6. 证据边界

结论受论文列出的问题、LLM、评估样本和候选预算限制。若缺少多种子分布、统一成本报告和全因子层级消融，不能将一次最优分数解释为稳定机制优势，更不能外推至任意 AAD。

## 7. 论文内定位

入口：[`aaai25.tex`](../../../../papers/HSEvo/aaai25.tex)，补充：[`appendix.tex`](../../../../papers/HSEvo/appendix.tex)。使用 `fig:hsevo_pipline`、`tab:hsevo_results`、`tab:ablation_hs`、`tab:ablation_flash_reflection`、`fig:hsevo_di`；附录 `tab:app-problem-size`、prompt examples。
