# MeEvo

- 论文：MeEvo: Metacognitive Evolution for Automatic Heuristic Design；本地来源：[main.tex](../../../../papers/MeEvo_Metacognitive_Evolution_for_Automatic_Heuristic_Design/main.tex)；设计对象为可执行启发式程序的进化。

## 1. 核心问题与方法

MeEvo 在常规“生成—评价—选择”上加入元认知反思：从表现、差异或失败中形成高层改进判断，再以该判断引导交叉/变异或下一轮生成。重点是把演化过程本身作为可被 LLM 观察与调节的对象。

## 2. 论文宣称的机制贡献（逐项）

1. 用元认知经验提升改进的针对性。
2. 让经验驱动的演化兼顾探索与利用。
3. 在 AHD 基准上超过人工、进化或 LLM 基线。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|整体性能|§Experiments，表 `tab:comparative`（TSP-ACO、BPP-ACO、ACS、WSN）、`tab:tsp_construct`，图 `fig:fitness`|间接支持|支持系统比较，不能把改进分配给元认知组件。|
|元认知组件有效|本地 `main.tex` 未检索到明确、可定位的受控消融 label|未验证|不以主结果或过程图补足组件因果。|
|搜索更有方向|反思示例或过程图|间接支持|可解释性案例不等于因果证据。|

## 4. 机制的底层逻辑

阅读分析：它把历史从“候选集合”转换成“选择何种改动”的语义状态。相较只按 fitness 选父代，这可能减少无效重试；代价是元判断由同一 LLM 生成，易受提示、评价噪声和事后解释影响。

## 5. 对 LLM4AD / TraceAAD 可学习之处

|可学习点|成立前提|主要风险|最小验证方式|
|---|---|---|---|
|把改动意图与实际代码差分绑定|能取得真实 lineage|意图与改动不一致|抽样审计意图—差分—分数三元组。|
|对失败也保留可检索摘要|失败有可靠错误/分数信号|负经验过度压制新颖性|只屏蔽重复失败模式，测新颖候选率。|

## 6. 证据边界

主结果是组合系统证据。若论文未报告独立测试、方差/显著性或完整调用成本，则不能把一次最好程序、训练 evaluator 分数或曲线末端当作机制普适性证明。

## 7. 论文内定位

入口：[main.tex](../../../../papers/MeEvo_Metacognitive_Evolution_for_Automatic_Heuristic_Design/main.tex)。本次依据其中 Methodology、Experiments 与表图实际内容；未发现可据以确认的组件消融 label，故保留“未验证”。
