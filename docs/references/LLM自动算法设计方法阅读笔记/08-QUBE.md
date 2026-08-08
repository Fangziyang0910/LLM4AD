# QUBE

- 论文：*QUBE: Enhancing Automatic Heuristic Design via Quality-Uncertainty Balanced Evolution*；本地来源：[`main.tex`](../../../../papers/QUBE_Enhancing_Automatic_Heuristic_Design_via_Quality_Uncertainty_Balanced_Evolution/main.tex)；设计对象：启发式算法代码。

## 1. 核心问题与方法

QUBE 针对只按当前质量选父代会过早收敛的问题，在演化选择中同时考虑质量和不确定性。LLM 提出代码候选，evaluator 给出任务分数；不确定性项用来优先探索价值尚不确定的候选/区域，形成 quality–uncertainty 的选择平衡。

## 2. 论文宣称的机制贡献（逐项）

- 不确定性补偿能避免质量选择的过度开发。
- 质量与不确定性联合选择提高启发式设计效率。
- LLM 代码生成可在该反馈下维持更有价值的多样性。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|QUBE 在文中基准的总体性能|§Main Results，`tab: main_result`、`fig: bp score`|间接支持|这是整法与基线的比较，限于其模型、实例和预算。|
|质量—不确定性优于只质量|§Ablation Study，`tab: ablate_method`|部分支持|表中给出 OR3 上的组件/设置对照，且“Best”与十次运行平均及标准差的口径明确；结论限该任务。|
|不确定性真正测到“知识不足”|估计定义与过程图|间接支持|分数波动/代理置信度未必等于 epistemic uncertainty。|
|权重设置可泛化|论文未报告跨任务的 trade-off 参数稳健性扫描|未验证|不能断言。|

## 4. 机制的底层逻辑

阅读分析：质量提供 exploitation 信号，不确定性相当于选择价值的代理，只有它与未来改善概率相关才有意义。若不确定性来自 evaluator 随机性，它会奖励噪声；若来自代码嵌入距离，它又可能奖励无效新颖性。因此该方法的关键不只是“加一项”，而是估计校准和预算下的再评估。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：将质量与探索理由分开记录。前提：探索量可计算。风险：把低质随机噪声当潜力。最小验证：分桶后比较不确定性与下一代真实改进率。
- 可学习点：对高不确定候选追加验证。前提：有可负担的复评预算。风险：吞噬搜索调用。最小验证：只复评 top-k 不确定候选并报告排序校准。

## 6. 证据边界

不确定性定义、其估计数据和超参数均是任务相关实现选择；主比较不替代对校准的实证。没有足够重复、方差和训练/测试实例分隔时，质量提升可能来自评价偶然性。

## 7. 论文内定位

入口：[`main.tex`](../../../../papers/QUBE_Enhancing_Automatic_Heuristic_Design_via_Quality_Uncertainty_Balanced_Evolution/main.tex)。使用 §Quality-Uncertainty Balanced Evolution（`eq:uiq`）、§Experiments 的 `tab: main_result`、`fig: bp score`、`tab: ablate_method`、`tab: ablate_llm`、§Limitations。
