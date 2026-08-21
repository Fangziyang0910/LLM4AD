# QUBE

- 论文：*QUBE: Enhancing Automatic Heuristic Design via Quality-Uncertainty Balanced Evolution*；本地来源：[`main.tex`](../../../../papers/QUBE_Enhancing_Automatic_Heuristic_Design_via_Quality_Uncertainty_Balanced_Evolution/main.tex)；设计对象：启发式算法代码。

## 1. 核心问题与方法

QUBE 针对只按当前质量选父代会过早收敛的问题，在 FunSearch 框架内重定义两个决策点的优先级准则。质量项是**簇的进化质量** $Q_t(C)$——簇内样本历史子代的平均分（直接估计"当亲代的价值"，与簇自身分数区分）；不确定性项是 **UCB 式访问计数** $k\sqrt{\ln t/N_t(C)}$（$N_t(C)$ 为簇内样本被选为亲代的次数）。UIQ 用于亲代选择与岛重置（按各岛最高 UIQ 簇评估存亡，替代 FunSearch 的最高分规则）。

## 2. 论文宣称的机制贡献（逐项）

- 不确定性补偿能避免质量选择的过度开发。
- 质量与不确定性联合选择提高启发式设计效率。
- LLM 代码生成可在该反馈下维持更有价值的多样性。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|QUBE 在文中基准的总体性能|§Main Results，`tab: main_result`、`fig: bp score`|间接支持|这是整法与基线的比较，限于其模型、实例和预算。|
|质量—不确定性优于只质量|§Ablation Study，`tab: ablate_method`（OR3，10 次运行平均）|部分支持|三级拆解同向：FunSearch* 分数选择 3.07% → 只换进化质量 2.98% → 只换 UIQ 选择 2.89% → 完整 QUBE（UIQ 选择 + UIQ 式重置）2.76%；"进化质量替代自身分数"与"不确定性加成"分别被隔离，结论限该任务。|
|不确定性真正测到“知识不足”|UIQ 定义（访问计数）与过程证据（Proportion of Change 随时间下降、Recent Best Score 持续增长）|间接支持|访问计数对应"被考虑的次数"，行为与 UCB 理论一致；它是选择频率的函数，与 epistemic uncertainty 是不同概念。|
|权重设置可泛化|附录 k 扫描（OR3、Weibull5k、cap set 各自的扫描曲线）|未验证|已扫描但最优 k 跨任务相差 5 个数量级，尺度耦合强，不能迁移。|

## 4. 机制的底层逻辑

阅读分析：质量项 $Q_t(C)$ 把选择信号对准"当亲代的价值"而非"自身分数"，是明确针对选择信号语义的修正；不确定性项是标准 UCB 访问计数，零额外评价成本（复用历史子代统计）。风险面在尺度耦合：最优 $k$ 跨任务相差 5 个数量级（OR3 上 0.0008、Weibull5k 0.0001、cap set 32），质量项与不确定性项的量纲关系强任务相关；若进化质量统计被 evaluator 随机性污染，它会奖励噪声亲代。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：将质量与探索理由分开记录。前提：探索量可计算。风险：把低质随机噪声当潜力。最小验证：分桶后比较不确定性与下一代真实改进率。
- 可学习点：对高不确定候选追加验证。前提：有可负担的复评预算。风险：吞噬搜索调用。最小验证：只复评 top-k 不确定候选并报告排序校准。

## 6. 证据边界

不确定性定义（UCB 访问计数）与 $Q_t(C)$ 的统计、$k$ 的取值均是任务相关实现选择；主比较在 80K–2M 样本档（岛重置周期以样本数计，TSP 用 1 岛关闭重置），与 1000 评价档结论不能互推。没有足够重复、方差和训练/测试实例分隔时，质量提升可能来自评价偶然性。换 Deepseek-coder-6.7b 结论方向不变。

## 7. 论文内定位

入口：[`main.tex`](../../../../papers/QUBE_Enhancing_Automatic_Heuristic_Design_via_Quality_Uncertainty_Balanced_Evolution/main.tex)。使用 §Quality-Uncertainty Balanced Evolution（`eq:uiq`）、§Experiments 的 `tab: main_result`、`fig: bp score`、`tab: ablate_method`、`tab: ablate_llm`、§Limitations。
