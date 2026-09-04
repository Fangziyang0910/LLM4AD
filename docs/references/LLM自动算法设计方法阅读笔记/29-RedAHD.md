# RedAHD

- 论文：*Reduction-Based End-to-End Automatic Heuristic Design with Large Language Models*；本地来源：`../../../../papers/RedAHD_Reduction_Based_End_to_End_Automatic_Heuristic_Design_with_Large_Language_Models/main_neurips.tex`；设计对象：从问题描述到可执行启发式的端到端 COP 设计。

## 1. 核心问题与方法

论文认为既有 LLM-EPS 把搜索限制在人工给定的通用算法框架（GAF）内。RedAHD 先让设计 LLM 生成语言约简：原问题 A 到候选问题 B 的实例映射 f、解映射 g、B 的文字描述和代码模板；按关联启发式 top-l 的平均 fitness 选择约简。随后在多个 B 上共用一个 LLM-EPS 种群：可用任一 B 的父启发式作为另一个 B 的算法参考；当某约简得分停滞时，LLM 改写 f、g 和模板，只保留改进。论文默认以 EoH 为底层，也展示了与其他 LLM-EPS 的关系。

## 2. 论文宣称的机制贡献（逐项）

- 把设计对象由固定框架中的函数扩展为“约简＋该约简下的启发式”。
- 多问题 LLM-EPS 允许跨约简转移算法思想，以扩大可探索空间而不增加每代候选数。
- 停滞触发的约简精炼，防止某一初始约简垄断种群。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|端到端约简设计能与固定 GAF 方法竞争|§Experiments，Tables `tab:main-sbs`、`tab:main-aco`（三次运行均值）与 `tab:tsp-sbs`（三次运行中最好的启发式，且该启发式因随机选起点对每实例再跑 3 次取均值——best-of-3 口径），六类 COP|间接支持|支持该配置在所列白盒/黑盒任务的整法结果；比较含文献复用结果，不能单独归因给约简。|
|约简精炼有益|Table `tab:ablation-reduc-refi`|直接支持|该消融改变精炼步骤，能检验其对报告任务的影响；不是对所有约简或预算的普遍证明。|
|多问题交叉参考有益|Appendix Table `tab:ablation-reduc`（M=1 vs M=3）|直接支持|支持该实验配置下的跨问题 LLM-EPS；Figure `fig:demo` 只是一个说明性个案。|
|模型与底层 EPS 影响结果方向|Tables `tab:ablation-llm`、`tab:ablation-llmeps`|部分支持|换 o3-mini 后 OBPP/CVRP 显著改善（CVRP OOD 13.516 优于 OR-Tools）；RedAHD[MEoH] 最优、RedAVO[ReEvo] 略差——证据方向为"更强模型/更强 EPS 更好、框架可移植"。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

约简实际改变了“什么程序可被说出来”的表示空间；跨约简父代把一个表示中的局部结构带入另一个表示，可能提供跳出局部模式的语义线索。精炼把历史的停滞当作表示失配信号。但约简得分来自少量训练实例和 top-l 启发式，因而也可能奖励偶然可用、却不可泛化的映射；仅以最终解合法性验证 f、g，不能证明映射保留了问题结构。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：把“表示/任务重述”当作可搜索对象。成立前提：映射、评估和原任务解回译能审计。主要风险：扩大对象同时混入不可执行或伪改进路径。最小验证：固定搜索预算，比较原表示与一个可验证重述，并记录合法率、训练/测试差距。
- 可学习点：停滞后改变搜索表述。成立前提：停滞不是 evaluator 噪声。主要风险：把预算花在重写而非改进。最小验证：只替换一个停滞分支，和同预算继续搜索作配对比较。
