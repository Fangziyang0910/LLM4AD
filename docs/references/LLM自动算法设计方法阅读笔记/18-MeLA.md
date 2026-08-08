# MeLA

- 论文：MeLA: Metacognitive LLM-Driven Architecture for Automatic Heuristic Design；本地来源：[main.tex](../../../../papers/MeLA_Metacognitive_LLM-Driven_Architecture_for_Automatic_Heuristic_Design/main.tex)；设计对象为 LLM 自动生成组合优化启发式。

## 1. 核心问题与方法

MeLA 将 AHD 描述为带元认知控制的迭代生成：除产生/评价候选外，模型还反思当前解法、管理经验并决定下一步改进。论文用总体架构图和 PE 图说明该“认识—调节—再生成”闭环；它试图让反馈成为显式上下文而非一次性文字。

## 2. 论文宣称的机制贡献（逐项）

1. 以元认知模块评估当前搜索状态与策略。
2. 将反思经验组织进下一次启发式生成。
3. 在不同组合优化任务中提高设计质量。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|整体 AHD 性能|实验比较段落与表 `tab:comparative`（Comparative Performance Analysis Across Different Problems）、图 `fig:fitness`、`fig:sta`|间接支持|整法比较不能单独归给元认知或经验模块。|
|元认知/经验模块|本地 `main.tex` 未检索到可定位的受控去模块表|未验证|不以架构图、主表或参数描述补足组件因果。|
|推理过程合理|`MeEvo.drawio.pdf`、`PE.drawio.pdf`|间接支持|图示描述流程，不能检验状态判断的正确性。|

## 4. 机制的底层逻辑

阅读分析：元认知的作用不是直接给代码打分，而是把“为什么继续、改变或回退”的判断引入生成条件。价值取决于状态摘要是否足以区分探索不足、停滞和评价噪声；若摘要错误，额外层会放大误导。

## 5. 对 LLM4AD / TraceAAD 可学习之处

|可学习点|成立前提|主要风险|最小验证方式|
|---|---|---|---|
|把状态判断与程序生成分开记录|判断能回连到评价事件|不可审计的高层叙事|记录判断、触发条件及随后一条边的结果。|
|只把经验证经验升格为长期记忆|经验有跨边重复支持|单次偶然改进被固化|设最小支持次数，比较升格前后命中质量。|

## 6. 证据边界

元认知、反思、经验存储和生成提示常作为联合系统出现；没有逐模块、跨任务、跨种子的充分对照时，不应将最终性能归为某个心理学式标签。须以正文的 evaluator、测试集与重复设置限制外推。

## 7. 论文内定位

入口：[main.tex](../../../../papers/MeLA_Metacognitive_LLM-Driven_Architecture_for_Automatic_Heuristic_Design/main.tex)。方法、实验和附录均由该主文件定位；流程图：`MeEvo.drawio.pdf`、`PE.drawio.pdf`。
