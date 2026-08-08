# Planning of Heuristics（PoH）

- 论文：Planning of Heuristics: Strategic Planning on Large Language Models with Monte Carlo Tree Search；本地来源：[example_paper.tex](../../../../papers/Planning_of_Heuristics_Strategic_Planning_on_Large_Language_Models_with_Monte_Carlo_Tree/example_paper.tex)；设计对象为组合优化启发式规划式搜索。

## 1. 核心问题与方法

PoH 将启发式设计看作规划：MCTS 树中的节点承载候选/状态，选择阶段依树策略挑选扩展位置，LLM 反思、改进并产生新启发式，评价后将结果回传。核心主张是先规划有潜力的改进路线，再调用生成器，而非对平坦种群无差别变异。

## 2. 论文宣称的机制贡献（逐项）

1. MCTS 将长程启发式改进分解为可规划决策。
2. 自反思把评价反馈转为下一步改进指导。
3. 规划与生成耦合提高有效搜索。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|整体性能|表 `tab:tsplib`、`tab: main-exp-gls`、`tab:fssp`|间接支持|整法比较，不能单独归给 MCTS 规划或反思。|
|MCTS 规划作用|§Ablation on Search Strategies，图 `fig:ablation`：MC、Greedy、Beam、MCTS，状态转移与动作生成相同、探索启发式数固定|直接支持|目标搜索策略的受控比较支持 TSP200 中 MCTS 的局部优势。|
|规划路线可解释|树/案例图|间接支持|可读路径不等于被证明导致提升。|

## 4. 机制的底层逻辑

阅读分析：规划层将“选哪个候选”提升为“在哪条改进路线继续投入”。其真正收益来自可区分路线后继潜力的信用信号；只有叶子最终 fitness 而无中间因果信息时，回传会把偶然收益归给整条路径。

## 5. 对 LLM4AD / TraceAAD 可学习之处

|可学习点|成立前提|主要风险|最小验证方式|
|---|---|---|---|
|在每条路线记录选择理由与实际后继|路径身份稳定|理由只是事后文本|按选择理由分桶，比较后继增益而非只看 best。|
|把规划预算和生成预算分开核算|调用日志可区分|规划 token 被忽略|报告每评价的 LLM 调用、tokens 与 evaluator 时间。|

## 6. 证据边界

树搜索的“访问”“价值”“回传”“渐进扩展”是不同变量；正文若未提供逐项消融，不可把主结果简化为 MCTS 已证明。训练评价与最终测试、一次成功路径与重复均值也必须分开。

## 7. 论文内定位

入口：[example_paper.tex](../../../../papers/Planning_of_Heuristics_Strategic_Planning_on_Large_Language_Models_with_Monte_Carlo_Tree/example_paper.tex)。方法图 `Planning of Heuristics Method`；主表 `tab:tsplib`、`tab: main-exp-gls`、`tab:fssp`；搜索消融 `fig:ablation`、`tab:LLM`；提示附录 `tab: tsp_prompt`、`tab: FSSP_prompt`。
