# Experience-Guided Reflective Co-Evolution

- 论文：Experience-Guided Reflective Co-Evolution of Prompts and Heuristics；本地来源：[iclr2026_conference.tex](../../../../papers/Experience-Guided_Reflective_Co-Evolution_of_Prompts_and_Heuristics/iclr2026_conference.tex)；设计对象是 AHD 中 prompt 与启发式代码的协同演化。

## 1. 核心问题与方法

论文不把 prompt 当作固定外壳：由启发式评价及反思经验同时更新 prompt 与代码，使后者在相应提示分布下被生成和选择。框架图、PromptEvolution、iteration 与 algorithm 图给出循环：生成、评价、反思/经验提取、提示更新、再生成。

## 2. 论文宣称的机制贡献（逐项）

1. prompt 与启发式共同进化，缓解固定提示的表达瓶颈。
2. 以正负经验引导反思和下一代提示。
3. 经验驱动提高搜索效率与代码质量。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|联合框架主效果|§Experiment/§Experiment Result/§Main Result，表 `main_result`（TGB、BOB）|间接支持|这是 prompt、经验、代码更新共同改变的整体比较。|
|提示共同进化的贡献|本地主 tex 未检索到标明固定 prompt/移除经验的受控消融表|未验证|不能由主表或流程图推出该组件的独立因果。|
|经验质量改善生成|`TSPPrompt.pdf`、`BPPPrompt.pdf`、`Case.pdf` 案例|间接支持|案例解释可读，不是重复的因果检验。|

## 4. 机制的底层逻辑

阅读分析：代码空间和提示空间相互条件化，等价于同时搜索“候选”和“产生候选的局部语言”。它可能扩大可到达区域，但 prompt 改变也改变了每次调用的信息量；若不把调用、tokens 与评价成本计入，效率归因会混杂。

## 5. 对 LLM4AD / TraceAAD 可学习之处

|可学习点|成立前提|主要风险|最小验证方式|
|---|---|---|---|
|把轨迹经验写成可审计的生成约束|经验确实来自已评价边|提示自我强化错误规律|冻结一份经验，做有/无经验的配对下一代试验。|
|把提示版本与程序谱系共同记录|每次调用可追溯|状态空间膨胀|只记录关键指导差分和生效边。|

## 6. 证据边界

需以主 tex 实际列出的任务、模型、预算及重复为准；图示和案例不能替代跨种子统计。联合共演化还混合了 prompt、反思、父代选择和代码更新，若缺少逐项消融，就只能说“整套循环”有效，不能宣称 prompt 演化单独有效。

## 7. 论文内定位

入口：[iclr2026_conference.tex](../../../../papers/Experience-Guided_Reflective_Co-Evolution_of_Prompts_and_Heuristics/iclr2026_conference.tex)。流程资产：`framework.pdf`、`algorithm.pdf`、`PromptEvolution.pdf`、`iteration.pdf`；案例：`TSPPrompt.pdf`、`BPPPrompt.pdf`、`Case.pdf`。本次只据主 tex 已呈现的 Method/Experiments/Appendix 定位；未把图资产当作消融证据。
