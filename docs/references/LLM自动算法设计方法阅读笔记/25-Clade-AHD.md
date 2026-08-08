# Clade-AHD

- 论文：Clade-AHD: Clade-level Selection for MCTS in Automatic Heuristic Design；本地来源：[main.tex](../../../../papers/Clade-AHD_Clade-level_Selection_for_MCTS_in_Automatic_Heuristic_Design/main.tex) 与 [appendix.tex](../../../../papers/Clade-AHD_Clade-level_Selection_for_MCTS_in_Automatic_Heuristic_Design/appendix.tex)；设计对象为 MCTS-AHD 的谱系（clade）级选择。

## 1. 核心问题与方法

Clade-AHD 指出逐节点 UCT 容易把预算锁给偶然高分的单节点。它把共享祖先的节点聚成 clade，在家族层估计潜力/分配扩展，再在选中家族内挑节点和动作；因此“节点访问”与“谱系预算”是两层变量，而非同一个 Q 值。

## 2. 论文宣称的机制贡献（逐项）

1. clade-level selection 降低节点级噪声和早熟利用。
2. 家族级多样性保留不同改进方向。
3. 与 MCTS-AHD 相比改善 AHD 搜索效果。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|整体比较|§Experiment，表 `tab:step-by-step`、`tab:step-bpp-online`、`tab:aco`；收敛图 `fig:convergence_curves`|间接支持|整法领先不能单独归给 clade 选择。|
|clade 选择|本地 `main.tex`、`appendix.tex` 未检索到可确认的 clade-vs-node 同预算消融 label|未验证|不以整体表、`flow.pdf` 或 `comparison.pdf` 补足层级选择因果。|
|谱系覆盖|`flow.pdf`、`comparison.pdf`|间接支持|流程/可视化不是性能因果证据。|

## 4. 机制的底层逻辑

阅读分析：clade 是对“同一早期设计思想的多个后续实现”的共享信用。它能降低单一 child 的评价方差，但也可能让弱祖先的家族持续占预算；clade 切分规则决定统计单元，必须明确。

## 5. 对 LLM4AD / TraceAAD 可学习之处

|可学习点|成立前提|主要风险|最小验证方式|
|---|---|---|---|
|按共同早期边聚合路线信用|谱系准确|家族定义任意|固定两种切分，比较预算分布和后继增益。|
|把家族探索与节点扩展分层|每层都有独立日志|双层 UCB 难审计|记录 clade 选择、节点选择、生成、回传四事件。|

## 6. 证据边界

主结果、树图和谱系案例不能证明聚类层的每个公式。尤其需核查 clade 数、每轮 LLM 调用、渐进扩展及回传目标是否固定；不同成本下的“访问更多”不是机制优势。

## 7. 论文内定位

入口：[main.tex](../../../../papers/Clade-AHD_Clade-level_Selection_for_MCTS_in_Automatic_Heuristic_Design/main.tex)、[appendix.tex](../../../../papers/Clade-AHD_Clade-level_Selection_for_MCTS_in_Automatic_Heuristic_Design/appendix.tex)。图资产 `flow.pdf`、`comparison.pdf`、`lambda.pdf`；表图 label 以两 tex 实际定义复核。
