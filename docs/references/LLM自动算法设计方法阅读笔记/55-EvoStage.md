# EvoStage

- 论文：*Advancing Automated Algorithm Design via Evolutionary Stagewise Design with LLMs*；本地来源：`../../../../papers/EvoStage_Evolutionary_Stagewise_Algorithm_Design/paper.pdf`；设计对象：多阶段优化算法与调度规则。

## 1. 核心问题与方法

Stagewise-Design 由 coordinator 自动把算法任务分解为顺序阶段，每完成一段就运行并把中间执行反馈用于反思和修正后续方向；EvoStage 再把它与 Global-Explore、Global-Enhance 两个整体视角算子放进进化框架。

## 2. 论文宣称的机制贡献（逐项）

- 阶段分解降低单次 LLM 设计复杂度。
- 实时中间反馈能在完整算法结束前纠正错误路线。
- 局部阶段设计与全局探索/强化互补。

## 3. 实验究竟支持了什么

|机制主张|论文证据|证据等级|判断|
|---|---|---|---|
|完整 EvoStage 在芯片布局上有效|Tables 1–4|间接支持|联合三算子与任务专用 evaluator 的结果。|
|Stagewise-Design 算子有益|§3.1 Ablation、Figure 6|直接支持|完整版本与 w/o Stagewise-Design 在只设计 learning-rate schedule 时对照，前者更快更好。|
|方法可扩展到 BO acquisition function|Table 5、Figure 9|间接支持|跨应用联合结果，未隔离阶段反馈。|
|阶段数与反馈频率的独立贡献|§2.2、Appendix settings|未验证|采用固定 3/4 阶段，未报告 matched sensitivity。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

阶段执行把迟到的终端分数变成较密集的路线反馈，并允许后段条件化于前段真实效果。分解若不对应算法的真实时间尺度，也可能把本应联合优化的决策人为割裂。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：把长改进 proposal 拆成可验证 checkpoints。最小验证：相同总执行预算比较一次性代码、固定分段、LLM 自适应分段，并追踪回退率。
