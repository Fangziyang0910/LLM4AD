# AlgoPilot

- 论文：*AlgoPilot: Fully Autonomous Program Synthesis Without Human-Written Programs*；本地来源：`../../../../papers/AlgoPilot_Fully_Autonomous_Program_Synthesis_Without_Human_Written_Programs/paper.pdf`；设计对象：由 Compare/Swap 轨迹恢复出的排序程序。

## 1. 核心问题与方法

AlgoPilot 先随机生成双循环 Python 函数及执行轨迹，用这些非人工算法轨迹训练 Trajectory Language Model（TLM）；再以排序环境奖励训练 Transformer，并把 TLM 的下一操作概率作为软奖励。最后把学到的操作轨迹交给 GPT-4o-mini还原 Python 程序。

## 2. 论文宣称的机制贡献（逐项）

- 用随机程序轨迹训练 TLM，避免以人工算法轨迹为监督。
- 用 TLM 奖励把“能完成任务”的 RL 轨迹约束成更像程序的重复结构。
- 从轨迹恢复出可复用的 Bubble Sort 代码。

## 3. 实验究竟支持了什么

|机制主张|论文证据|证据等级|判断|
|---|---|---|---|
|RL 能学会在短数组上排序|§3.1.3、Figures 1–2|直接支持|成功率和操作数随训练报告；数组 16 未获得显著成功率。|
|TLM 引导后出现可识别的算法轨迹|§3.4、Figures 4–5|部分支持|给出约 95% 成功率、3–5 个轨迹差异和十次可恢复 Bubble Sort 的案例，但缺少无 TLM 的等预算结构性指标对照。|
|实现了完全自主的程序生成|§3.4 与 §4|反向或混合证据|最后一步依赖已知 Bubble Sort 的 GPT-4o-mini；作者明确把去除这项先验列为未来工作。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

任务奖励只约束终点，TLM 密集奖励约束路径形状，因此可能把无规律但成功的动作序列推向可压缩循环。这里的“轨迹”是候选算法执行过程，不是 AAD 的父子改进历史。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：把执行轨迹的可压缩性或规则性作为辅助信号。前提：不会奖励低效的表面重复。最小验证：固定成功率后比较程序长度、复杂度和泛化规模。
