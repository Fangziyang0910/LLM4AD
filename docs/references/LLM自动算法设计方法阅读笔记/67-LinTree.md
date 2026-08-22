# LinTree

- 论文：*LinTree: Improving LLM Reasoning with Explicitly Structured Search Histories*；本地来源：LaTeX 源码目录 [`LinTree_Improving_LLM_Reasoning_with_Explicitly_Structured_Search_Histories/`](../../../../papers/LinTree_Improving_LLM_Reasoning_with_Explicitly_Structured_Search_Histories/)；研究对象：把 LLM 推理轨迹视为线性化搜索树、以显式树拓扑作为上下文的搜索策略学习（Qwen3-0.6B，SFT+GRPO）。

## 1. 核心问题与方法

两个问题：条件于完整搜索轨迹的策略是否优于只见局部状态的 LLM 启发式搜索；把轨迹的树拓扑显式化是否让历史更有用。隐式轨迹线性化为 `EXPAND ACT {action} -> {resulting state}`；LinTree 显式格式加父指针 `EXPAND sid=i ACT ... -> sid=j {state}`。两种格式来自同一批底层搜索、同一训练流程，差异仅归因于表示。GRPO 奖励 $R(\tau)=\mathbf 1[\text{valid}\wedge\text{correct}](1-\lambda\sum_{t}\gamma^t)$（$\lambda=0.005$、$\gamma=0.99$），正确轨迹恒为正收益；启发式侧用 softmax 策略梯度 $\pi(c\mid\mathcal C)=e^{-h_\theta(c)}/\sum e^{-h_\theta}$。附录报告两种被弃奖励（封顶线性罚、几何衰减乘子）不稳定。

## 2. 论文宣称的机制贡献（逐项）

- 显式树拓扑（父指针 + 状态 ID）使轨迹条件化策略稳定超过强基线。
- 访问状态的平均两两距离更高（explicit 4.19 > implicit 4.11 > BFS 3.99）：显式结构使探索更分散、少重访。
- 计划提取失败率更低（SFT 阶段 41.44%→26.00%，Sokoban 80.78%→54.17%）。

## 3. 实验究竟支持了什么

|主张|论文证据|证据等级|判断|
|---|---|---|---|
|显式树结构优于隐式线性化|三个全可观域（Blocks World 4–10、10×10 Navigation、去两箱 Boxoban）各 20k SFT/20k RL/1k 验证：GRPO-explicit 100.0/100.0/89.6 全面占优，扩展数 8.25→7.31、14.80→14.28、63.54→52.82|直接支持|同一批底层搜索、同一训练流程，差异仅表示格式——表示对照干净。|
|裸轨迹访问不足以胜过强基线|GRPO-implicit（97.3/94.9/85.9）不稳定优于 BFS(RL)（99.8/100/99.1）|反向或混合证据|"给历史"本身不兑现优势，结构化才兑现。|
|生成约束可再省扩展|Sokoban 加非法动作禁止约束达 98.9，追平 BFS(RL) 99.1 且扩展更少（54.70 vs 64.08）|部分支持|域特定约束，通用性未测。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

轨迹条件化的价值不在是否提供历史，而在历史的结构化表示：父指针让模型能区分"同一状态的多次访问"与"不同分支"，这是去重、回溯与跨分支迁移的前提。与 ToT/LATS/LLM-A* 等外部控制器（LLM 只查局部视图）的差别在于：策略本身条件于全轨迹。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：改进轨迹进入生成上下文时带显式谱系标识（父指针/路线 ID），而非纯线性文本。前提：序列化格式与现有 prompt 兼容。风险：token 开销与 ID 语义漂移。最小验证：同轨迹内容、有/无谱系标识两档固定锚点对照。
- 可学习点：把评价预算效率写进生成目标的有界折扣奖励（正确恒正、扩展受罚）。最小验证：不同 λ 下的扩展数-质量前沿。

## 6. 证据边界

三个受控全可观域、单一 0.6B 底座；作者自认开放域推理需额外设计；评估实例数与重复次数未明确（报标准误）。结论限小模型、短程规划域。

## 7. 论文内定位

Algorithm 1（best-first + LLM 启发式）、两种序列化格式定义（Method）、三域主表与扩展数统计（Results）、被弃奖励与计划提取分析（Appendix）。
