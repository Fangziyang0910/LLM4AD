# GAE

- 论文：*GAE: Graph-Augmented Evolution for Scientific Discovery via Reinforcement Optimization*；本地来源：`../../../../papers/GAE_Graph_Augmented_Evolution_for_Scientific_Discovery/paper.pdf`；设计对象：非线性振子符号方程程序。

## 1. 核心问题与方法

GAE 用 relational GNN 把 AST 解析成 typed computation graph 并在线回归程序质量；离散 SAC meta-controller 从 2700 维动作空间（变异类型 × 目标节点 × 参数索引）选出编辑元组、其变异类型分量注入 prompt 作结构提示（LLM 无有效输出时整个元组作 AST 级兜底变异）；每父代 8 个 offspring 的组相对奖励（相对父代真实分的改进量、组归一优势）再用 PPO-clip 目标在线更新 LLM 全权重。**SAC 不选父代**——正文明确父代在所有配置下均匀采样自 elite archive（摘要"dynamically selecting optimal parents"与正文矛盾，以正文为准）；父代嵌入只作 SAC 状态。

## 2. 论文宣称的机制贡献（逐项）

- 图表示为结构相近的程序提供局部可学习状态。
- SAC 学习"改哪里/怎样改"（编辑类型/节点/参数），替代固定 mutation prompt；不承担父代选择。
- 在线 GRPO 让 LLM 算子随搜索经验适应；SAC 奖励含新颖性项与复杂度惩罚（QD 式）。

## 3. 实验究竟支持了什么

|机制主张|论文证据|证据等级|判断|
|---|---|---|---|
|完整 GAE 在振子发现上有竞争力|Tables 1–2、Figure 2|间接支持|Table 2 的受控低预算比较显示联合系统较好；Table 1 引用的高预算结果不可直接横比。|
|GNN、SAC、GRPO 各自有效|§4、Appendix B|未验证|Appendix B 只说明参数与训练信号可分解，没有报告 w/o-component 成绩。|
|发现式子能外推且有物理可解释性|Table 2、§4 qualitative equations|部分支持|ID/OOD NMSE 与公式案例支持该任务上的外推；只有一个符号回归域。|
|在线 GRPO 在搜索中持续改善算子|Figure 2、§3.5|未验证|有性能轨迹但无权重固定 LLM 的匹配对照，无法归因训练。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

图编码器压缩结构，controller 分配编辑方向，GRPO 更新代码生成分布，三者分别对应 state、action 和 operator learning。不过共享终端 reward 会让三条学习链强耦合，单一最终胜出不能证明各自机制。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：把“父代结构—编辑类型—子代结果”作为 controller 训练样本。前提：先用 component-off 消融证明额外训练真正有益。

## 6. 证据边界

核心实验是单一 Nonlinear Oscillators 数据集、GAE 3 runs；基线预算与模型并非全部一致。KL 项按正文仅监控而不贡献梯度，这与常规 GRPO 描述不同，复现时不能按名称臆测。
