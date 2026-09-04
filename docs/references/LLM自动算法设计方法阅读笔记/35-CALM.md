# CALM

- 论文：*CALM*；本地来源：`../../../../papers/CALM/main.tex` 与 `appendix.tex`；设计对象：启发式种群、产生它们的本地 LLM 参数，以及二者的在线共进化。

## 1. 核心问题与方法

CALM 的主张不是单纯管理上下文，而是在进化式 AHD 中让“算法”和“LLM”共同更新。种群中每条启发式保存 idea、code、performance；每轮先选一个可行算子生成 prompt (q)，再从本地策略 \(\pi_\theta\) 采样 G 个 response，解析并评估后形成 prompt–response–performance triples。可行候选进入种群，奖励则用于 GRPO 在线更新 \(\pi_\theta\)，最终返回 best-so-far。论文用 INT4 Qwen2.5-7B-Instruct，只微调 1.15% 权重（§`sec:experiment`）。

该闭环有两层指导。verbal guidance 是 prompt/种群层：初始化；injection 用已保存的组件摘要注入新组件；replacement 按三类指令重写局部；crossover 一半按性能、另一半以 idea-token 新颖性配对；simplification 消除反复改写造成的冗余。发生停滞时，collapse 只保留初始 seed 与当前最好启发式再重启种群，触发由 \(c_n\delta_0\) 的增长概率和硬上限 C 决定。numerical guidance 是 GRPO：不可行输出按缺 idea、缺 code、格式错误、运行/超时、随机性给层级负奖励；可行输出相对 prompt 中最优父代计分，复制已有性能被惩罚、退化按相对差距惩罚、超越父代得到正奖励。

## 2. 论文宣称的机制贡献（逐项）

- 将启发式进化产生的 triples 转为 GRPO 信号，使局部 LLM 与算法种群共同演化。
- 以细粒度 verbal operators 保留可归因的局部改动，并以多样性交叉和简化调节探索/复杂度。
- 以父代相对质量和可行性层级构造数值 reward，缓解把整个结果错误归因给 response 的问题。
- 以 collapse 打破种群近亲化和停滞。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|完整 CALM 的端到端性能|Tables `tab:obp`、`tab:tsp`、`tab:cvrp_op`，均三次运行平均|间接支持|主表同时改变本地模型、GRPO、reward 与 verbal operators；不能给任何单一机制分配功劳。|
|GRPO 是该配方的关键部分|Table `tab:ablation` 的 `local, w/o GRPO`，OBP 1.78%、OP 19.89%，相对 CALM local w/ GRPO 的 0.71%、17.41% 是表中最大退化|部分支持|这是同一本地模型配方中关闭 GRPO 的直接对照；仍随 GRPO 一起去除了在线参数更新，不能分离优化器与训练数据闭环。|
|提出的相对/复制惩罚 reward 优于两种替代|Table `tab:ablation` 的 `rew∈{0.5r_rand,1}` 与 `rew=performance`|部分支持|两变体保持不可行惩罚，但同时移除了复制惩罚/改变质量归因；支持完整 reward 设计优于这两种替代，不能逐项证明每个 reward term。|
|collapse 有益且触发过早有害|Table `tab:ablation` 的 `w/o Collapse` 与四组 \(\delta_0,C\)；§Discussion “Impact of collapse”|部分支持|无 collapse 和参数敏感性均有同任务数值；最严格的 \(0.005,15\) 明显退化，说明不是“越频繁越好”。|
|各 verbal operator 及多样性交叉有贡献|Table `tab:ablation` 的 `w/o diversity`、`w/o crossover`、`w/o injection`、`w/o replacement`、`w/o simplification`|部分支持|它逐个移除五个操作/选择规则，在 OBP、OP 上均劣于完整 CALM；删除算子也改变了可行动作空间，故不等同于独立语义原理的普遍证明。|
|GRPO 训练过程中赶超 API 基线|Fig. `fig:training-curve`|间接支持|三次平均 best-heuristic 曲线是过程证据；它与表中消融共同支持训练有效，但不能替代受控组件结论。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

verbal 层把进化历史转成 LLM 可消费的改动语言，并刻意缩小一次变异的范围：这样 response 之间的质量差更可能指向局部结构，而不是整段代码同时变动。numerical 层再将 evaluator 的结果写入模型参数，因而不是只在当前种群中“挑好代码”，而是改变下一轮采样分布。父代相对 reward 尝试处理一个关键混杂：子代强可能主要因为 prompt 已含强父代。collapse 则切断同质谱系的提示自强化。四者仍相互耦合：更好的 operators 会制造更可学的 GRPO triples，GRPO 又会改变各 operator 的响应质量；现有消融不能把这种交互完全拆开。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：把“轨迹如何构造下一步 prompt”（verbal）与“是否训练模型参数”（numerical）作为两项独立研究变量。成立前提：各自 token、LLM query、evaluator 预算可对齐。主要风险：把参数学习收益误记为历史利用。最小验证：先固定模型权重，仅比较轨迹提示；再在同一 triples 和预算下测试在线更新。
- 可学习点：记录局部改动、父代、response、合法性和相对分数，建立可审计 credit 链。成立前提：能判定重复/可行并重放 evaluator。主要风险：全局 reward 错配给长代码的每个 token。最小验证：统计 injection/replacement 后相对父代的改进、退化、重复与不可行比例。
- 可学习点：把简化和重启作为控制复杂度/停滞的最小动作。成立前提：停滞由固定 evaluator 的 best-so-far 定义。主要风险：重启消耗探索预算，或简化删去关键机制。最小验证：固定总 queries，报告代码长度、unique lineage、突破次数和测试分数。
