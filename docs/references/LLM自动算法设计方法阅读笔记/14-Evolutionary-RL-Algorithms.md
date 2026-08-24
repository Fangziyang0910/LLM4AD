# Evolutionary Discovery of RL Algorithms via LLMs

- 论文：*Evolutionary Discovery of RL Algorithms via LLMs*；本地来源：[`main.tex`](../../../../papers/Evolutionary_Discovery_of_RL_Algorithms_via_LLMs/main.tex)；设计对象：固定网络与优化器下的 RL 更新规则/损失代码。

## 1. 核心问题与方法

论文搜索的不是策略网络结构，而是学习更新逻辑。GPT-5.2 与 Claude 4.5 Opus 依 prompt 生成候选算法代码；候选在五个 Gymnasium 训练环境上训练，将经环境归一化后的回报汇总为 fitness。选择时加入 Levenshtein 结构正则以控制突变幅度。最终选出 CG-FPD 与 DF-CWP-CP，并在十个环境比较 PPO、A2C、DQN、SAC。

## 2. 论文宣称的机制贡献（逐项）

- 语言模型可在更新规则空间提出超越预设 actor-critic/TD 结构的算法。
- 部分结构相似性正则平衡可用代码保留与探索。
- 固定 256×256 MLP、Adam 和 action head，使比较聚焦学习逻辑。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|演化 fitness 随代改善|Fig. `evolution_curves`（§Results）|直接支持|曲线为两 evolutionary seeds 的均值±标准差，支持该设置下的搜索进展。|
|发现算法在十环境与基线竞争|Table `results`、Fig. `env_comparison`、§Results|间接支持|整法比较不能证明是 LLM、演化或任一发现结构单独造成。|
|中等 Levenshtein 正则优于端点|Fig. `alpha_ablation`、§Ablation Studies|部分支持|正文比较 α=0、1，并报告主实验 α=.5 的两 seed 最终 fitness 范围；.5 非与端点同图的完整受控曲线。|
|CG-FPD 不需要 terminal value bootstrap|Table `ablation_value_bootstrap`|部分支持|添加 TD(0) terminal bonus 降峰值、降方差；只检验一个具体改动。|

## 4. 机制的底层逻辑

阅读分析：固定表示和优化器减少“把大网络当算法创新”的混杂，而结构正则限制 LLM 产生不可运行或无关的重写。fitness 只由五个训练环境汇总，故它仍可能选择对这些环境和归一化边界过拟合的更新规则；post-evolution hyperparameter tuning 又使最终十环境分数不能视作纯搜索阶段的直接泛化测量。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：将结构距离当作可调搜索旋钮，而非默认真理。前提：距离与功能保真相关。风险：压制必要重构。最小验证：α∈{0,.5,1} 在相同 seeds/预算下完整重复，并报有效率和 held-out 收益。
- 可学习点：固定非目标组件以减少机制混杂。前提：固定组件不成为性能瓶颈。风险：错把交互效应排除。最小验证：少量架构复现实验检查排序是否稳定。

## 6. 证据边界

进化只用五个训练环境、两个 evolutionary seeds，单代约 30 小时、四张 A100；最终算法以五个训练 seed、每 seed 选最高 eval checkpoint、100 episodes 评估并汇总。环境归一化参考值含经验基准，影响选择目标；结果虽含未见环境，仍有 post-evolution 调参，不能将其等同严格无调参泛化。文中没有以同一随机设计检验所有生成模型、prompt 与正则交互。
