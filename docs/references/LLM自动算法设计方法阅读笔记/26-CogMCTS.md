# CogMCTS

- 论文：CogMCTS: Cognitive-Guided MCTS for Iterative Heuristic Evolution；本地来源：[ijcai26.tex](../../../../papers/CogMCTS_Cognitive-Guided_MCTS_for_Iterative_Heuristic_Evolution/ijcai26.tex)；设计对象是 ACO、GLS 和构造式组合优化启发式。

## 1. 核心问题与方法

CogMCTS 保留 MCTS 的选择—扩展—模拟—回传，根节点虚拟、其余节点是可执行 Python 启发式。选择采用归一化 UCT、访问数和渐进扩展，探索衰减 $\lambda_0=0.1$；扩展有 i、em1/em2、m1/m2。em1 是对比/快速认知生成，em2 以正负知识库进行复杂认知；回传更新路径上 Q 与 N。

## 2. 论文宣称的机制贡献（逐项）

1. 双轨扩展结合快速与复杂认知。
2. 正负经验库使多轮反馈可累积。
3. 在 MCTS 内平衡探索、利用和认知指导。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|主任务结果|§Experiments 的 ACO、GLS、KP 表（表注为三次平均）|间接支持|整体系统比较，不能拆给 em1/em2 或 UCT。|
|扩展动作有用|§Ablation Study，表 `tab:kp100_actions`：移除 em1、em2、二者|直接支持|同一 KP 设定的目标动作移除，支持该局部机制贡献。|
|认知周期 $C_t=2$|表 `aco_op_kp`、`aco_mkp_cvrp`|部分支持|是参数比较，支持所测任务配置，非普遍最优周期。|
|过程/知识存储|§Cognitive-Guided Mechanism 与图|间接支持|展示规则，不检验知识真假。|

## 4. 机制的底层逻辑

阅读分析：em1 让局部候选对比产生即时修改，em2 用跨轮正负库做较慢的验证；$C_t$ 试图避免一轮就把偶然变化归为经验。MCTS 回传把叶子结果分配给路径，但若 em1/em2 单次调用成本不同，按“评价次数”对齐仍可能隐藏计算不公平。

## 5. 对 LLM4AD / TraceAAD 可学习之处

|可学习点|成立前提|主要风险|最小验证方式|
|---|---|---|---|
|明确区分快/慢经验的写入条件|有连续窗口和全局 best 事件|用最终分数过滤掉有价值失败|测正/负库命中后的一步增益与新颖性。|
|动作级而非节点级消融|动作日志准确|多动作改变 token/cost|固定总调用与 evaluator 数，逐项关停动作。|

## 6. 证据边界

正文设 $T=1000$、初始节点 10、每函数 60 秒；GPT-3.5-turbo 与 GPT-4o-mini，主 AHD 方法各三次。表的平均值及 KP 动作消融支持有限场景；ReEvo 在 KP 的早停无数据也使部分横比缺失，不能被当作零分基线。

## 7. 论文内定位

入口：[ijcai26.tex](../../../../papers/CogMCTS_Cognitive-Guided_MCTS_for_Iterative_Heuristic_Evolution/ijcai26.tex)。方法 §Methodology（选择、扩展、回传）；实验 §Experiments；消融 §Ablation Study，`tab:kp100_actions`、`aco_op_kp`、`aco_mkp_cvrp`。
