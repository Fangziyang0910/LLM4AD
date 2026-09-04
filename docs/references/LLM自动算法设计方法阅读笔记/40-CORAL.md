# CORAL

- 论文：*CORAL: Towards Autonomous Multi-Agent Evolution for Open-Ended Discovery*；本地来源：`../../../../papers/CORAL_Towards_Autonomous_Multi_Agent_Evolution_for_Open_Ended_Discovery/colm2026_conference.tex`，含 `sections/method.tex`、`sections/experiments.tex`、`sections/appendix.tex`；设计对象：开放式发现任务中的搜索过程、共享记忆与多个自主 agent。

## 1. 核心问题与方法

CORAL 把检索、提出、评估、更新四阶段的决定权由固定外循环移给 agent。agent 自主决定看什么历史、何时本地测试/调用 evaluator、写回什么知识；多个 agent 异步共享持久记忆而非预定义角色或通信拓扑。系统以 heartbeat intervention 防漂移，并有资源/反作弊 safeguard。它的核心不是 ACO/某个程序槽位，而是开放式搜索组织。

## 2. 论文宣称的机制贡献（逐项）

- 持久共享记忆支持跨长时程的知识积累。
- 异步多 agent 通过共享工件协同而不规定角色。
- heartbeat 干预维持持续、可控的自治搜索。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|完整 CORAL 能完成开放式压力任务|`sections/experiments.tex` 主实验表/图（单 agent CORAL 在全部 11 个任务得分最高、8 项新 SOTA；改进率高 3–10×、5–20 次评估收敛，基线 60–100）|间接支持|整体胜出不能分配给记忆、异步性或 heartbeat。|
|知识积累的作用|Table `tab:ablations` 的 "Knowledge Accumulation (1-Agent)"（关闭使 Kernel Eng. 1350→1601 即 18.6% 退化、Polyominoes 80.2→77.3、Txn 4566→4444）|部分支持|表明确消融该核心组件，支持三项 stress-test 和 Claude Code + Opus 4.6 的设置。|
|多 agent 共演化超出单纯增加计算|Table `tab:ablations` 的 "Co-evolution (4-Agent)"（vs best-of-4 独立单 agent：1103 vs 1180、84.2 vs 80.8、4694 vs 4629）；附录轨迹分析（Kernel Eng. 36% 尝试以他人 commit 为父且改进率 17% vs 9%；Polyominoes 87% 轮次引用他人知识）|直接支持|匹配的 best-of-4 对照排除"四份独立计算"解释；轨迹分析给出协同的机制性证据；范围仅三任务。|
|heartbeat 是性能来源|方法描述与系统分析|未验证|在实验标签中未见对 heartbeat 的独立受控消融。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

共享记忆可把一个 agent 的可复用发现暴露给其他探索轨迹，异步并行则提升分支多样性。其有效性依赖写入质量和检索选择；错误知识会跨 agent 放大。best-of-4 对照消除了部分计算量解释，却仍不能证明每一个记忆格式、干预规则或异步调度细节。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：持久知识以“工件＋评价＋适用范围”写入，不只存自然语言结论。前提：读取可回链。风险：共享记忆污染。最小验证：单 agent 固定预算下比较有/无可审计知识积累。
- 可学习点：多轨迹协作先与 best-of-n 独立重复做公平比较。前提：总模型调用、并行和 evaluator 预算匹配。风险：把更多计算误认为协同。最小验证：复刻论文的 4 对 4 设计于一个任务。
