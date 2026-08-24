# AHD-Agent

- 论文：*AHD-Agent: Agentic Reinforcement Learning for Automatic Heuristic Design*；本地来源：`../../../../papers/AHD_Agent_Agentic_Reinforcement_Learning_for_Automatic_Heuristic_Design/neurips_2026.tex`，含 `Sections/Methodology.tex`、`Sections/Experiment.tex` 与 Appendix；设计对象：启发式设计过程中的 agent policy 与生成代码。

## 1. 核心问题与方法

AHD-Agent 将自动启发式设计表述为 agentic RL：agent 在任务上下文中规划、生成/修改程序、调用评估并据反馈继续行动；RL 训练其长程动作策略，而不只在固定进化循环内调用 LLM。其关键科学问题是：把搜索控制权交给训练后的 agent 是否比固定 AHD 提示/进化更有效。

## 2. 论文宣称的机制贡献（逐项）

- 将 AHD 的多步搜索决策训练为 RL policy。
- agent 具备生成、评估、反思和迭代的闭环。
- 以多任务实验比较固定 AHD 基线。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|完整 agentic RL 配方优于比较方法|§`sec:overallResult`，Table `tab:table20-rl-training-mean-gap-no-para`；未见域为 Table `tab:table21-generalization-mean-gap-no-para`，连续域为 `tab:caf-table4-comparison`|间接支持|端到端结果不能证明 RL、agent loop 或某个工具动作各自有效。|
|跨域 RL 训练的训练域数影响|§`sec:cross-domain-ablation`，Fig. `fig:cross-domain-slopes`，所有变体 500 steps|部分支持|逐步扩展训练 mixture，并在未训练的 OP-ACO 和 in-domain TSP-ACO 报告变化；它检验域混合规模，不是 RL-on/off 消融。|
|agent 对诊断工具的利用|Appendix §`apx:tool_ablation`，Table `tab:deepseek-v4-tool-ablation-two-domain`|部分支持|同为 DeepSeek-V4-Flash，比较 evaluator-only 与 full tools；表明该 agent 配置从工具获益，不能归因到某一个工具。|
|多步 agent 行为有因果作用|§Training Curves 的 Fig. `fig:rl-training-reward-turns`|间接支持|reward/turn 数过程曲线只表明训练中行为改变，不证明其导致质量提高。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

RL 可以把 evaluator 的延迟奖励分配给“何时探索、何时修复、何时停止”等控制选择；但环境同时包含代码执行、提示和搜索预算，reward 改善也可能只来自更多尝试。对 AAD，agent 的行动序列必须与候选 lineage 一起保存，才能审查信用分配。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：把控制器动作、观察和结果记录为显式轨迹。前提：每一步 evaluator 成本可计。风险：长轨迹掩盖预算扩张。最小验证：固定调用/eval 数，比较一个预定义控制器与 agent。
- 可学习点：训练与搜索分层报告。前提：有 frozen policy 对照。风险：把模型参数变化误说成历史利用。最小验证：同一 policy 在未见任务上的零更新评估。

## 6. 证据边界

§Experimental Setup 与 Appendix `appendix:rl-training`、`app:data_split_protocol` 给出训练/数据边界；训练动态为 500 steps（§Training Curves）。总体设计曲线由五次独立 design runs 对齐；工具消融只覆盖 CVRP-Constructive、TSP-ACO 与 DeepSeek-V4-Flash。没有固定同一 policy 后仅关闭 RL 的匹配实验，不能将总体优势归因给“RL 本身”；也没有逐工具或逐动作消融。
