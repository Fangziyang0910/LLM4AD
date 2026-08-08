# Beyond Inference-Time Search

- 论文：*Beyond Inference-Time Search: RL Synthesizes Reusable Solvers*；本地来源：`../../../../papers/Beyond_Inference-Time_Search_RL_Synthesizes_Reusable_Solvers/template.tex`；设计对象：可复用的完整 solver 代码及其 GRPO 训练策略（Hero）。

## 1. 核心问题与方法

Hero 用 Qwen2.5-Coder-14B 通过 GRPO 训练生成 standalone solver，不在每个测试实例反复 best-of-k 搜索。奖励由格式、执行、结构 scaffold/课程式反馈和可行性门控的目标奖励组成；系统 prompt 要求 Deconstruct–Hypothesize–Critique。主任务是 SDS，另有 JSSP；论文还将一个生成代码冻结后跨测试集执行，检验“compile once”。

## 2. 论文宣称的机制贡献（逐项）

- RL 将算法行为写入可一次生成、可多实例复用的 solver。
- 结构 scaffold 与硬可行性门控改善程序有效性和搜索质量。
- 相对增加 base model sampling，专门 policy 更有效率且更稳健。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|完整 Hero 在 SDS 有较高通过率/低 VBS gap|Fig. `fig:main_performance`、Table `tab:main_results`（3 seeds、N=3000）|间接支持|这是完整 RL、prompt、奖励和 evaluator 的联合结果。|
|solver 可跨实例复用|Table `tab:compile_once_main`；Appendix `app:baseline_eval_additions` 固定代码协议|直接支持|Frozen Hero 在整套 held-out SDS 上不变执行；论文报告选择规则和 matched fixed-code evaluator。|
|结构 scaffold 有益|§Ablation “Necessity of Structural Scaffolding”；Appendix Table `tab:ablation_configs` 的 `w/o Structure`|部分支持|`w/o Structure` 同时移除结构检测与 curriculum，故支持该组合而非两者各自。|
|硬 gate/奖励归一化有益|Appendix `app:ablation_rewards`（Soft Gate、normalization sensitivity）|部分支持|与 Hero 匹配的 3-seed 比较直接支持所列替代配方；不是每个奖励项的独立效应。|
|RL 的收益超过单纯增加采样|Fig. `fig:scaling_gap`|部分支持|同 prompt 的 base pass@k 曲线与 greedy Hero 有直接对照，但训练成本未包含在该摊销执行成本列。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

RL 把“写出正确搜索循环和约束保护”从实例内采样转成模型策略；硬 gate 先驱动可行性，再对可行样本优化目标，适合代码奖励稀疏的场景。VBS 是比较方法中的最强可行解而非认证最优，因此 gap 的绝对值应理解为相对基准。固定代码结果也可能受 SDS 家族与模板相似性影响。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：把可执行性、可行性、质量分层报告。前提：三层 evaluator 语义固定。风险：仅优化格式/通过率。最小验证：给每层单独计数并测试 quality conditional on feasible。
- 可学习点：若主张可复用算法，必须固定代码跨 held-out 实例跑。前提：冻结选择规则不手挑。风险：每实例重新生成掩盖泛化。最小验证：采用 Appendix 的固定代码协议。

## 6. 证据边界

GRPO 配置为 14B、group 64、90 steps、360 prompts/23,040 episodes（Table `tab:grpo_hyperparams`）；SDS 训练集 10,000 实例（Table `tab:dataset_generation`）。SDS 主结果三 seed，JSSP 也是 101/202/303 三 seed，但奖励/solver contract 有域适配；时间列排除 LLM inference，不能读为端到端训练成本比较。HumanEval/MBPP 为三 seed（Table `tab:bigcode_results`），只支持未见显著通用代码退化的该测试。

## 7. 论文内定位

`template.tex`：§`sec:methodology`、§`sec:experiments`；Figs `fig:main_performance`、`fig:scaling_gap`、`fig:prompt_comparison`、`fig:logic_gap`；Tables `tab:main_results`、`tab:compile_once_main`、`tab:ablation_configs`、`tab:grpo_hyperparams`、`tab:bigcode_results`；Appendices `app:rewards`、`app:experimental_details`、`app:baseline_eval_additions`、`app:transfer_domains`。
