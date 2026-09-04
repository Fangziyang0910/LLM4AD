# TraceAAD 各版本提示词与上下文组织索引

本目录收录 TraceAAD 自诞生以来全部 35 个具有真实代码实现与最终实验记录版本的**提示词模板原文（Verbatim）与上下文组织结构说明**。

所有文件均统一按照四部分结构组织：
1. **上下文组织逻辑**：详细解析输入给大语言模型的上下文区块构成、排序与信息流。
2. **算子逻辑**：阐明该版本包含的算子设计意图、指导指令与搜索行为约束。
3. **特殊机制说明**：说明该版本特有的机制（如两阶段调用、跨轨迹证据池、双调用事实描述、代码注释剥离、严格单代码块契约等）。
4. **真实完整的提示词模板**：原封不动还原真实的提示词文本与输出契约模板。

---

## 演进阶段分类与版本索引

### 阶段 1：轨迹种群进化与两阶段调用（V1–V7）
本阶段主要探索提示词中如何表示“历史动作与轨迹”，多采用动作提议（Action Proposal）与代码实现（Code Implementation）的两阶段调用。
- [TraceAAD-V1](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V1-Prompt.md)：两阶段调用机制，显式展开最近 5 步 `(parent, action, child, fitness_change, outcome)` 历史。
- [TraceAAD-V2](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V2-Prompt.md)：引入模式记忆库（Pattern Memory）与对比反馈（Contrast Feedback），提供模式感知动作生成。
- [TraceAAD-V3](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V3-Prompt.md)：精简动作表达，引入跨轨迹动作证据池（Cross-Trajectory Action Evidence）与参数化算子约束。
- [TraceAAD-V4](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V4-Prompt.md)：将轨迹形式化为 MDP 设计状态历史，在代码实现期首次注入历史轨迹与算子约束（`trace_ideate` 与 `trace_refine`）。
- [TraceAAD-V5](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V5-Prompt.md)：主结构父代与参考知识解耦（Primary 作为唯一结构父代，Reference 仅作知识参考），确立四类语义算子与双轨迹上下文。
- [TraceAAD-V6](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V6-Prompt.md)：废除复杂结构化动作，全面回归自然语言 Action 驱动，明确 Idea ≤ 300 字符契约与辅助函数支持。
- [TraceAAD-V7](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V7-Prompt.md)：Prompt 上下文与 V6 保持同构以维持稳定基准，调度层引入代码哈希去重与 UCB 探索衰减。

### 阶段 2：完整树搜索与双层决策成型（V8–V9）
确立以程序树（Program Tree）为基础骨架，废除两阶段调用，单次请求直接生成 Idea 与完整代码。
- [TraceAAD-V8](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V8-Prompt.md)：首个完整树搜索实现。在树结构上单阶段生成，继承四类语义算子，上下文独立区分自顶向下形成路径与已有下游直接尝试。
- [TraceAAD-V8.3](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V8.3-Prompt.md)：引入“代码搜索生成 + 独立事实描述”双调用体系（解耦意图声称与实际行为），确立五类精细算子体系（REFINE, TUNE, SIMPLIFY, INNOVATE, CROSSOVER）。
- [TraceAAD-V9](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V9-Prompt.md)：纯树简约基准，统一标准化来时路三元组格式（Idea、Diff 摘要、Outcome/Fitness 变化），继承正统四类语义算子。

### 阶段 3：锚点机制、连续调度与粒度分化（V9.1–V9.7）
围绕锚点（Anchor）选择、动态算子调度与提示词上下文精简展开密集探索。
- [TraceAAD-V9.1](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V9.1-Prompt.md)：剥离传统交叉机制，由四大纯语义意图驱动（Ideate, Refine, Synthesize, Transfer），引入批次候选生成指令。
- [TraceAAD-V9.2](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V9.2-Prompt.md)：规范局部历史窗口，锚点作为选择与信用单位，包含来时路与直接下游尝试，强化无注释可执行代码契约。
- [TraceAAD-V9.3](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V9.3-Prompt.md)：三阶段独立调用契约（策略规划 -> 轨迹决策 -> 代码实现），配合调度层三步 Rollout 绑定。
- [TraceAAD-V9.4](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V9.4-Prompt.md)：恢复原子单步生成，引入显式失败记忆（Failure Memory）防止局部重复踩坑。
- [TraceAAD-V9.5](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V9.5-Prompt.md)：首创严格单代码块输出契约（Strict Markdown Contract），确立标准父代形成历史事件表。
- [TraceAAD-V9.6](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V9.6-Prompt.md)：轻量级差分来时路，仅保留最近邻核心演变差分。
- [TraceAAD-V9.7](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V9.7-Prompt.md)：正统来时路规范，彻底剔除直接子代尝试，仅保留自根至父的形成路径，分化 Refine 与 Explore 意图。

### 阶段 4：结构迁移、探索后验与思想聚类（V9.8–V9.13）
引入跨谱系知识融合算子，探索异构思想迁移与提示词语义诱导。
- [TraceAAD-V9.8](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V9.8-Prompt.md)：引入 Hypothesis 轨迹分段与衰减宽限保护，严格区分 Refine 与 Explore 意图。
- [TraceAAD-V9.9](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V9.9-Prompt.md)：几何秩软化采样调度，保持提示词意图分离。
- [TraceAAD-V9.10](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V9.10-Prompt.md)：探索后验反馈驱动，维护“锚点 x 意图”联合臂后验分布。
- [TraceAAD-V9.11](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V9.11-Prompt.md)：停滞触发 Explore 并在其后强制绑定一步 Landing 平飞修复。
- [TraceAAD-V9.12](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V9.12-Prompt.md)：局部失败率自适应算子概率调度。
- [TraceAAD-V9.13](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V9.13-Prompt.md)：代理区域前沿条件化探索，在 Explore 提示词中注入宏观特征前沿表。

### 阶段 5：纯树化、连续调度与去注释控制（V9.14–V9.22）
彻底移除两级队列与外部种群，全面转向纯树演进；深入研究代码注释残留对 LLM 认知的污染，建立去注释机制。
- [TraceAAD-V9.14](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V9.14-Prompt.md)：纯树极简骨架，树上节点仅保留代码与评估结果，剥离派生代理状态。
- [TraceAAD-V9.15](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V9.15-Prompt.md)：引入有界代码修复机制（Bounded Repair / EH），针对语法或运行时报错给出一轮针对性修复机会。
- [TraceAAD-V9.16](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V9.16-Prompt.md)：显式可靠性契约约束（输入不变性、有界时间复杂度）。
- [TraceAAD-V9.17](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V9.17-Prompt.md)：连续块调度（Continuous Chunk Scheduling），Prompt 保持严格 matched 对齐并支持 Repair 协议。
- [TraceAAD-V9.18](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V9.18-Prompt.md)：衰减机会评分与极短全局数值事实（Global-Facts-Lite）。
- [TraceAAD-V9.19](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V9.19-Prompt.md)：在线行为景观控制（BehaveSim）与代码注释干扰消融对比（DEVELOP, EXPLORE, CROSSOVER）。
- [TraceAAD-V9.20](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V9.20-Prompt.md)：探索上下文解耦与直接结果账本（Direct Outcome Ledger），在 Explore 时截断长父链。
- [TraceAAD-V9.21](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V9.21-Prompt.md)：假说与实现解耦两阶段调用，跨分支共享公共实验卡片（Public Cards）。
- [TraceAAD-V9.22](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V9.22-Prompt.md)：冻结批次上下文（Frozen Batch Context）与分支强隔离。

### 阶段 6：多算子并行生成与选择性机制（V10–V10.2）
当前主线基准体系。按当前真实质量概率性选择父节点，单次展开多算子生成；验证选择性上下文的因果有效性。
- [TraceAAD-V10](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V10-Prompt.md)：五类语义算子体系（Develop, Pivot, Transfer, Restart, SemanticRepair），生成端与评判端状态隔离。
- [TraceAAD-V10.1](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V10.1-Prompt.md)：当前主线规范版本。概率性质量父选择，Refine / Pivot / Fuse 并发派生；标准代际演变趋势标注；严格两段式单代码块输出契约。
- [TraceAAD-V10.2](file:///home/fang/code/LLM4AD/LLM4AD/docs/prompts/TraceAAD-V10.2-Prompt.md)：Prompt View 代码注释剥离（`strip_comments_for_prompt`），老 Generation 优先裁剪，机制替代型 Anti-bloat 准则与 Boltzmann 访问惩罚。
