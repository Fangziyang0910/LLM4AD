# MeEvo

- 论文：MeEvo: Metacognitive Evolution for Automatic Heuristic Design；本地来源：[main.tex](../../../../papers/MeEvo_Metacognitive_Evolution_for_Automatic_Heuristic_Design/main.tex)；设计对象为可执行启发式程序的进化。

## 1. 核心问题与方法

MeEvo 把 LLM 产物拆成两个成分：**reasoning trace 作为可遗传基因型，heuristic code 作为表型**。诊断：自然进化（FunSearch/EoH/ReEvo/MCTS-AHD）探索表型但丢弃基因型（交叉继承代码片段却不继承"为何有效"的策略知识）；元认知进化（MeLA）积累基因型但无法发现推理轨迹之外的新策略族。方案是**单一种群内两个过程循环交替**（默认 2 代 NE + 1 代 ME，反对并行共进化：交叉需稳定父代池、ME 需跨多代轨迹区分模式与噪声）：NE 内交叉概率 $p_c=0.5+0.2(1-\text{Eval}/L)$ 从 0.7 线性降到 0.5（交叉=早期探索，变异=后期利用）；ME 两步管道（反思产出结构化 Meta Insight：收敛诊断四分类/局限分析/改进路径，再映射为代码与轨迹）。推理轨迹永不回改（类比表观遗传）。

## 2. 论文宣称的机制贡献（逐项）

1. 用元认知经验提升改进的针对性。
2. 让经验驱动的演化兼顾探索与利用。
3. 在 AHD 基准上超过人工、进化或 LLM 基线。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|整体性能|§Experiments，表 `tab:comparative`（5 问题 × 2 骨干 × 5 次独立运行）|间接支持|DeepSeek 下五问题全部最优（如 ACS 578.16±5.21 vs MeLA 588.50）；8 组 Mann-Whitney U 检验 7 组显著（4 组完全分离 U=0）。|
|两过程必须交替|消融表 `tab:ablation_nm`：(N,/)、(/,M)、(1,1)、(1,2)、(2,1) 五配置|直接支持|单过程各有崩坏域（纯 ME 在 ACS 1293.54、纯 NE 98.19 vs 默认 578.16/50.80）；(2,1) 最优。|
|探索先利用后的方向性|$\alpha/\beta/K$ 敏感性表|部分支持|固定 $p_c$（β=0）与反向调度均劣于默认；$\alpha$ 提到 0.6 使 ACS 恶化（收敛期变异压力须约 50%）；父代池 K=5 呈 U 形。|
|搜索更有方向|反思示例或过程图|间接支持|可解释性案例不等于因果证据。|

## 4. 机制的底层逻辑

阅读分析：它把历史从“候选集合”转换成“选择何种改动”的语义状态。相较只按 fitness 选父代，这可能减少无效重试；代价是元判断由同一 LLM 生成，易受提示、评价噪声和事后解释影响。

## 5. 对 LLM4AD / TraceAAD 可学习之处

|可学习点|成立前提|主要风险|最小验证方式|
|---|---|---|---|
|把改动意图与实际代码差分绑定|能取得真实 lineage|意图与改动不一致|抽样审计意图—差分—分数三元组。|
|对失败也保留可检索摘要|失败有可靠错误/分数信号|负经验过度压制新颖性|只屏蔽重复失败模式，测新颖候选率。|

## 6. 证据边界

主结果是组合系统证据，但组件消融与显著性检验齐全（5 次运行 × Mann-Whitney）。成本约 49 万 tokens（L=100），为 FunSearch 的 7 倍、ReEvo 的 2 倍；作者论证额外 token 在复杂约束问题（ACS/WSN 对 FunSearch gap 114.94%/89.72%）杠杆最大，在已收敛的简单问题上收益小。TSP-Construct 上仍全面落后 GP 与 POMO——元认知收益有任务边界。

## 7. 论文内定位

入口：[main.tex](../../../../papers/MeEvo_Metacognitive_Evolution_for_Automatic_Heuristic_Design/main.tex)。使用 Methodology（交替协议、$p_c$ 调度、ME 两步管道）、Experiments 的 `tab:comparative`、消融 `tab:ablation_nm` 与 $\alpha/\beta/K$ 敏感性表、Mann-Whitney 检验；相关工作含本组最系统的方法谱系表（Natural Evolution / Prompt-level / Metacognitive / hybrid 四类）。
