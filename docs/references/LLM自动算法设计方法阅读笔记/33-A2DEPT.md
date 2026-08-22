# A2DEPT

- 论文：*A2DEPT: Large Language Model–Driven Automated Algorithm Design via Evolutionary Program Trees*；本地来源：`../../../../papers/A2DEPT_Large_Language_Model_Driven_Automated_Algorithm_Design_via_Evolutionary_Program_T/paper.tex`；设计对象：完整可执行 solver 程序。
- 一句话定位：把 LLM-AHD 的搜索对象从固定框架内的启发式组件扩展为完整 solver 程序（template-bound AHD → open-ended AAD），并论证这种开放式搜索在工程上可以跑通、在性能上存在固定模板达不到的空间。

## 1. 问题：framework bottleneck

既有 LLM-AHD 方法（FunSearch、EoH、ReEvo、MCTS-AHD）共享一个前提：固定 solver 框架，LLM 只优化其中的启发式槽位。A2DEPT 用一组 pilot study（Fig. `fig:motivation`）质疑这个前提：在一种构造框架下进化到很好的启发式，换一个框架就失效——框架选择是一个被隐藏、却先行锁死性能上限的设计变量。换句话说，template-bound AHD 优化的始终是一个 basin 内的解，而 basin 本身不在搜索空间之内。同一 pilot 的另一半（Fig. `fig:motivation`(b)）给出开放度的代价：设计目标从 scoring heuristic 扩到 $(H,\pi)$ 再到 $(H,\pi,\rho)$，性能改善的同时不可执行率从 2.6% 升到 14.8% 再到 25.0%——搜索空间开放度与可执行性的矛盾是后文维护闭环的动机。

作者由此把研究问题重述为：如果允许 LLM 修改控制流、算法范式与 helper 函数——从启发式函数设计师变成算法架构师——搜索仍然可控吗？障碍被归纳为三个：自由程序的可执行性不稳定、程序空间爆炸、大粒度耦合修改下信用分配不透明。整篇论文就是对这些障碍的系统回答。

## 2. 方法：一条自我咬合的设计链

A2DEPT 的机制单独看都是标准件，值得记录的是它们咬合的逻辑（§`sec:methodology` 的 design rationale 段）：

搜索对象是完整程序，因此需要一棵保留全部已评估节点的树来组织 lineage——作者明确这棵树是父子变换轨迹的 memory，而非 rollout 密集的规划结构，更接近谱系档案而非 MCTS。父子间用模拟退火接受，贴合"树记录的是改进关系"这一定位；但开放程序空间的结构改动收益不即时兑现，SA 拒绝率过高、父前沿迅速收缩，于是用 Boltzmann 抽样从全树历史节点补足父集——允许暂时差的节点继续存活。保住的多样性正是 macro-mutation 做范式级重构所需的 stepping stone；而自由重构产生的代码又 necessitates 程序维护闭环：role-aware 解析区分不可变定义与可变策略，调用图识别未解析符号并迭代提示 LLM 补全至依赖闭合，再做可达性裁剪。维护不只是修复器，还是 macro 算子的另一半——macro 的 prompt 明确鼓励 top-down 设计，允许调用尚未定义的 helper，把实现留给维护循环（Template II-2）；"先当架构师、后当实现工程师"的分工把"一次生成同时完成新思想、全部控制流和接口一致性"的难度拆成两层。其可修复范围是缺符号级的简单错误（未定义依赖、不可达代码），跨模块类型契约的破坏不在能力之内（见 §4.5 的 bitmask 案例）。最后，评价反馈驱动节点局部的算子权重调度，在 micro-tuning 与 macro-mutation 之间分配生成努力，回应信用分配问题：其对信用不透明的解法是先在较小粒度上改，使归因变得容易，而非提供更多信息。更新后的算子权重由子代继承（warm start），构成轻量的谱系级搜索偏好；softmax 恒给两个算子非零概率，错误的继承偏好在数步内被反馈纠正。全局最优停滞时触发 re-annealing 升温，且同一温度同时驱动 SA 接受、Boltzmann 采样与算子调度三个机制。

## 3. 实验究竟支持了什么

| 主张 | 论文证据 | 证据等级 | 判断 |
| --- | --- | --- | --- |
| 程序级搜索优于组件级 AHD | Table `tab:main_results`：4 个标准基准 × 2 backbone 对 FunSearch/EoH/ReEvo/MCTS-AHD 全胜 | 间接支持 | 支持完整配方；表示、维护、控制器同时变化，不能归因给程序树单一因素。摘要/贡献 (iv)/§`sec:exp_standard` 三处称"对最优竞品 AAD 方法的**平均 gap 相对下降** 9.8%"，但该数字**无法从主表复现**：按 DeepSeek 组逐任务最优竞品均值（17.35%）对 A2DEPT（10.85%）实算相对下降约 37.5%——论文内部数字不一致，引用时以主表为准。 |
| 每个循环组件有贡献 | Table `tab:ablation_unified`：去 Boltzmann、去自适应调度、随机选择、固定模板均退化 | 直接支持（任务条件） | 只覆盖 CVRP/FJSP 两任务，是任务条件下的组件证据，非普适必需性。 |
| 维护提高可执行性 | §`subsec:engineering` 的机制描述 | 未验证 | 消融没有"只关维护"的变体；三大宣称机制之一缺少独立量化。 |
| AAD 后期上限更高 | Table `tab:convergence_stages`、Fig. `fig:convergence` | 部分支持 | 两点快照加单任务曲线，见 §4.1。 |
| AHD 与 AAD 互补 | Tables `tab:paradigm_generalization`、`tab:gls_random` | 直接支持（FJSP 单点） | 见 §4.3。 |

## 4. 研究见解（本文分析，非作者已证明结论）

### 4.1 收敛阶段性：早期预算购买结构，后期兑现

附录 `app:convergence_stages` 给出 $T{=}200$ 与 $T{=}1000$ 两点快照：早期 A2DEPT 仅在 CFLP 落后于 MCTS-AHD（21.87% 对 17.67%），其余三任务已领先；到 $T{=}1000$ 四任务全部最优且差距拉大（如 FJSP 9.95% 对 24.51%）。CVRP 收敛曲线显示前几百次评估的快速下降由 macro-mutation 的范式跳变驱动，而种群类基线与 MCTS-AHD 更早进入平台。

这幅图像比"探索—利用"更具体：AHD 在给定 basin 内优化，AAD 同时在搜索 basin 本身，因此 AAD 的预算分配可能具有内在阶段结构——早期结构探索（structural exploration）→ 确定算法骨架（discover backbone）→ 后期局部精修（local refinement）。MIS 成功谱系（§4.2）正是这一模式的单例：greedy 构造 → ILS 跳变 → 堆化加速与启发式精修。两条保留：早期落后只在 CFLP 出现，"前期吃亏"不能泛化；阶段结构与"种群方法不适合 AAD"都是待验证假设，论文只提供了与之一致的观察。

### 4.2 Stepping stone：中间算法的延迟效用

MIS 成功谱系（`app:case_study_success`，Fig. `fig:code_evolution`）是全文最有说服力的定性证据：贪心构造（ID 2，24.38）→ 一个因改进不足被 SA 主选择拒绝、被 Boltzmann 捞回的中间结构（ID 19，25.06）→ 第 4 代 macro-mutation 从构造式跳到完整 ILS（ID 70，25.94：perturbation、多算子局部搜索、basin-hopping 控制）→ 后期堆化加速与启发式引导（ID 318，26.44）。这是真正的 algorithm design 叙事——范式迁移，而非 priority function 调参。

其揭示的现象可概括为中间算法的延迟效用：节点价值不等于当前 fitness，暂时不优秀的结构可能是后续范式突破的 stepping stone；greedy top-k 会剪掉 ID 19 这类节点，因而与开放式算法发现在原则上相冲突。边界同样明确：这是单条成功谱系的事后叙述，属描述性证据——事后位于成功谱系只说明它属于来时路，不能证明当时重访它具有较高前瞻价值；"剪掉 ID 19 则 ILS 跳变不发生"是未检验的反事实。

### 4.3 固定模板是一种 algorithmic inductive bias

FJSP 上固定 GLS 框架的 MCTS-AHD（10.76%）明显优于全部 AAD 式方法（A2DEPT 18.95%）。作者进一步把 GLS 中 LLM 设计的 guidance matrix 换成随机矩阵（Table `tab:gls_random`）：FJSP 仅从 0.873 降到 0.808，而 CFLP 从 0.621 崩到 0.302。对照把 FJSP 的 GLS 优势定位到专家设计的邻域算子，而非 LLM 设计的知识；CFLP 则相反。

这个实验比主结果更有方法论价值：固定模板本质上是 algorithmic inductive bias。先验可靠时，AHD 是小搜索空间加强先验的高效组合；没有可靠框架时，AAD 的大搜索空间才有发现潜力。研究问题因此不应是"AAD 是否淘汰 AHD"，而是"何时应信任已有 algorithmic prior，何时应让 LLM 打破它"。作者原文止于 AHD 与 AAD 互补（`app:gls_random` 末段），prior 信任问题的表述是本文的引申。

### 4.4 轨迹只被用作谱系记忆，尚未进入生成上下文

A2DEPT 反复强调 tree、trajectory、lineage，节点也存父代/算子历史，交叉伙伴甚至按 LCA 深度衡量谱系多样性。但检查生成 prompt（`app:prompts_operators`）：micro-tuning 只给任务描述与当前函数代码，无分数、无历史；macro-mutation 多一项 `PREVIOUS_THOUGHT`，即父代生成时附带的一句算法描述；crossover 融合两个父代的当前代码。多代改进来时路、修改—结果序列均不进入生成条件。

也就是说，A2DEPT 的 trajectory 是 genealogy：history 字段被调度与交叉消费，而非被 LLM 消费。它证明了保留谱系结构有价值，却没有利用改进路径本身包含的算法设计知识——路径经历了哪些思想、哪一步为何成败、当前算法如何形成，均不参与下一步提议。这是本文留下的明确空位。

### 4.5 贡献校准

SA、Boltzmann 选择、softmax 算子调度、语义交叉单独看均为标准件。本文的实在贡献是：（i）问题框架——把 framework bottleneck 说清楚并用 pilot 展示其代价；（ii）系统组织——把结构搜索、可执行性、粗粒度信用三个难点组织成能跑通的闭环，消融显示去掉任一环至少在一个任务退化；（iii）证明完整算法结构搜索存在固定模板达不到的空间（主结果与 ILS 谱系）；（iv）用 FJSP/GLS 对照诚实地划定适用边界。

最薄弱环节有二。其一，维护机制作为三大宣称贡献之一没有独立消融，且其效果与修复消耗的 LLM 预算从未分开计量。其二，失败案例暴露了 dependency closure 的天花板（`app:failure_analysis`，IDs 158/160）：LLM 发起 set 到 bitmask 的重构，helper 实现正确，但迁移是 partial 的——下游算子仍按 set 接口工作，程序语法完整、依赖闭合，却因跨模块类型契约被破坏而崩溃。dependency closure 只能修"缺函数"，修不了 architectural consistency；full-program AAD 的下一步问题是维护演化中的 system-level invariants，这类失败恰恰是模板法天然免疫的。

## 5. 对 LLM4AD / TraceAAD 的启示

- **轨迹空位是差异化位置。** A2DEPT 证明仅保留谱系（作结构记忆）已能支撑范式跳变个例，但轨迹内容不进入生成；TraceAAD 的轨迹条件生成正对这一空位，本仓库固定锚点实验已独立支持父代来时路的单步价值，两处证据互补。风险边界：对方的谱系叙事只支持"保留"有价值，不能引为"利用"有价值的证据。
- **Stepping stone 对分配的含义。** 节点价值不等于当前分数与访问计数的函数。混合选择（SA 接受 + Boltzmann 补充）在 A2DEPT 中承担的正是父代选择与预算分配职责，"允许暂时没变好的分支以非零概率存活"与 V9.8 对 Explore child 的衰减边界宽限是同一原则的两种实现——一个用采样概率、一个用分数宽限；消融中去掉 Boltzmann 在 CVRP/FJSP 均退化。最小验证：离线重放中比较剪除被 SA 拒绝节点与现状下分支后续收益分布。
- **多尺度算子与意图调度的对照。** micro-tuning / macro-mutation / 语义交叉把修改尺度显式分层，算子偏好按节点维护、由评价反馈更新并随谱系继承。这与 Refine/Explore 意图平行（族内小改 vs 结构性重写），但对方的调度单位是节点局部的算子权重、修改粒度即算子本身，TraceAAD 的 $\pi(o_t)$ 是意图层分布；A2DEPT 是"修改尺度作为可学习谱系状态"的参照实现，且它只调节算子选择概率、不改变生成提示本身（§4.4）。
- **阶段结构的假设价值。** AAD 预算分配可能天然有"结构探索→范式选择→局部精修"的阶段；本仓库 V9.7 诊断中路线乐观项干预全部发生在前 1/3 与此图像相容，但两者都只是观察，尚不能互证。
- **程序级扩展的前提。** 只有当设计对象确需跨函数控制流时才值得扩展表示；前提是入口、不可变契约与执行验证已固定，且修复只作评估前卫生、不计入优化信用。本文未拆分原始可执行率、修复后可执行率与修复成本，这一拆分本身就是可借鉴的实验设计。

## 6. 证据边界

全系统同时改变搜索对象、维护与控制器，联合结果不能归因单组件。预算口径为 500 次 LLM 调用（修复循环同样消耗），与按真实评价次数统一的口径不可直接对齐；附录 `app:scaling_budget` 的等 token/货币预算实验（5/10 CNY 下仍全胜）部分回应该公平性质疑。标准基准仅 3 次重复，部分单元格标准差大；消融只覆盖 CVRP/FJSP；$k$ 敏感性在 CVRP 上方向不一致（$k{=}7$ 优于 $k{=}5$）。高约束任务（`tab:high_constraint`：CEVRPTW gap 0.00、IR 2.56；MRCPSP IR 15.89，20 次重复）是"开放式 AAD 可执行性"主张的最直接证据；OOD 迁移矩阵（`app:ood_generalization`，VRPTW Solomon C/R/RC 3×3）显示跨分布 gap ≤ 约 1.5× 内分布；Gemini 2.5 Flash-lite 上退化（`app:llm_backbones`），方法对模型代码能力有下限要求。收敛阶段证据为两点快照，case study 为单条谱系事后叙述。

## 7. 论文内定位

`paper.tex`：§`sec:introduction`（pilot 与三挑战，Fig. `fig:motivation`）、§`sec:methodology`（含 design rationale 段）、Tables `tab:main_results` / `tab:ablation_unified`；附录 `sec:appendix_implementation`（Algorithm `alg:aadept` 与全部超参）、`app:prompts`（Template I/II 系列）、`app:convergence_stages`、`app:scaling_budget`、`app:gls_random`、`app:case_study`（Fig. `fig:code_evolution` 与 `fig:failure_code`）。
