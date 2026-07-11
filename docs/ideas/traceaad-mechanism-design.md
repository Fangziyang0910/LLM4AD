# TraceAAD 设计：过程信息为一等公民的融合搜索

2026-07-08

本文件是当前唯一 TraceAAD 的完整机制设计。它从第一性原理重新组织搜索，并充分吸收 EoH / ReEvo / HSEvo / MEoH / MCTS-AHD / ShinkaEvolve / PathWise 各家经审计后确认有效的成分。

## 0. 设计立场

早期原型是在"trajectory = 一串 node"这个具象容器上做加法。当前设计抓住 trajectory 的**灵魂**而非外壳：

> **把"算法改进的过程信息"提升为搜索的一等公民。** 一个程序好不好（结果）和它是怎么变好的（过程）是两种不同的信息；现有方法几乎只利用前者。TraceAAD 让过程信息参与采样、生成、信用分配三个环节，并以此贯穿三层记忆与三个回路。

由此推出三个贯穿性主张：

1. **泛化内嵌进 credit，而非事后评估。** 各家都只优化 in-distribution fitness，几乎无人直面"找到的程序是否泛化"。TraceAAD 把"一个改进是否跨 instance/规模稳定"作为 credit 信号本身。
2. **Context 是因果叙事，不是 lineage 堆砌。** 给 LLM 的不是"这些是你的祖先"，而是"这些动作带来了正/负改进、这些模式跨路径反复有效、好与坏的差异在哪"。这是 MCTS-AHD s1、ReEvo reflection、PathWise critic 三者以因果为中心的融合。
3. **记忆分层、回路分离。** 程序/路径/模式三层记忆 + 进化/蒸馏/反思三个回路，职责清晰，避免把"存什么、采什么、学什么"搅在一起。

## 1. 从第一性原理重看 AAD 搜索

把 LLM-based AAD 拆成 8 个根本问题，每个给出各家解法与 TraceAAD 的取舍（融合是透明的）：

| 根本问题 | 各家做法 | TraceAAD 取舍 |
| --- | --- | --- |
| 经验如何表示 | EoH=程序集；MCTS=完整树；PathWise=语义图 | **三层记忆**：程序（结果）+ 路径（过程）+ 模式（知识） |
| 采样什么 | EoH=program；MCTS=node；Shinka=weighted program | 采样对象是**改进路径**，路径价值是多维的 |
| credit 怎么分配 | MCTS=backprop(max/mean，over-credit) | **stepwise attribution**，每步只承担自己的 Δ；并叠加泛化信号 |
| 给 LLM 什么 context | EoH=个体；MCTS s1=lineage；ReEvo=reflection；PathWise=critic | **因果叙事 + 蒸馏模式 + 对比反馈**，三者融合 |
| 如何避免坍缩 | HSEvo=diversity 测量；Shinka=novelty+islands；MEoH=多目标 | **多层多样性 + islands + 多目标 non-dominated + novelty gate** |
| 算子/策略预算 | Shinka=portfolio+bandit；MCTS=UCT 自动 | **自适应算子组合**，覆盖 exploit/explore/recombine/simplify/generalize/jump 全谱，bandit + 阶段感知 |
| 反馈噪声/deceptive | 几乎无人直面（ReEvo reflection 部分缓解） | **对比式评估 + fitness 置信度**，相对排序更鲁棒 |
| 终极目标=泛化 | 无人内嵌 | **泛化维度进 credit 与 value** |

关键拒绝：不取 MCTS-AHD"保留弱节点"动机（未验证）、不取 `max`-backprop（over-credit）、不取 PathWise 多 agent entailment 图（价值未证实且 token 翻倍）、不取 ReEvo"verbal gradient/landscape"理论框架（不可证伪），只取它们各自真正有效的机制。

## 2. 整体架构：三层记忆 + 三个回路

```
        ┌─────────────────── 三层记忆 ───────────────────┐
        │  Pattern Memory   跨路径蒸馏的可复用机制/教训    │ ◀── 蒸馏回路
        │  Trajectory Memory 改进路径 + stepwise outcomes │ ◀── 进化回路（主）
        │  Program Memory    所有程序 + 多目标评估(ground) │
        └────────────────────────────────────────────────┘
                                  ▲ 因果叙事+模式+对比
            反思回路 ──▶ 生成(LLM) ──▶ 评估 ──▶ credit ──▶ 更新三层 + survival
```

- **进化回路（主）**：trajectory 采样 → 生成 → 评估 → stepwise credit → survival。
- **蒸馏回路（周期）**：从 Trajectory Memory 提炼 Pattern Memory。
- **反思回路（触发）**：对比式反馈 → verbal lessons → 注入生成 context。

三层记忆 + 三回路对应吸收：EoH（program memory + 进化回路）、MCTS（路径视角但不绑 node）、ReEvo（反思回路→verbal signal）、PathWise（对比反馈，但轻量、无多 agent 图）、ShinkaEvolve（蒸馏→meta-scratchpad；算子组合）、HSEvo/MEoH（多目标、diversity 进 survival）。

## 3. 三层记忆的数据结构

跳出具象形式，按职责定义：

**Program Memory（结果库，ground truth）**
```
Program = { id, code, idea, fitness_vector,      # 跨 instance/规模的多目标向量
            runtime, complexity, confidence,      # 评估置信度(方差)
            parent_id, mechanism_tag, task_evals }
```

**Trajectory Memory（过程库，采样对象）**
```
Trajectory = { id, steps: [Step], endpoint_id,
               value: ValueVec, island_id, visit_count, status }
Step      = { program_id, operator, mechanism_tag,
              delta_vector,           # 跨 instance 的 Δfitness 向量
              outcome, generalization_signal }
```
注意 `delta_vector` 是**向量**而非标量——这是泛化 credit 的基础。trajectory 不再是"node 序列",而是"带因果的改进叙事"。

**Pattern Memory（知识库，注入 context）**
```
Pattern = { id, kind: MECHANISM|LESSON|ANTI_PATTERN,
            text, mechanism_tag, support_ids,    # 支撑它的 trajectory/step
            generalization_score, confidence, updated_iter }
```

## 4. 多维 Trajectory Value

trajectory 的价值不塌缩成单标量（MEoH 教训），而是一个向量，采样用可配置标量化、survival 用 non-dominated：

```
ValueVec(T) = ( V_quality(T), V_potential(T), V_diversity(T),
                V_novelty(T), V_generalization(T) )

V_quality        = normalize( endpoint.fitness_vector )           # 多目标归一
V_potential      = stepwise_recent_improvement(T) − saturation(T) # 过程信息
V_diversity      = marginal_diversity_contribution(T, Memory)     # 多层，见 §8
V_novelty        = 1 − max_similarity(T, Memory)                  # 多层，见 §8
V_generalization = stability_of_endpoint_gain_across_instances(T) # §5
```

采样（含探索项）：
```
select(T) = argmax  scalarize(ValueVec(T)) + c(t)·UCB(T)
            UCB(T) = sqrt( ln(N) / (visit_count(T)+1) )
            c(t)   = c0·(1 − t/max_iter)
```
`scalarize` 默认加权求和，但任一维度可置零以退化（便于聚焦调参）。**`V_potential` 与 `V_generalization` 是相对各家的真正增量**：前者把过程信息变成采样信号，后者把泛化目标变成采样信号。

## 5. Stepwise + 泛化 Credit

### 5.1 stepwise attribution（拒绝 backprop）
每个 step 只承担自己的 `delta_vector`，后代功劳不沿路径回传（直接规避 MCTS `max`-backprop over-credit）：
```
step_credit(s) = robust_mean(s.delta_vector)        # mean − λ·std，见 §5.2
path_value(T)  = Σ_i γ^(L−i) · step_credit(s_i) / Z # 近端加权
                 + λ_pos · positive_step_ratio(T)
                 − λ_down · downside(T)
```

### 5.2 泛化信号（核心增量）
评估返回的不是单标量，而是跨 instance/规模的 `fitness_vector`。一个 step 的改进是否"泛化"，由其 `delta_vector` 决定：
```
generalization_signal(s) = sign_consistency(s.delta_vector)   # 各 instance 上同号的占比
                         × robust_magnitude(s.delta_vector)   # mean − λ·std
```
- 改进在所有 instance 上都正（稳定增益）→ 高泛化 credit，视作**可迁移机制**，沉淀进 Pattern Memory。
- 只在部分 instance 上正（疑似 overfit / deceptive）→ 低 credit，标记为 local hack，不轻易进 elite。

这把"泛化"从跑完才看的事后指标，变成每次评估都产生的、驱动搜索的信号——填补了各方法的共同空白。

## 6. 因果叙事 Context 构造

给 LLM 的 context 以**因果和模式**为中心组织，而非堆砌 lineage。三段式：

**A. 因果叙事（this trajectory）**
```
在你这条改进路径上：
  step k−2: 算子=endpoint_mutation, 机制=local_density,  Δ=+0.8 (泛化: 5/5 instance 稳定)
  step k−1: 算子=backtrack,          机制=NN_rank,        Δ=−0.3 (退步)
  step k:   算子=endpoint_mutation,  机制=row_normalize,  Δ=+0.1 (饱和)
当前方向: 改进率下降 → 建议换机制族或从 step k−2 的高泛化前缀分叉。
```
（吸收 MCTS-AHD s1 的 lineage，但每个 step 带 Δ 与泛化标注，因果显式化。）

**B. 蒸馏模式（cross-trajectory，from Pattern Memory）**
```
跨路径反复有效的机制: [local_density] 在 4 条轨迹上带来稳定增益;
                      [edge_contrast] 在 2 条轨迹上仅局部有效(慎用)。
近期教训: ...
```
（吸收 ReEvo reflection / ShinkaEvolve meta-scratchpad 的 verbal signal，但以"机制 × 泛化"组织。）

**C. 对比反馈（critic contrast）**
```
近期最佳改进 vs 最差退步的关键差异: ...（由反思回路产出，见 §9）
```
（吸收 PathWise critic 的对比反馈，但单次 LLM、不做多 agent rollout。）

加上 base program、目标函数契约、算子约束。两阶段生成不变（action → code），但 action 阶段的 context 是上述三段式。

## 7. 自适应算子组合（全谱 + bandit + 阶段）

算子覆盖搜索全谱，按角色分组：

| 角色 | 算子 | 吸收自 |
| --- | --- | --- |
| exploit | Endpoint Refine（高泛化方向继续） | EoH M1/M2 |
| path-correct | Backtrack Branch（从高泛化前缀分叉） | MCTS-AHD progressive widening |
| recombine | Mechanism Crossover（迁移单个稳定机制） | EoH/ReEvo crossover + §5.2 泛化筛选 |
| simplify | Distill（去冗余、降复杂度） | EoH M3 |
| generalize | Scale-Transfer（把机制换到不同规模验证） | **新增**，直接驱动泛化 |
| explore | Novelty Jump（换机制族） | ShinkaEvolve novelty / EoH E1/E2 |

选择：
```
candidates = {op | trigger(op, T, Memory) 满足}
op ~ softmax( operator_value(op) / τ_op )   over candidates
operator_value = α·gain + β·valid + βn·novel − δ·regress − δ·cost
τ_op 按搜索阶段衰减: 早期偏 explore/recombine, 中期 recombine/generalize, 晚期 exploit/simplify
```
每步用"child 是否进 non-dominated / 是否高泛化"更新 bandit。novelty jump / scale-transfer 在搜索停滞时由阶段逻辑强制提升权重。

## 8. 多层多样性 + Islands + Novelty

**多层相似度**（novelty gate 与 V_diversity/V_novelty 共用）：
```
sim(T1,T2) = wd·sim_code + wm·sim_mechanism + wt·sim_trajectory_pattern
```
程序层（code embedding）、机制层（mechanism tag/profile）、轨迹层（operator+outcome 序列）。机制层比代码层更稳定，是主信号。

**Islands（防全局收敛到一个 basin）**：trajectory pool 分成 K 个岛，岛内竞争生存，岛间周期性 migration（迁移高泛化 trajectory）。吸收 ShinkaEvolve islands，但作用在 trajectory 层。

**Survival**：岛内用 `ValueVec` 做 non-dominated + dominance-dissimilarity；novelty gate 在入池前拒绝高相似新 trajectory（省评估后的投资）。

## 9. 鲁棒反馈：对比评估 + 置信度

直面"反馈带噪/deceptive"：

- **fitness_vector + confidence**：每个程序在多 instance 上评估，返回均值与方差；方差大→置信低，不轻易进 elite。
- **对比式评估（周期）**：除绝对 fitness 外，定期做 pairwise 对比（同 instance 上谁更好），维护相对排序（Bradley-Terry/Elo 风格）。相对排序对 instance 间尺度差异和噪声更鲁棒，用作 V_quality 的稳健补充。
- **反思回路**：当 plateau / best 不刷新 / 置信度普遍低时触发，对比近期 best vs worst rollout，产出 Pattern（LESSON/ANTI_PATTERN）注入 §6.C。

## 10. 主回路（完整流程）

```
TraceAAD.run():
    initialize()                                         # §11
    for t in range(max_iter):
        if not has_budget(): break
        T        = select_trajectory(TrajectoryMemory)   # §4 多维 value + UCB
        op       = portfolio.choose(T, phase=t)          # §7 bandit + 阶段
        ctx      = build_context(T, op, PatternMemory, critic)  # §6 三段式
        action   = llm_action(ctx)                       # stage 1
        for a in action:
            code  = llm_code(a, T.base)                  # stage 2
            fvec, conf, rt, cx = evaluate(code)          # §9 多 instance + 置信度
            if fvec is None: portfolio.update(op, FAIL); continue
            delta_vec = fvec − endpoint.fitness_vector
            gen_sig   = generalization_signal(delta_vec) # §5.2
            prog   = ProgramMemory.add(...)
            step   = Step(prog, op, mechanism_tag, delta_vec, outcome, gen_sig)
            new_T  = TrajectoryMemory.extend_or_branch(T, op, step)  # branch 用 §7 backtrack/crossover
            new_T.value = compute_value(new_T)           # §4
            if novelty_gate(new_T): TrajectoryMemory.add(new_T, island=...)
            portfolio.update(op, feedback(new_T, gen_sig))
        TrajectoryMemory.visit(T)
        if stagnant: islands.migrate(); force_phase_shift()         # §8
        if t % K_distill == 0: PatternMemory.distill(TrajectoryMemory)  # 蒸馏回路
        if reflect_trigger(): critic.reflect() → Patterns            # 反思回路
    return best by (V_quality, V_generalization)
```

返回不只看绝对 best，而看**质量 × 泛化**的 non-dominated 集——直接服务"可泛化算法"目标。

## 11. 初始化与终止

- 初始化：生成 `n_init` 个**机制族不同**的起点（强制跨机制族多样性，而非仅 thought 多样性），每个成一条 length-1 trajectory，分到不同 island。
- 终止：预算=已评估程序数。返回 non-dominated 程序集（含泛化标注），供下游按需选质量最优 / 泛化最优 / 折中。

## 12. 配置（关键项）

```
memory: { islands: 4, max_trajectories_per_island, pattern_capacity }
value:  { w_quality, w_potential, w_diversity, w_novelty, w_generalization, c0, top_k, temperature }
credit: { discount, lambda_pos, lambda_down, lambda_gen, robust_lambda }
similarity: { w_code, w_mechanism, w_trajectory, novelty_threshold }
portfolio: { alpha, beta_v, beta_n, delta_r, delta_c, tau_schedule }
feedback: { n_instances, contrastive_every, confidence_threshold }
reflection: { K_distill, patience_reflect, top_k_patterns }
```

## 13. 与早期原型 / 各方法的对照

相对早期原型的机制变化：

| 维度 | 早期原型 | 当前 TraceAAD |
| --- | --- | --- |
| trajectory | node 序列容器 | 带因果 Δ 向量的改进叙事 |
| credit | stepwise Δfitness(标量) | stepwise Δ**向量** + 泛化信号 |
| value | quality+path+simplicity+novelty | 上述 + **generalization 维度** |
| context | endpoint + step outcomes + lessons | **因果叙事 + 模式 + 对比**三段式 |
| 多样性 | 机制层 + novelty gate | **多层 + islands** |
| 反馈 | 单 fitness | **对比评估 + 置信度** |
| 算子 | 5 算子 | 6 算子(加 **Scale-Transfer** 直驱泛化) |
| 返回 | best 程序 | **质量×泛化 non-dominated 集** |

吸收/拒绝表见 §1 末。

## 14. 实现路径

仍落在 `llm4ad/method/traceaad/`，但重构 around 三层记忆：

- `memory/{program,trajectory,pattern}.py`（三层）
- `credit.py`（stepwise + 泛化）、`value.py`（多维）、`similarity.py`（多层）
- `context.py`（三段式）、`operators/{...}.py`（6 算子）、`portfolio.py`（bandit+阶段）
- `islands.py`、`feedback.py`（对比+置信度）、`reflection.py`
- `traceaad.py`（主回路串三回路）

平台接口不变（`LLM.draw_sample` / `SecureEvaluator` / `ProfilerBase`）。关键新增依赖：跨 instance 评估（task 已支持）、可选 code embedding（机制层相似度起步可先用 tag 粗粒度）。

---

一句话：**当前 TraceAAD 把"改进过程信息 + 泛化"立为核心，围绕三层记忆、三回路、多维 value、因果叙事 context、islands 多样性和鲁棒对比反馈组织搜索。** 它保留 trajectory 这一身份，但重新定义了 trajectory 是什么、值多少、怎么用。
