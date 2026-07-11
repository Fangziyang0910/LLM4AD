# TraceAAD 审计驱动改进计划与实现

## 背景与运行边界

- 依据：`docs/experiments/traceaad-full-mechanism-parameter-audit.md`。
- 被审计 artifact：`LLM4AD/experiments/tsp_construct/traceaad/20260708_203505/`。
- 该旧 run 已结束；2026-07-10 检查时没有 TraceAAD 进程或 tmux session 在运行。现有 PathWise 进程与本次改动无关，未触碰。
- 本次修改只影响后续新启动的 TraceAAD run，不改写旧 artifact，也不把旧结果解释成新机制结果。

## 改进计划

按审计暴露出的因果链推进，而不是先扫参数：

1. 修复搜索对象与 elite 生命周期：active-pool 稳健归一化、unique-path 去重、elite 保活和最小抽样概率、Pareto 生存、migration 移动现有 trajectory 而非 clone。
2. 修复 operator 信用：每 iteration 只按 best-of-batch 更新一次，统一 `[-1, 1]` reward，加入 global/near-record、EMA、概率下限、后期 novelty 上限和真实 LLM+evaluation 成本。
3. 修复 novelty 失败循环：提高停滞阈值，增加 operator/family cooldown，以 fresh-start 自身历史选择 family，不再用 surviving active count。
4. 修复知识信用：按 unique graph edge 蒸馏；使用真实 support；允许成功率下降；按 `operator x mechanism` 统计；修复否定语义；lesson 去重；anti-pattern 可恢复。
5. 收紧算子语义：backtrack 只从严格内部 base 分支；simplify 需同时满足相对复杂和停滞/平台条件；crossover 要求 donor 质量与互补性；scale-transfer 在无真实泛化证据时禁用。
6. 修复搜索循环与 gate：parse failure 不消耗 evaluation budget并补抽；global record 绕过 novelty gate；同机制同分行为重复可拒绝；初始化有限重试。
7. 修复参数和可复现性：明确 active cap，提升 quality 权重与 top-k，关闭伪泛化，降低无效回路频率，固定 task/search seed，使用实例私有 RNG，记录全部 value/portfolio 参数。
8. 加强可观测性：记录 batch 信用、真实成本、gate 原因、migration 前后数量与 island sizes；公开 active trajectory 和 portfolio 快照供测试/审计。
9. 完成定向、模块、全仓测试和双轴 code review，再做 scoped commit。

## 已落地机制

### Selection 与 survival

- selection/value/survival 对相同 `node_ids + edge_ids` 只保留一个代表；quality normalization 只使用 unique active endpoints，并做 10%/90% 分位裁剪，archived 节点和历史 outlier 不再压缩当前 pool。
- path potential 只有 endpoint quality 高于 `0.50` 时才生效，避免“低质终点 + 大幅恢复”主导选择。
- elite 总在候选集中，并有 `0.15` 直接抽样概率；每个 active island 至少贡献一个候选。
- UCB 后期保留 `0.05` floor，停滞时增加探索，而不是衰减到零。
- survival 使用 ValueVec 的 non-dominated fronts，scalar 只作同一 front 内排序；canonical global elite 先占保留名额。
- migration 轮换原 trajectory ID，保留 visit/value，不新增 clone；island 映射改为稳定 SHA-256。

### Portfolio 与 novelty

- 同一 operator iteration 只更新一次，使用 batch 内最佳候选；各算子 reward 统一裁剪到 `[-1, 1]`。
- global record 与 near-record 有独立信用；统计使用 EMA；所有 eligible operator 有 `0.05` 概率下限；late novelty 不超过 `0.20`。
- 成本包含 action LLM、code LLM 和 evaluator 时间，以 120 秒为 cost scale。
- novelty 默认在 stagnation `>=12` 后才 eligible，触发间隔至少 8 轮；同 family 连续失败 2 次后冷却 24 轮，所有 family 被阻塞时 novelty 不可选而不是绕过 cooldown。
- family 选择只看 novelty fresh-start 的 attempts/successes；生成内容标签优先于 requested hint。

### Pattern、feedback 与算子

- PatternMemory 对 support 幂等，按 `operator x mechanism` 统计 attempts、successes、rate 和 failure streak。
- distill 遍历 unique graph edges，不受 trajectory fork/archived 重复影响；rate 使用当前全部证据，可上升也可下降。
- prompt 显示 aggregate improve rate、当前 operator improve rate 和真实 unique support，不再把 scalar improve rate标成 generalization。
- 标签推断识别 remove/replace/avoid/without 等否定方向；实际 action/idea/code 优先，hint 只 fallback。
- RankingModel 的 Elo-like rank 已进入 best/worst contrast；Elo 只在连通比较分量内排序，未连通分量用 raw fitness 决定，避免局部胜者跨轨迹误排。
- backtrack、simplify、crossover、scale-transfer 的 trigger/base 语义按计划收紧。

### 搜索循环与初始化

- budget 现在表示 evaluator calls；program parse failure 会继续补抽，连续无 evaluation 的 iteration 有上限；portfolio phase/UCB/cooldown 按 `evaluator calls / actions_per_iteration` 推进，不被 parse failure 提前推入后期。
- 初始化也按 evaluator calls 计数并有限重试；四个默认 seed 明确对应 `nn_rank`、`local_density`、`row_normalize`、`sparsified_candidate`，前四个 seed 分配到四个 island。
- gate 保护使用生成当下的 live global best；同一 batch 的后续次优候选不能借旧 incumbent 绕过 gate。
- scalar-only 默认 robustness/generalization 为零；ValueVec 在 `w_generalization=0` 时也不携带伪泛化维度。只有显式允许且 evaluator 返回富 `EvalResult` 时才激活：`fitness_vector` 持久化到 ProgramNode，parent/child per-instance win/tie/loss 直接形成 step transfer signal。
- search seed 使用 TraceAAD 实例私有 RNG，并注入 trajectory selection 与 portfolio，不污染宿主进程或其他方法的随机流。

## 下一轮正式参数

正式入口：`LLM4AD/experiments/tsp_construct/traceaad/run_experiment.py`。

| 参数 | 新值 | 目的 |
| --- | ---: | --- |
| evaluation budget | 1000 | 保持与旧 run 可比，但按真实 evaluator calls 结束 |
| active / island cap | 160 / 40 | 消除名义 `max_active=1000` 的 inert 配置 |
| value weights | 0.50 / 0.20 / 0.15 / 0.15 / 0.0 | quality 优先，关闭无证据 generalization |
| top-k / temperature | 12 / 0.8 | 放宽 hard top-5，同时保留适度采样 |
| distill / reflect / migration | 20 / 20 / 20 | 降低重复知识与过频繁回路 |
| min reflect new edges | 8 | 没有新 evidence 时不重复反思 |
| evaluators | 1 | 当前逐 candidate 等待 future，多 worker 不产生方法级并发 |
| task / search seed | 2024 / 2024 | 固定数据和搜索随机性 |
| near-record tolerance / bonus | 0.10 / 0.25 | 保留接近纪录的高质量 fresh/operator 信用 |

`max_trajectory_length=8`、`novelty_threshold=0.92`、LLM temperature `1.0` 保留，因为单次旧 run 不能证明更优取值；它们进入后续 ablation，而不是在本轮凭观察拍值。

## 验证计划

代码层：定向测试覆盖 selection、operator、credit、run policy、smoke；随后运行全部 `test_traceaad*.py` 和全仓 pytest，并执行双轴 standards/spec review。

实验层不把一次新 run 当结论。固定模型、task split、1000 evaluator calls 和 seed，至少比较：

1. 旧 artifact（仅作历史参考，预算语义不同）；
2. audit-driven 完整版；
3. 完整版去掉 Pattern/reflect；
4. top-k `5 / 12` 与 trajectory length `8 / 12` 的小型 ablation；
5. 至少 3 个 search seeds。

核心指标：final best、best sample order、post-best conversion、elite selection share、operator/family attempts 与 upper-tail success、unique endpoint coverage、gate rejection/override、active endpoint concentration、LLM-hours per record。只有引入 held-out/per-instance/多规模 vector 后，才启用并评价 scale-transfer 与 generalization。

## 验证结果

- TraceAAD 专项：`74 passed`。
- 全仓：`171 passed, 2 subtests passed`。
- `py_compile`：TraceAAD package、operators、正式 runner 和 example runner 全部通过。
- `git diff --check`：通过。

### Standards review

首轮发现 3 项：未连通 Elo 跨分量误排、parse failure 提前推进 phase、全局 RNG 污染。三项均已修复并补回归测试；最终 re-review clean，无剩余 actionable finding。

### Spec review

首轮及复核共发现 6 项：伪 robustness、unique path 未去重、near-record 缺失、cooldown 可整体绕过、fitness vector 未进入信用、lesson 仅精确文本去重。六项均已修复并补回归测试；最终 re-review clean。多配置、多 seed 对照仍是后续实证验证，不是本次代码实现缺口。
