# TraceAAD 经验重构计划

## 1. 决策与意图

本次重构采用无标签的边级经验检索（方案 B）：

> 保留「父代 → 算子 → action → 子代 → 效果」这一真实实验记录，不再将 action 归入预设 `mechanism_tag`。后续搜索直接检索典型成功/失败 action，不再依赖任务相关的机制词表。

目标是消除 TSP 机制词表对通用算法设计的归纳偏置，同时保留 TraceAAD 「从历史试错中获取上下文」的核心思想。

## 2. 问题与证据

当前 `_MECHANISM_KEYWORDS` 明确面向 TSP constructive，包含 `local_density`、`nn_rank`、`row_normalize`、`hybrid_distance` 等路由任务语义。这些标签同时进入了经验统计、轨迹相似度、Crossover donor、Novelty Jump 和 island 分配，影响不是局部的。

标签还存在因果归属问题。当前从 `action + idea + 完整子代代码` 中按固定顺序匹配第一个关键词，容易识别子代继承的旧机制，而不是本次 action 引入的变化。现有日志中，同一个 Gravity-Based Candidate Selection action 曾被标为 `local_density` 和 `row_normalize`。三组正式路由实验中，`local_density` 分别占子代标签的约 $48.4\%$、$55.8\%$ 和 $51.0\%$，表明统计已被词表顺序和完整代码污染。

因此，本次不把现有词表换成另一套「通用词表」。固定的修改类型仍无法稳定表示开放的算法机制，也不应用于硬过滤或信用分配。

## 3. 目标与非目标

### 3.1 目标

1. TraceAAD 核心实现中不存在 task-specific 机制词表。
2. 推导边完整保留父子关系、算子、action、有向 $\Delta$、outcome 和 iteration。
3. refinement prompt 能从当前 run 的真实边中获取少量成功/失败 action 作为经验。
4. Crossover、Novelty、相似度与 island 分配只使用任务无关信号。
5. 不增加 LLM 调用，不引入 embedding 模型或额外服务。

### 3.2 非目标

- 不实现开放词汇、动态聚类或 LLM 机制归纳（方案 C）。
- 不统计不同自然语言 action 之间的「同类成功率」。
- 不改变轨迹价值、UCB、OperatorPortfolio 信用或评估预算口径。
- 不重写历史实验工件中已记录的 `mechanism_tag`。

## 4. 设计原则

1. **事实与归纳分离。** `DerivationGraph` 仍是实验事实源；经验检索只是对图边的有界视图。
2. **不确定语义不进入硬控制。** 没有可验证的分类时，不根据自然语言标签禁用算子、donor 或探索方向。
3. **小界面。** 实现一个具体的 `ExperienceMemory` module，不预先建立 adapter 或动态词库抽象。
4. **上下文有界。** 经验块默认最多注入 $2$ 条成功与 $2$ 条失败 action，每条 action 最多保留 $300$ 个字符。
5. **先建立无标签基线。** 若无标签经验检索不能带来可验证收益，则直接保留短期轨迹叙事，不继续扩张长期经验机制。

## 5. 目标机制

### 5.1 边级实验记录

`ImprovementEdge` 保留：

```text
id, parent_id, child_id, action, operator, delta, outcome, iteration
```

删除 `ProgramNode.mechanism_tag` 和 `ImprovementEdge.mechanism_tag`。节点表示程序状态，边表示一次具体修改，不再把二者强制压缩成单一机制类别。

Novelty Jump 没有父边，不进入 action 经验检索。其成败仍由 OperatorPortfolio 按整个算子更新。

### 5.2 `ExperienceMemory` module

`ExperienceMemory` 作为 `DerivationGraph` 上的只读经验视图，不复制图边，不维护第二份事实。其外部界面保持为一个查询：

```python
examples(
    *,
    operator: str,
    positive_k: int = 2,
    negative_k: int = 2,
) -> ExperienceBatch
```

查询规则：

1. 仅检索 action 非空的 refinement 边。
2. 成功例来自 `outcome=improve`，失败例来自 `outcome=regress`；首版不注入 plateau。
3. 优先返回当前 operator 产生的记录；数量不足时，用其它 operator 的全局记录补齐。
4. 成功例按有向 $\Delta$ 降序；失败例按 $\Delta$ 升序（最强退步优先）；并列时优先 iteration 较新者。
5. 对去首尾空白并合并连续空白后完全相同的 action 全局去重；同一 action 同时出现成功和失败记录时，优先保留当前 operator 的记录，再保留绝对 $\Delta$ 最大者，并列时取较新记录；不做语义聚类。
6. `ExperienceBatch` 返回结构化示例，prompt 格式化仍由 `context.py` 负责。

该 module 隐藏检索、去重、排序和 fallback 规则；调用方只需指定当前算子和数量上限。

### 5.3 动作上下文

`build_action_prompt` 将 `[Mechanism Patterns]` 替换为 `[Past Action Evidence]`：

```text
Successful past actions:
- [operator=...] action=... delta=...

Failed past actions:
- [operator=...] action=... delta=...
```

这些示例是当前 run 内的任务内经验，但它们的产生和检索不依赖任务词表。现有的轨迹最近 $5$ 步因果叙事和 RankingModel best/worst 对比保留，但从对比块中删除 mechanism 字段。

### 5.4 初始化

删除 `_INIT_MECHANISM_HINTS`。第一个初始候选要求一个简单、完整的有效方案；后续初始候选在 prompt 中列出已生成的简短 idea，要求使用明显不同的算法思路。这只是 run-local 去重，不预设任务机制。

### 5.5 相似度与轨迹价值

删除 `mechanism_profile` 和 `mechanism_similarity`。多层相似度改为：

$$
\mathrm{sim}(\tau_a,\tau_b)
=0.7\,\mathrm{sim}_{code}+0.3\,\mathrm{sim}_{trajectory},
$$

其中 `code` 仍是终点程序 token Jaccard，`trajectory` 仍是 `(operator, outcome)` 行为指纹 Jaccard。同步删除 `ValueWeights.w_sim_mechanism`，将默认权重调整为 `w_sim_code=0.7`、`w_sim_trajectory=0.3`。Diversity、Novelty 与 novelty gate 的其它定义不变。

### 5.6 Crossover

保留现有 `mechanism_crossover` 算子名，避免不必要的日志口径变更，但其 donor 选择不再读取 mechanism 统计或反模式。

Crossover 始终可触发；donor 按互补性与质量软排序（不因门槛硬禁用）：

$$
\begin{aligned}
\mathrm{comp}&=1-\left(0.7\,\mathrm{sim}_{code}+0.3\,\mathrm{sim}_{trajectory}\right),\\
\mathrm{donor\_score}&=\mathrm{comp}+0.3Q.
\end{aligned}
$$

Crossover 约束只提供 donor idea，要求移植一个明确的算法思路；不再提供伪机制族名称。

### 5.7 Novelty Jump 与 island

Novelty Jump 始终可作为探索候选（无停滞门槛、无算子冷却硬门控），删除：

- `_CANDIDATE_FAMILIES`；
- 机制族 Beta 后验选择；
- family failure streak / cooldown；
- anti-pattern 过滤；
- 全局 best 停滞门槛与 novelty 触发冷却。

探索频率改由 OperatorPortfolio 的 EMA、阶段 bonus 与 late novelty 概率上限调节。

fresh-start prompt 改为要求一个与当前活跃精英 idea 明显不同的完整方案，并可列出最多 $4$ 个已有 idea 作为避免重复的参考。生成后仍由程序/轨迹相似度 novelty gate 决定是否保留。

Novelty 新起点不再按 mechanism hash 分岛，而是分配到当前活跃轨迹最少的 island，并在并列时选择编号最小者。初始化仍使用 `slot % n_islands`。

### 5.8 删除周期机制归纳

删除 `PatternMemory`、`Pattern`、`aggregate_patterns` 和基于 mechanism best/worst 的周期 `reflect`。同时删除构造参数：

```text
pattern_aggregate_interval
patience_reflect
min_reflect_new_edges
```

RankingModel 仍在每次 refinement 前生成当前 best/worst 对比，但只展示 fitness 与 idea。迁移周期 `migration_interval` 保留。

## 6. 涉及的代码与文档

| 位置 | 计划改动 |
| --- | --- |
| `schema.py` | 删除 node/edge 的 `mechanism_tag` 和 `Pattern`；增加经验查询返回结构 |
| `derivation_graph.py` | 删除 mechanism 字段传递，保留完整边级事实 |
| `experience_memory.py` | 新增无标签边级经验检索 module |
| `pattern_memory.py` / `pattern_loops.py` | 删除 |
| `operators/base.py` | 删除词表、标签推断及 PatternMemory 依赖 |
| `context.py` | 机制模式块替换为成功/失败 action 经验块 |
| `similarity.py` / `value.py` | 删除机制相似度，调整两层相似度权重 |
| `operators/crossover.py` | donor 改为代码/轨迹互补性 + 质量 |
| `operators/novelty.py` | 删除预设族与 family 冷却，改为开放 fresh start |
| `islands.py` | mechanism hash 改为 least-loaded 分配 |
| `feedback.py` | best/worst 对比去除 mechanism 输出 |
| `traceaad.py` | 初始化、生成评估、上下文、日志、周期 hook 与构造参数收口 |
| `__init__.py` | 公开导出从 `PatternMemory` 改为 `ExperienceMemory` |
| `experiments/*/traceaad` | 删除已废弃的 pattern/reflection 参数；保持实验预算不变 |
| `tests/method` | 删除标签词表测试，增加无标签经验、算子与跨任务测试 |
| `docs/ideas/TraceAAD完整机制设计.md` | 按实现重写三层记忆、相似度、Crossover、Novelty 与上下文章节 |
| `docs/ideas/AAD搜索机制综合.md` | 将 TraceAAD 的经验压缩描述从机制聚合更新为无标签 action 检索 |
| `docs/worklog/2026-W29.md` | 实现完成后记录重构意图、验证与阶段判断 |

历史日志和结果页不回写；它们保留当时实验的真实语义。

## 7. 实施顺序

实施前先固化重构前 A 版基线：确认当前变更的所有权，并用 scoped commit 或可追溯 patch 保存代码、测试与 runner 状态。不在无可追溯基线的脏工作树上直接开始删除机制字段。

### 阶段 1：事实模型与经验检索

1. 先为 `ExperienceMemory.examples` 写失败测试。
2. 增加 `ExperienceBatch/ExperienceExample`，实现去重、排序、operator 优先与全局 fallback。
3. 从 schema 和 derivation graph 删除 `mechanism_tag`。
4. 确保图边、轨迹延伸和信用分配测试通过。

### 阶段 2：上下文替换

1. 将 action prompt 的模式块替换为经验示例块。
2. 保留短期轨迹因果叙事和 best/worst 对比。
3. 对 prompt 顺序、数量上限、文本截断和空记忆行为增加测试。

### 阶段 3：任务无关的多样性与算子

1. 删除机制相似度，收口 novelty/diversity 权重。
2. 改写 Crossover donor 选择和约束文本。
3. 改写 Novelty Jump 的 trigger、fresh-start 约束和 island 分配。
4. 删除初始化 TSP hints，改为基于已有 idea 的 run-local 多样化。

### 阶段 4：删除旧经验回路

1. 删除 `PatternMemory`、`Pattern`、`pattern_loops` 和周期 hook。
2. 删除旧构造参数、runner 字段、日志字段与公开导出。
3. 全局搜索残留的预设机制名、`mechanism_tag`、`PatternMemory` 与 `aggregate_patterns`。

### 阶段 5：文档与实验入口收口

1. 按最终代码更新完整机制设计文档。
2. 更新当周 worklog，记录设计判断和验证。
3. 核对 TSP/CVRP/OP runner 的参数与 `run_config.json` 输出。

每个阶段单独完成相关测试，不在中间状态保留两套并行的经验系统。

## 8. 测试计划

### 8.1 `ExperienceMemory` 单元测试

- 同一图边只返回一次。
- 完全相同的 action 文本在成功/失败两组间也只返回一次，不同 action 不被强制合并。
- 成功/失败分组和 $k$ 上限正确。
- 当前 operator 证据优先，不足时全局 fallback。
- 排序对 maximize/minimize 任务一致；使用已定向的 `edge.delta`，不在经验 module 内再次翻转。
- 空图、只有 plateau、action 为空等情况返回空结果。

### 8.2 Prompt 测试

- refinement prompt 显示最多 $2$ 条成功和 $2$ 条失败 action。
- 不出现 mechanism family、aggregate improve rate 或 anti-pattern 文本。
- 空经验时有简短空状态，不影响 action 输出格式。
- 初始化和 Novelty prompt 不包含 distance、density、nearest-neighbor 等 TSP 预设。
- best/worst 对比仅显示 idea 和 fitness。

### 8.3 相似度与算子测试

- 两层相似度权重归一化，结果保持在 $[0,1]$。
- 改变旧 mechanism 文字不再影响 diversity/novelty。
- Crossover 拒绝低质量或低互补 donor，并在合格者中选择新公式最高者。
- Crossover trigger 不依赖任务词表或经验标签，且始终可候选。
- Novelty 始终可候选，由 portfolio 调节频率；不再存在「所有 family 被禁用」的死路。
- Backtrack / Simplify 仅保留结构性可行性 trigger。
- Novelty 新起点进入当前最小 island，并列行为确定。

### 8.4 集成与策略测试

- 用 stub LLM / evaluator 跑通初始化、refinement、Crossover、Novelty、survival 和最终返回。
- 使用不包含 routing 语义的任务模板和 action，确认不需要任何机制词表即可运行。
- 评估失败不建边，因此不进入经验检索。
- 样本预算、parse failure、随机种子、elite 保护和轨迹访问计数行为不因重构变化。
- 日志保留 parent/child/edge/operator/action/delta/outcome，不再输出误导性 `mechanism_tag`。

### 8.5 回归检查

```bash
uv run pytest tests/method/test_traceaad_experience.py -q
uv run pytest tests/method/test_traceaad_operators.py -q
uv run pytest tests/method/test_traceaad_selection.py -q
uv run pytest tests/method/test_traceaad_run_policy.py -q
uv run pytest tests/method -q
uv run pytest -q
git diff --check
```

额外做静态残留检查：

```bash
rg "_MECHANISM_KEYWORDS|_CANDIDATE_FAMILIES|mechanism_tag|PatternMemory|aggregate_patterns" \
  llm4ad/method/traceaad tests/method docs/ideas/TraceAAD完整机制设计.md
```

该命令应无命中。历史实验目录不在清理范围内。

## 9. 验收条件

1. TraceAAD 当前代码、测试和权威机制文档中无预设机制词表与 `mechanism_tag` 依赖。
2. 边级事实完整，经验块可稳定返回有界的成功/失败 action。
3. Crossover、Novelty、similarity、island 和初始化在不含 routing 词汇的集成测试中正常工作。
4. 所有 TraceAAD 定向测试与全仓测试通过，`git diff --check` 无误。
5. 没有新增 LLM 调用、embedding 依赖或隐式评估预算。
6. 完整机制设计文档与最终代码一致，不再描述已删除的机制聚合和反模式。

## 10. 实验验证

代码验收后再进行方法效果验证，不把长实验作为重构代码的前置条件。

对比组：

- A：重构前固定标签 TraceAAD（由当前 commit / 已有工件保留）；
- B0：去除固定标签，但不注入跨轨迹 action 经验，仅保留短期轨迹叙事；
- B：无标签边级经验检索 TraceAAD。

任务至少覆盖：

1. 一个路由任务，用于观察删除旧归纳偏置后是否发生灾难性退化；
2. 一个非路由组合优化任务，如 online bin packing 或 JSSP；
3. 一个 machine learning 或 science discovery 任务。

各组使用相同的 evaluation budget、LLM call budget、model、prompt 输出数和并发配置，每个 task 至少 $3$ 个 repeat。比较最终 held-out 分数、达到最优分的 sample 位置、合法样本率、后期改进率、各算子使用/改进情况和 prompt token 开销。

方法判断：

- 若 B 在非路由任务上比 A 更稳定，相对 B0 也有一致收益，且在路由任务上无明显灾难性退化，则保留无标签经验检索。
- 若 B 不能稳定超过 B0，则删除 `ExperienceMemory` prompt 块，仅保留短期轨迹因果叙事和 OperatorPortfolio。
- 本轮不因实验结果转入动态词库方案 C；C 需另行立项。

## 11. 风险与控制

| 风险 | 控制 |
| --- | --- |
| 无语义聚类时，历史 action 与当前 base 可能不相关 | 优先当前 operator，仅注入 $2+2$ 条，并保留当前轨迹上下文为主信号 |
| 强退步 action 过长，增加 prompt 开销 | 每条 action 固定长度截断，并在 action 调用日志记录经验块字符数 |
| 删除机制相似度后 novelty gate 分布变化 | 对两层相似度、gate 阈值和接受率增加定向测试与运行日志 |
| Crossover 缺少语义互补性 | 使用代码差异 + 轨迹行为差异 + 质量门槛，并通过算子消融判断是否保留 |
| 旧参数或文档残留导致两套语义并存 | 代码、runner、测试和权威文档做全局 `rg` 检查，历史 artifact 明确排除 |

## 12. 停止条件

本次重构在以下条件满足后停止：

1. 方案 B 的代码、测试、runner 和完整机制文档全部收口；
2. 无预设机制词表或其变体留在当前 TraceAAD 实现中；
3. 定向测试、全仓测试和静态残留检查通过；
4. 形成可启动 A/B 长实验的稳定入口。

动态词库、语义聚类、embedding 检索和额外 reflector LLM 均不属于本次实现范围。
