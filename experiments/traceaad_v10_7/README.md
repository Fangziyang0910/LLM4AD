# TraceAAD V10.7 / V10.7R

正式批次 `20260907_bounded_formal` 使用的是原 V10.7 `sampled_trajectory_v1`；其修复、48 次真实 smoke 及启动核查见[冻结启动记录](launch_20260907.md)，结果不应归到新的默认配置。

当前默认配置为 [V10.7R](../../docs/methods/TraceAAD-V10.7R-机制设计.md)：仍用一次调用生成 `Idea → Code`，但先确定 Refine、Pivot 或 Fuse 所需的证据角色，再从全档案选择材料。原 V10.7 设计与已有实验身份见[冻结机制页](../../docs/methods/TraceAAD-V10.7-机制设计.md)。

搜索树、8 根初始化、父代联合分布、R/P/F 请求比例 0.50/0.15/0.35、公共任务信息与真实评价预算沿用基线。V10.7R 不增加第二次校准、成功接力、行为探针或额外评价。

上下文只有一种机制：先确定 Refine、Pivot 或 Fuse 所需的证据角色，再从全档案选择材料。旧的 `ancestor_history`、`uniform_trajectory_v1`、`sampled_trajectory_v1` 已删除（`sampled_trajectory_v1` 仍是冻结批次 `20260907_bounded_formal` 的运行配置，其实现以提交 `ac6f4b9c` 为准）。

V10.7R 的 Refine 优先展示一份与底座相邻的真实形成对照；Pivot 优先展示控制结构不同的替代参考；Fuse 先按全局质量层选择迁移主参考，同层内优先控制结构不同的候选，再补一份跨质量对照。Refine 和 Pivot 不强行填满三份程序。没有能容纳的 donor 时 Fuse 回退 Refine。

`--max-context-programs=3` 包含底座；可用材料少或上下文不足时减少数量。Idea 视图最多 256 tokens，超长整段省略；出现 `Algorithm N` 或 `算法 N` 临时编号的历史 Idea 也整段省略。匹配该模式的 Python 注释只从提示视图移除，字符串、可执行代码与原始档案不变。输出 Idea 不超过 100 words，并须独立说明主要决策和关键计算。总输入默认上限 16128 tokens，不叠加祖先历史区。Init 尚无底座时保持从头生成，不采参考。

同代码随机选一条完整评价记录。质量层使用线性插值的 1/3、2/3 分位数，边界相等归较低层，同分总在同层。只对候选组合做精确容量检查，每个参考槽位最多无放回尝试 32 次。

单路运行示例：

```bash
uv run python -m experiments.traceaad_v10_7.run \
  --task tsp_construct \
  --run-name smoke_v107_tsp \
  --budget 10 \
  --backend local
```

生成 15 路正式运行计划并检查可启动项：

```bash
uv run python -m experiments.traceaad_v10_7.launch --dry-run
```

运行新批次时使用不同 `--run-name`（单路）或 `--batch` 和 `--session-prefix`（批量）。批量入口会把最大程序数传到每路命令。检查点严格校验源码、生成协议与机制配置；旧源码生成的检查点不按新语义恢复，旧策略批次只能用冻结代码续跑。

每路仍写入 `run_config.json`、`tree_state.json`、`pending_candidate.json`、`llm_calls.jsonl`、`events.jsonl`、`evaluations.jsonl`、`tokenizer_calls.jsonl` 和 `logs/run_summary.json`。`llm_calls.jsonl` 每个候选只有生成调用；不再记录 `thought_alignment` 阶段。事件不再包含摘要状态、摘要 token、独立摘要调用及分拆耗时字段，`llm_seconds` 表示唯一生成调用耗时。

采样记录按展示顺序的节点、代码哈希、角色、tokens、父代/donor 位置、质量层、容量拒绝、视图省略和参考不足。事件同时记录 `parent_delta`、`context_delta`、`frontier_delta`，以及同一父代代码和完全相同 Prompt 此前的尝试次数。参考曝光不增加父代选择次数。采样结果、原 Prompt、RNG 和计数前值在请求前一起持久化，恢复不重新采样。

已删除全档案精确 token 预筛。记录 sampling_seconds、scheduling_seconds 和 tokenizer_requests（缓存未命中的计数接口调用数，不含底层 HTTP 重试）；结合 llm_seconds、eval_seconds 和 tokenizer_calls.jsonl 分析成本。`--max-context-programs=1` 直接跳过参考采样。

验证命令：

```bash
.venv/bin/pytest -q tests/method/test_traceaad_v107.py tests/method/test_traceaad_v107_sampling.py tests/experiments/test_traceaad_v107_launch.py
```

训练可视化复用多版本监控入口，默认打开 V10.7，可在页面顶部切换 V10.6 等历史版本：

```bash
bash experiments/traceaad_v10_6/start_monitor.sh --restart
```

浏览器打开 <http://127.0.0.1:8765/?version=v10_7>，查看 15 路进度、最佳成绩曲线、候选评价散点、生成/评价耗时及节点 Idea、代码与血统。散点横轴使用真实评价编号（包含失败评价占用的预算）。
