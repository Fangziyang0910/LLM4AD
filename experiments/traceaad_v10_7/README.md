# TraceAAD V10.7

正式批次 `20260907_bounded_formal` 已启动 15 路；修复、48 次真实 smoke 及启动核查见[启动记录](launch_20260907.md)。

V10.7 用一次调用生成 `Idea → Code`，默认从全档案质量分层采样参考，与选中底座组成最多三份完整程序的上下文。原祖先历史作为基线模式保留。

搜索树、8 根初始化、父代联合分布、R/P/F 请求比例 0.50/0.15/0.35、公共任务信息与真实评价预算沿用基线。新模式的 Fuse donor 是已采参考中 fitness 最高的程序，允许同血缘；没有不同代码且能容纳的参考时回退 Refine。

| `--context-policy` | 上下文 |
| --- | --- |
| `ancestor_history` | 原 V10.7 祖先历史，继续使用 `--traj-gens`、`--history-tokens` 与 `--donor-topk` |
| `uniform_trajectory_v1` | 全档案去重后均匀抽参考 |
| `sampled_trajectory_v1`（默认） | 全档案去重后按质量分层抽参考 |

两个采样模式共用相同程序格式、按内部 fitness 从低到高的排序及算子指令，只改变参考抽样概率。`--max-context-programs=3` 包含底座；可用材料少或上下文不足时减少数量。原始归档代码完整展示，Idea 视图最多 256 tokens，超长整段省略，容量紧张时还可省略 Idea。原始 Idea 留档；输出要求不超过 100 words。总输入默认上限 16128 tokens，不叠加祖先历史区。Init 尚无底座时保持从头生成，不采参考。

同代码随机选一条完整评价记录；直接在全部去重候选上以线性插值的 1/3、2/3 分位数划分 low/middle/high，边界相等归较低层，同分总在同层。先均匀抽非空层，再均匀抽程序，第二份优先其它层。只对抽中的程序做精确组合容量检查，每个位置最多无放回尝试 32 次。

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

运行对照时分别指定上述三个 `--context-policy`，并使用不同 `--run-name`（单路）或 `--batch` 和 `--session-prefix`（批量）。批量入口会把模式与最大程序数传到每路命令，并拒绝用不同配置恢复同一 manifest。检查点严格校验源码与机制配置；旧检查点不按新语义恢复。

每路仍写入 `run_config.json`、`tree_state.json`、`pending_candidate.json`、`llm_calls.jsonl`、`events.jsonl`、`evaluations.jsonl`、`tokenizer_calls.jsonl` 和 `logs/run_summary.json`。`llm_calls.jsonl` 每个候选只有生成调用；不再记录 `thought_alignment` 阶段。事件不再包含摘要状态、摘要 token、独立摘要调用及分拆耗时字段，`llm_seconds` 表示唯一生成调用耗时。

采样模式记录 `context_policy`、按展示顺序的 `context_node_ids`、实际程序数、各程序 tokens、父代/donor 的展示位置、质量边界与层、容量拒绝、抽取过程及不足数量。`prompt_tokens` 是完整 chat 输入的精确计数。来源 ID 只进入日志；Prompt 仅包含任务、程序材料、角色与输出指令。参考曝光不增加父代选择次数。采样结果、原 Prompt 和 RNG 状态在发出请求前一起持久化，恢复不重新采样。

容量失败后在当前质量层无放回重试，层耗尽才换层，每个槽位总计最多 32 次。已删除全档案精确预筛。记录 sampling_seconds、scheduling_seconds 和 tokenizer_requests（缓存未命中的计数接口调用数，不含底层 HTTP 重试）；结合 llm_seconds、eval_seconds 和 tokenizer_calls.jsonl 分析成本。`--max-context-programs=1` 直接跳过参考采样。

验证命令：

```bash
.venv/bin/pytest -q tests/method/test_traceaad_v107.py tests/method/test_traceaad_v107_sampling.py tests/experiments/test_traceaad_v107_launch.py
```

训练可视化复用多版本监控入口，默认打开 V10.7，可在页面顶部切换 V10.6 等历史版本：

```bash
bash experiments/traceaad_v10_6/start_monitor.sh --restart
```

浏览器打开 <http://127.0.0.1:8765/?version=v10_7>，查看 15 路进度、最佳成绩曲线、候选评价散点、生成/评价耗时及节点 Idea、代码与血统。散点横轴使用真实评价编号（包含失败评价占用的预算）。
