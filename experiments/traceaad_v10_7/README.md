# TraceAAD V10.7

V10.7 用一次调用生成 `Idea → Code`，默认从全档案质量分层采样参考，与选中底座组成最多三份完整程序的上下文。原祖先历史作为基线模式保留。

搜索树、8 根初始化、父代联合分布、R/P/F 请求比例 0.50/0.15/0.35、公共任务信息与真实评价预算沿用基线。新模式的 Fuse donor 是已采参考中 fitness 最高的程序，允许同血缘；没有不同代码且能容纳的参考时回退 Refine。

| `--context-policy` | 上下文 |
| --- | --- |
| `ancestor_history` | 原 V10.7 祖先历史，继续使用 `--traj-gens`、`--history-tokens` 与 `--donor-topk` |
| `uniform_trajectory_v1` | 全档案去重后均匀抽参考 |
| `sampled_trajectory_v1`（默认） | 全档案去重后按质量分层抽参考 |

两个采样模式共用相同程序格式、按内部 fitness 从低到高的排序及算子指令，只改变参考抽样概率。`--max-context-programs=3` 包含底座；可用材料少或上下文不足时减少数量。完整 Idea 和原始归档代码均不截断，也不叠加祖先历史区；总输入默认上限 16128 tokens。Init 尚无底座时保持从头生成，不采参考。

同代码随机选一条完整评价记录；在可容纳的候选上以线性插值的 1/3、2/3 分位数划分 low/middle/high，边界相等归较低层，同分总在同层。先均匀抽非空层，再均匀抽程序，第二份优先其它层。先检查单参考是否可容纳，再为每个位置最多无放回尝试 32 次组合容量检查。

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

容量失败后在当前质量层无放回重试，层耗尽才换层，每个槽位总计最多 32 次。精确的单参考预筛对新父代仍可能需要 O(档案大小) 次 tokenizer 请求，32 次限制不覆盖预筛；缓存只复用完全相同的请求。该开销在 `tokenizer_calls.jsonl` 中记录，应与输入 token、LLM 耗时一起纳入实验成本；目前没有通过真实服务测量该开销。`--max-context-programs=1` 会直接跳过参考预筛。

验证命令：

```bash
.venv/bin/pytest -q tests/method/test_traceaad_v107.py tests/method/test_traceaad_v107_sampling.py tests/experiments/test_traceaad_v107_launch.py
```
