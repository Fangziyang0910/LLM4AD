# PathWise and MCTS-AHD potential stall audit

日期：2026-07-09
分类：worklog

本次检查目标是确认 MCTS-AHD 修复 `s1` 后是否还有类似长跑卡点，并对照 `reference_code/PathWise/` 检查当前 `llm4ad.method.pathwise` 是否保持原始机制。

## 验证

在 `/home/fang/code/LLM4AD/LLM4AD` 中运行：

```bash
uv run ruff check llm4ad/method/mcts_ahd llm4ad/method/pathwise tests/method/test_mcts_ahd_mechanics.py tests/method/test_pathwise_mechanics.py --select F821,F822,F823 --output-format=full
uv run pytest tests/method/test_mcts_ahd_mechanics.py tests/method/test_pathwise_mechanics.py -q
```

结果：未发现未定义变量类问题；机制测试 `26 passed`。

## MCTS-AHD

未再发现和 `s1` 的 `NameError` 同类的直接未定义变量问题。当前三个 TSP construct tmux run 仍在推进，日志持续有 `HTTP 200 OK` 与 `expand` 事件。

仍有一个潜在长跑风险：`_sample_evaluate_register()` 只有在成功 parse 并完成 evaluation 后才递增 `_tot_sample_nums`。如果 LLM 长时间返回无法解析的文本，函数会返回 `False`，但不消耗 sample budget，也不累计 consecutive failure。初始化循环遇到这种情况会 `continue`，理论上可能长时间没有预算进度。这个点是否修，需要先确认原始 MCTS-AHD 对无效 parse 是否计入 `max_fe`。

## PathWise

当前集成版不是和原始 PathWise 完全一致，主要差异如下：

- 原始 world-model rollout 对无效输出会最多 retry 3 次，失败后用 parent/population code fallback；当前集成版 parse 失败只计一次 invalid sample，然后返回无效 rollout。
- 如果一个 inner step 没有 valid rollout，当前集成版会让 `_construct_entailment_graph()` 返回 `final_node=None`，外层 run 直接 break；原始代码在没有 entailment 时会 fallback 到当前 population 的最好个体并继续。
- 原始 policy action 对 invalid parent selection 会 retry 1 次，仍失败则跳过；当前集成版直接使用 fallback action。
- 当前 `PathWise.run()` 捕获普通异常后记录 error，但 finally 仍会写 `status=finished`，除非 search 被显式 abort。这和 MCTS-AHD 修复前的误判风险类似。
- 当前默认 `max_sample_nums=100`，原始主配置 `config_pathwise.yaml` 为 `max_fe=500`；若做原始实验复现，应显式对齐。
- 当前 prompt 是按 LLM4AD 高分约定重写的紧凑 prompt，不是直接使用原始 `prompts/common/*.txt`。
- 原始 inner step 对 actions 并行处理；当前集成版 sequential 地跑 actions/rollouts，只在 evaluation executor 层并行。该点主要影响吞吐与随机调用顺序。

机制层面最值得优先修的是 PathWise 的 world-model retry/fallback、no-entailment fallback，以及 run summary 的错误状态。

## 2026-07-09 follow-up

复查 MCTS-AHD 原始实现后，确认 `ec_fe_max` / `eval_times` 是 function evaluation budget。原始 `InterfaceEC.get_algorithm()` 和 `evolve_algorithm()` 在调用 evaluation 前递增 `eval_times`，但 LLM/parse 阶段在 `get_offspring()` / `Evolution._get_alg()` 内部 retry，不把裸 LLM 请求次数计入 `ec_fe_max`。因此 LLM 请求异常不计入 LLM4AD 集成版 `_tot_sample_nums` 是合理的；当前 `max_sample_nums=1000` 对应原始 MCTS-AHD TSP 的 1000 次 function evaluation budget，而不是 1000 次 LLM API 请求。

已按原始 PathWise 失败处理补齐集成版：

- policy invalid parent selection 先 retry 1 次，仍失败则跳过该 action；LLM 请求异常才 fallback 到 best-state action。
- world-model invalid output 最多 retry 3 次，仍失败则 fallback 到 parent/population code，并对 fallback code 做一次 evaluator 评估。
- 如果 inner entailment 没有产生 final node，则 fallback 到当前 population 的最好个体，并通过 evaluator 形成 fallback node，避免外层提前 break。
- `PathWise.run()` 对未处理异常写 `status=error`，对 `KeyboardInterrupt` 写 `status=interrupted`，显式 abort 仍写 `status=aborted`，避免假 `finished`。

验证：

```bash
uv run pytest tests/method/test_mcts_ahd_mechanics.py tests/method/test_pathwise_mechanics.py -q
uv run pytest tests/method -q
uv run ruff check llm4ad/method/mcts_ahd llm4ad/method/pathwise tests/method/test_mcts_ahd_mechanics.py tests/method/test_pathwise_mechanics.py --select F821,F822,F823 --output-format=full
git diff --check
```

结果：MCTS + PathWise 机制测试 `29 passed`；全 method 测试 `91 passed, 2 subtests passed`；未定义变量检查通过；diff whitespace 检查通过。

## 2026-07-09 PathWise budget default

按原始 PathWise 主配置 `config_pathwise.yaml` 的 `max_fe=500`，将 LLM4AD 集成版默认预算从 100 调整为 500：

- `llm4ad/method/pathwise/paras.yaml`: `max_sample_nums: 500`
- `PathWise.__init__`: `max_sample_nums` 默认值同步为 500
- `max_fe` alias 覆盖逻辑同步从旧默认 100 改为新默认 500

验证：`uv run pytest tests/method/test_pathwise_mechanics.py -q` 结果 `9 passed`；未定义变量检查通过；`git diff --check` 通过。
