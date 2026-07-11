# MCTS-AHD reference mechanism alignment

2026-07-09

本次把 LLM4AD 内部 `llm4ad.method.mcts_ahd` 的几个机制点进一步对齐 `reference_code/MCTS-AHD-master/` 原始实现。重点只放在会影响搜索行为和 prompt context 的地方，平台接口、profiler、LLM adapter 和 evaluator 抽象保持 LLM4AD 风格。

## 对齐内容

- 初始化改为原始顺序：先用 `i1` 生成第一个 root child，然后用当前 `brothers` 集合连续执行 `e1`，直到 `init_size`。MCTS root children 直接包含全部初始化兄弟节点；population survival 不再决定 root 初始节点集合。
- 搜索循环重新维护原始意义上的 `nodes_set`，作为全局候选/精英上下文传给 expansion。MCTS 树仍保留所有展开节点，population survival 只管理精英集合。
- `e1` reference sampling 改为 `random.choices`，允许重复，匹配原始 `parent_selection_e1` 的 uniform-with-replacement 行为；root 只有一个已有分支时也允许 `e1` 扩展。
- progressive widening 条件改为原始严格条件 `int(visits ** alpha) > len(children)`。
- `e2` 的另一个父代优先从传入的 `nodes_set` 中按 rank probability 选择，再追加当前 father，匹配原始 `evolve_algorithm(pop, father, "e2")` 的父代构造语义。
- `s1` path population management 按原始 `nlargest(objective)` 的等价负分语义处理：LLM4AD 中 score 是负 objective，因此对应为 `nsmallest(score)`；去重依据也改回 algorithm 描述。
- base prompt 的函数上下文更接近原始 `Evolution` / `problem_adapter`：描述要求放在 brace 中，函数名、输入参数名和输出名显式列出；TSP constructive 的输出名对应 `next_node`。

## 验证

```bash
cd /home/fang/code/LLM4AD/LLM4AD
uv run python -m py_compile llm4ad/method/mcts_ahd/mcts_ahd.py llm4ad/method/mcts_ahd/prompt.py
uv run pytest tests/method/test_mcts_ahd_mechanics.py -q
```

结果：`27 passed, 2 subtests passed`。

## 论文实验设置补充

`papers/MCTS-AHD/icml2025.tex` 中主实验设置为：对每个 application scenario，每个 LLM-based AHD 方法做 3 次 independent runs，报告平均性能；绝大多数任务的设计阶段 evaluation budget 为 `T=1000`。表格 caption 也分别确认 TSP/KP、ACO、online BPP、BO 等主结果均按 3 次运行取平均。

例外需要单独标注：显著性检验 appendix 对部分任务扩展到 up to 10 runs；ablation/曲线图通常是 5 次或 10 次；TSPLib 部分先从 GPT-4o-mini 的 3 次设计运行中取 best heuristic，再对每个实例用不同 starting nodes 跑 3 次取平均。

## LLM 请求参数补充

论文明确写出的 LLM 侧设置是模型版本和 temperature：`GPT-4o-mini` 对应 `gpt-4o-mini-2024-07-18`，`GPT-3.5-turbo` 对应 `gpt-3.5-turbo-0125`，并固定 `temperature=1.0`。论文没有明确给出 `top_p`、`max_tokens`、LLM request timeout 等请求参数。

原始代码中 `cfg/llm_client/openai.yaml` 也只配置 `model: gpt-4o-mini` 和 `temperature: 1.0`；`utils/llm_client/openai.py` 调用 OpenAI Chat Completions 时显式传入 `model`、`messages`、`temperature`、`n`、`stream=False`，未传 `top_p` 或 `max_tokens`。因此复现实验时，`temperature=1.0` 是论文/原始代码明确依据；其他请求参数若在 LLM4AD runner 中设置，属于平台/后端适配参数，应单独记录。
