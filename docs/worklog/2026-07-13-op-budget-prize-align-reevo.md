# OP 任务 budget/prize 配置对齐 ReEvo/DeepACO 标准

日期：2026-07-13
分类：worklog

## 背景
首个 OP（`orienteering_construct`）实验发现 best 从 sample 6 起卡死在 27.43 不动，三个 run 完全一致。定位到 LLM4AD 官方 OP 实现的 budget 设置不对齐标准。

## 根因
`get_instance.py` 默认 `max_length_ratio=0.35` → budget = 0.35×problem_size。OP50 下 budget=17.5 远超单位正方形内 TSP50 最优 tour（≈5.5），约束永不触发，OP 退化为无约束，所有启发式都收完所有 prize（best=27.43 = sum(prizes) 均值上界），无区分度。prize 也用 `Uniform[0.1,1.0]` 而非 ReEvo 的 Kool2019 离散分布。

## 修正（仅改 1 个文件）
`llm4ad/task/optimization/orienteering_construct/get_instance.py`：
- budget：新增 `_STANDARD_OP_MAX_LENGTH = {50:3, 100:4, 200:5, 500:8, 1000:12}`（ReEvo/DeepACO 标准），按 problem_size 取值；非标准 size 保留 `max_length_ratio×size` fallback。
- prize：改为 `p_i = (1+⌊99·d_{0i}/max_j d_{0j}⌋)/100`，depot prize=0。

`run_experiment.py` / `generated_data_config.py` / `Evaluation` 接口均未改 —— OP50 自动走标准分档。

## 影响与注意事项
- **影响所有 OP 实验**：OP50 现在用 budget=3（而非 17.5），分数 scale 完全不同。后续 pathwise/traceaad 在 OP 上自动受益（共用 `get_instance`）。
- **`run_config.json` 里的 `max_length_ratio=0.35` 是 legacy 字段**，对 OP50 实际无效（被标准分档覆盖）。读 run_config 时注意：OP50 真实 budget=3，而非 0.35×50。
- **基线参考**：修正后 ratio 启发式（prize/距离贪心）在 OP50 train 上得 14.07（上界 29.01 的 48.5%），default 启发式得 3.71。可作为各 method 的参考下界。
- **与 ReEvo 的可比性**：配置（budget/prize）现已对齐，但 ReEvo 论文 OP 用 **ACO 框架**（多蚁协作 + 信息素），不是 construct 单步启发式，数值不直接可比；要对比需自行实现 ACO baseline 或比 evolution curve 趋势。

## 相关
- 实验过程与 run artifact：`docs/experiments/2026-07-13-mcts-ahd-orienteering-construct-3runs.md`
- 复现 ReEvo OP 设置的依据：`papers/ReEvo/appendix/03_benchmark_problems.tex`（OP 节，prize 与 max length 公式）
