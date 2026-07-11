# MCTS-AHD TSP reporting repeat scope

日期：2026-07-10
分类：experiment / source research

本笔记只核对 MCTS-AHD 原始论文与原始代码中关于 TSP constructive / TSP results 的重复实验和结果汇报口径。使用的 primary sources 限于 `papers/MCTS-AHD/icml2025.tex` 中相关表格段落，以及 `reference_code/MCTS-AHD-master/README.md`、`cfg/`、`source/`、`main.py`、`ahd_adapter.py` 中能说明 run/reporting 的文件。

## 结论

1. 主表 `Table \ref{tspkp}` 的 TSP constructive 结果明确是三次重复后的平均值，不是 best-of-three，也不是 mean±std。论文 caption 写明：每个 LLM-based AHD method 运行三次，并汇报 average performances；表头只给 `Obj.` 和 `Gap` 两类指标，没有 std/variance 列。证据：`papers/MCTS-AHD/icml2025.tex:280-305`。
2. 论文实验设置段也明确说每个 application scenario 对每个 LLM-based AHD method 做 three independent runs，以减少 statistical biases。证据：`papers/MCTS-AHD/icml2025.tex:320-324`。
3. 主表 TSP 数字的含义是：在 1,000-instance test sets 上的 objective 和 optimality gap；TSP optimal 由 LKH 得到。证据：`papers/MCTS-AHD/icml2025.tex:284-289`，以及 LKH 设置说明 `papers/MCTS-AHD/icml2025.tex:1288-1288`。
4. 主表没有方差或标准差。若问“有没有 std/variance”，答案是：主表没有；论文只在 appendix 的 p-value 表对部分任务列出 run1..run10、avg、std、p-value。该 appendix 不是主表口径。证据：主表列结构见 `papers/MCTS-AHD/icml2025.tex:286-300`；p-value 表见 `papers/MCTS-AHD/icml2025.tex:1060-1072`。
5. 对 TSPLib 的 TSP results，论文采用另一套口径：EoH 和 MCTS-AHD 先从 GPT-4o-mini 的三次 step-by-step constructive AHD run 中取 best-performing heuristic；然后在每个 TSPLib instance 上用不同 starting nodes 跑三次并取平均 performance。这个是 TSPLib appendix 结果，不是主表 `tspkp` 的口径。证据：`papers/MCTS-AHD/icml2025.tex:1131-1140`。
6. 原始代码 README/cfg/source 能说明单次 run 和输出 best artifact，但没有在这些文件中明说三次重复如何自动聚合，也没有看到 mean/std 汇总脚本或配置。README 只说设置 `cfg/config.yaml` 后运行 `main.py`，若要同时跑多个 evaluation 需要复制环境/问题/cfg，reported runs 另行提供；cfg 默认 `max_fe=1000`；source 在每个 run 内保存 `population_generation_*` 和 `best_population_generation_*`，并返回 best code。证据：`reference_code/MCTS-AHD-master/README.md:40-49`，`reference_code/MCTS-AHD-master/cfg/config.yaml:1-19`，`reference_code/MCTS-AHD-master/source/mcts_ahd.py:188-198`，`reference_code/MCTS-AHD-master/main.py:23-37`，`reference_code/MCTS-AHD-master/ahd_adapter.py:14-22`。

## 未明说项

- 主表没有明说每次独立 AHD run 的随机种子、run id、或三次平均的具体计算脚本。
- 主表没有明说三次平均是否附带置信区间、方差或标准差；表中也没有这些列。
- README/cfg/source 没有明说“运行三次后取平均”这一 reporting 规则；这个规则来自论文文本和表格 caption。
- 对主表而言，论文没有把结果写成 `mean±std`；若需要 std，只能参考 appendix p-value 表中部分任务的额外多次运行，而不能从主表直接读出。

## 口径速记

| 场景 | 重复口径 | 主数字 | std/variance |
|---|---|---|---|
| TSP constructive 主表 `tspkp` | LLM-based AHD methods 三次独立 run | average performance，列为 `Obj.` 和 `Gap` | 未报告 |
| Appendix p-value TSP50 | up to ten runs | run-wise values + avg | 报告 `std` |
| TSPLib appendix | 先选三次 AHD run 中 best-performing heuristic；每个 instance 再用 3 个 starting nodes 平均 | optimality gap average performance | 表中未报告 |
