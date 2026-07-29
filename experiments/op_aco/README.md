# OP-ACO 实验

与 ReEvo / HSEvo / PathWise / CALM 的 Orienteering ACO 协议对齐。权威配置见 `docs/实验配置.md`。

## 数据与 ACO

| 项目 | 配置 |
|---|---|
| 训练 | OP50 × **5**，seed **1234** |
| 验证 | OP50/100/200 × 64，seed **3456** |
| 测试 | OP50/100/200 × 64，seed **4567** |
| 路程预算 | OP50/100/200 → maxlen **3 / 4 / 5** |
| Prize | Kool 式（相对 depot 距离离散化后归一化） |
| ACO | **20 ants × 50 iterations**，`aco_seed=1234` |
| 指标 | 平均 collected prize，**越高越好** |

## 入口

```bash
# MCTS-AHD
uv run python experiments/op_aco/mcts_ahd/run_experiment.py

# PathWise
uv run python experiments/op_aco/pathwise/run_experiment.py

# TraceAAD
uv run python -m experiments.traceaad.run --task op_aco --version v4 --backend local

# 测试评估（默认 test_50/100/200）
uv run python experiments/op_aco/evaluate_best_on_test.py <run_dirs...> \
  --output-dir experiments/op_aco/<method>/eval_best_<tag>
```

TraceAAD 入口默认写入 `version4/`。实验覆盖与完成状态统一维护在
`docs/实验覆盖.md`。
