# PathWise 在 OP 上的三次重复实验

## 设置

- 方法：PathWise
- 任务：`orienteering_construct`，训练集，OP50，16 个实例，seed=2024
- 模型：`qwen3.6-27b-awq`，vLLM endpoint `http://222.201.145.8:8080/v1`
- 搜索预算：`max_sample_nums=500`
- PathWise 参数：`pop_size=6`、`num_actions=2`、`num_rollouts=2`、`max_inner_steps=3`、`num_evaluators=4`
- 三个 run 通过 tmux 独立启动，启动间隔约 5 秒，并设置 `NO_PROXY=222.201.145.8,localhost,127.0.0.1,::1`

## Run

- `pathwise_op_rep1`：`experiments/orienteering_construct/pathwise/20260714_105543_rep1/`
- `pathwise_op_rep2`：`experiments/orienteering_construct/pathwise/20260714_105543_rep2/`
- `pathwise_op_rep3`：`experiments/orienteering_construct/pathwise/20260714_105543_rep3/`

## 启动检查

2026-07-14 10:59 左右，三个 tmux pane 均为 `dead=0`，已生成 `run_config.json`、`tmux_run.log` 和 PathWise 日志。三个 run 分别推进到 22、22、21 个 sample，当前 best score 分别为 14.103125、14.09375、14.103125；未发现 traceback 或 HTTP 4xx/5xx。实验仍在运行，最终结果待 500 个 sample 完成后再评估测试集并写入 `docs/results/`。
