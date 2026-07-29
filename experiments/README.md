# Experiments

实验入口按 task 和 method 组织，运行工件仍写入对应方法目录：

```text
experiments/<task>/<method>/<version>/<run_name>/
  run_config.json
  tmux_run.log
  logs/
```

`run_config.json` 是每个 run 自动生成的参数快照，不需要手写。三个重复必须拥有
独立目录，才能分别恢复、检查和测试。

这些运行目录只保存在实验机器本地，不进入 Git。Git 只跟踪本目录中的实验入口、
评估脚本和绘图脚本；完成实验后，将三次重复的聚合结果和必要曲线整理到
`docs/results/<task>/`。不要单独提交 `run_config.json`、`results.json`、日志、
checkpoint 或生成程序。

## TraceAAD

TraceAAD V4/V5 共用参数化入口，不再为每个 task 和版本复制
`run_experiment.py`。

单次运行：

```bash
uv run python -m experiments.traceaad.run \
  --task online_bin_packing \
  --version v4 \
  --backend local \
  --run-name 20260729_obp_v4_rep1 \
  --repeat 1
```

三重复 tmux 批量启动：

```bash
uv run python -m experiments.traceaad.launch \
  --task online_bin_packing \
  --version v4 \
  --backend local \
  --repeats 3
```

支持的 backend 为 `local`、`server1` 和 `zhong`。可以用 `--base-url`、
`--model`、`--no-proxy` 覆盖 profile；API key 只从环境或仓库 `.env`
读取，不作为命令行参数。

恢复单个运行：

```bash
uv run python -m experiments.traceaad.run \
  --task online_bin_packing \
  --version v4 \
  --backend local \
  --resume-from experiments/online_bin_packing/traceaad_v4/version4/<run_name>
```

其它方法目前仍使用各 task/method 目录中的原生入口。历史 timestamp 目录及其中
的配置和结果继续保留在本地，不因入口收口而改写，也不进入 Git。
