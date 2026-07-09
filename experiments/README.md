# Experiments Layout

Experiments are organized as:

```text
experiments/<task>/<method>/
  run_experiment.py
  <timestamp>/
    run_config.json
    tmux_run.log
    logs/
```

Each task/method pair has a small script with parameters written directly in Python. For example:

```bash
uv run python experiments/tsp_construct/traceaad/run_experiment.py
```

For long runs, launch from the nested `LLM4AD/` repo with tmux:

```bash
TS=$(date +%Y%m%d_%H%M%S)
tmux new -d -s traceaad_tsp_construct_$TS \
  "cd /home/fang/code/LLM4AD/LLM4AD && \
   NO_PROXY=222.201.145.8,localhost,127.0.0.1,::1 \
   uv run python experiments/tsp_construct/traceaad/run_experiment.py"
```

Migrated timestamp folders keep their historical launch settings in `run_config.json`. Future launches should use the `run_experiment.py` under the matching task/method directory.
