# LLM4AD — 自动算法设计研究平台

本仓库基于 [Optima-CityU/LLM4AD](https://github.com/Optima-CityU/llm4ad)，研究大语言模型驱动的自动算法设计。主线方法为 **TraceAAD**，在 TSP、CVRP、OP、在线装箱和 VRPTW 等任务上与其他自动算法设计方法比较。

- 科研协作方式：[AGENTS.md](AGENTS.md)
- 研究问题、近期方法与论文材料：[科研文档导航](docs/README.md)
- 运行入口与实验配置：[实验导航](experiments/README.md)

## 目录

```text
llm4ad/             方法、任务、评价器与 LLM 客户端
experiments/        各方法的运行入口及本地结果，共享工具在 runners/
docs/               研究认识、方法设计、实验分析、阅读笔记与报告
tests/              方法、任务和工具的测试
```

相关论文在 `../papers/`，原始参考代码在 `../reference_code/`，默认只读。原始实验工件保留在实验机器本地，不进入 Git。

## 开发与运行

环境依赖由 `pyproject.toml` 管理。已有环境使用 `.venv/bin/python`；运行具体版本前查阅对应实验入口。例如：

```bash
.venv/bin/python -m experiments.traceaad_v10_5.run --help
```

任务数据配置在 `llm4ad/task/optimization/generated_data_config.py`。方法通过 evaluation 提供的任务描述与程序模板构造提示词。新任务或对照方法的设置参考论文与对应实验协议。

修改后运行相关测试；需要全量回归时使用：

```bash
.venv/bin/python -m pytest tests/ -q
```

## 致谢

基于 [Optima-CityU/LLM4AD](https://github.com/Optima-CityU/llm4ad)（BSD License）。
