# CALM (w/o GRPO) 机制说明

本仓库当前阶段对照方法记为 **CALM (w/o GRPO)**：只保留搜索框架（verbal 算子、池管理、collapse、numeric_refine、接受/拒绝），关闭权重更新。实现位于 [`llm4ad/method/calm/`](../../llm4ad/method/calm/)，入口 [`experiments/runners/calm/`](../../experiments/runners/calm/)。权威参考代码：`../reference_code/CALM/`（上游 [whxru/CALM](https://github.com/whxru/CALM)）。

## 主张边界

- 当前阶段：与冻结 LLM 的 TraceAAD / EoH / ReEvo / MCTS-AHD / PathWise 同档比较搜索机制。
- 进入微调后：再对照完整 **CALM (w/ GRPO)**；禁止用未微调方法对打 GRPO 版作为搜索优劣主证据。
- 正式表与日志必须写 `CALM (w/o GRPO)`，不得写成完整 CALM。

## 算子与调度

| 调度名 | Prompt 标签 | 父代 |
| --- | --- | --- |
| simplification | `simplify` | 1 |
| injection | `injection` | 1 |
| replacement | `replacement_ins` / `_hyp` / `_crd` | 1 |
| crossover | `crossover` | 2 |
| creation | `create`（兜底） | 0 |
| revisit | 复用历史 prompt | 同原 |
| numeric_refine | 非 LLM，AST 扰动数值 | 1 |

`ub_*` 为多项式采样权重；`n_prompts` 次采样得到本轮实际次数。人口未满时偏 injection；`<2` 个启发式时禁 crossover。父代按 rank 倒数概率；crossover 以 0.5/0.5 做 performance / diversity。

## 检测修复（相对上游）

上游 `Prompt.is_injection` / `is_simplification` 仍匹配**旧**文案，与当前 prompt 不一致，导致这两类算子被标成 `initialization`、reward 固定为 0。

本仓库按算子意图修正检测，匹配当前 prompt：

- injection：`Inject one novel, meaningful component into the following algorithm while preserving its main data flow`
- simplification：`Please create a locally refined version of the following algorithm`

其余 prompt 字符串（含拼写 `sepcific`、Local Refinement Guidance、七条 design strategy、replacement 三模式）与参考实现一致。

## 适配层

| 原文 | 本仓库 |
| --- | --- |
| OpenAI / Unsloth | `LLM.draw_sample` |
| Ray 问题环境 | 各 task `*Evaluation` + `SecureEvaluator` |
| `evaluation_budget` | `max_sample_nums`（正式对照 1000） |
| `configs/*/template.py` | 任务 `template_program` |
| `configs/*/seed.py` | `llm4ad/method/calm/seeds/`（签名已对齐 LLM4AD） |

每任务 `population_size`、`ub_*`、`revisit_*`、numeric/profile 超参来自 CALM `configs/<task>/local.yaml`。采样侧同时使用该文件中的 `generation_temperature` 与 `generation_top_p`。

## 分数方向

LLM4AD 评测统一越高越好（距离/箱数类为负值）。池内比较直接使用该分数，不改变 Evaluation 语义。

## 已知适配缺口：performance profile

原版 CALM 在评测时保留**逐实例** `perfs` 向量，用于：

- `archive_profile_tolerance` 去重
- profile reward gate / bonus
- numeric refine 的 parent profile 去重

本仓库 `SecureEvaluator` / task Evaluation 当前只返回聚合标量 score。移植中将 `perfs = [score]`，使上述机制在**标量语义**下仍可运行，但与原文多维 profile 行为不等价：yaml 里按多实例向量调的 tolerance / gate 会退化为对均值的比较。若要对齐原文 profile 行为，需要 Evaluation 暴露逐实例分数后再接入。
