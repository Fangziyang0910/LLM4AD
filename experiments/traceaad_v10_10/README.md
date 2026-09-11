# TraceAAD V10.10

从 V10.9（1091）复制后移除新结构试用和直接子代试错上下文，保留质量分配与形成历史，加入宽容解析、具体错误记录和最多一次修复。2026-09-11 修订：初始化收缩为 sequential informed initialization，删除 Init 阶段 AST 去重与提示中的防御性约束；正常搜索上下文改为统一短形成路径；解析协议改为代码优先——程序单独决定候选能否评价，说明按 tagged/prose/missing 三种来源记录且不再设标签、位置或字数门槛，VRPTW 非法构造携带具体条件、修复反馈附真实时限与候选代码定位，契约写明运行环境版本与整次评价时限（解析策略 `code_first_description_extracted_v1`）。设计见 [V10.10 机制设计](../../docs/methods/TraceAAD-V10.10-机制设计.md)。正式批次 `20260910_v1010_formal`（2026-09-10 启动）运行的是修订前机制；历次修订改变机制与检查点指纹，需另起新批次。

解析修订的离线重放（零 LLM 调用）：`python experiments/traceaad_v10_10/analysis/replay_parser.py`，对 15 路全部已持久化响应重放新旧解析并输出分类、修复链对账与文件快照，报告写入本地 `results/analysis/`。2026-09-11 重放：旧 7,772 次解析失败中 7,676 次被新解析接受（代码通过语法/接口/编译校验），30 次暴露真实代码错误，66 次仍不可提取；新旧均接受的 14,641 次代码文本零差异，零回归。

默认五任务 × 三重复，seed 0/1/2，每路 1000 次真实评价、8 个有效根、32K 总上下文、16K 输出上限。修复后实际调用评价器也计入这 1000 次，LLM 修复调用在 `llm_calls.jsonl` 中标记 `stage=repair`，保留 usage、耗时与 `repair_of`。解析失败与搜索阶段父代/donor 重复过滤不占评价次数；初始化不做重复过滤。

输入不再固定限制为 16128 tokens。普通生成和错误修复都使用 `min(16384, 32768 - 256 - 实际输入tokens)` 作为本次输出上限；完整输入不做裁剪。服务的总窗口仍需容纳输入和实际输出。

当前配方：Refine、Tune、Fuse、Pivot 各 25%。父代选择在全档案上使用 ESS-8 Boltzmann 质量分布，Pivot 与均匀抽样 1:1 混合，selection counts 只记录不参与概率。四个算子统一使用「当前程序完整代码 + 成绩 + 最近八条短形成路径」，每步历史为该步生成算法的 Idea、算子与前后 fitness；Fuse 额外加入本轮 donor 的完整代码与成绩，donor 无可用计算时允许直接改善宿主。初始化为 sequential informed initialization，按生成顺序展示全部已有根的完整代码与实测 fitness，不做重复规避；donor 只抽一次。历史不展开祖先代码、diff 或历史 donor，一次组装后检查总容量。删除在线 AST 修改分类，原始代码可供离线分析。

先核验，再创建独立冻结副本：

```bash
.venv/bin/python -m pytest -q tests/method/test_traceaad_v1010.py tests/method/test_traceaad_v109.py tests/task/test_vrptw_failure_reasons.py tests/experiments/test_traceaad_v1010_launch.py tests/experiments/test_traceaad_v1010_monitor.py tests/experiments/test_traceaad_v1010_replay.py
.venv/bin/python -m experiments.traceaad_v10_10.freeze --batch v1010_formal --session-prefix v1010
```

进入返回的 runtime，使用返回的 `launch_command` 启动。调度继承 V10.9 的空槽分配、冻结身份检查及断点恢复；新 results 路径和会话前缀与 V10.9 隔离。不要在运行中的冻结副本修改代码。

单路入口可通过以下命令查看：

```bash
.venv/bin/python -m experiments.traceaad_v10_10.run --help
```

2026-09-10 核验：V10.10、V10.9、V10.8 方法及 V10.10/V10.9 调度、监控与 OP 任务测试共 115 项通过；单路命令行入口可用。覆盖候选错误修复与系统错误终止、错误消息清理和限长、新算子比例落日志、Tune/Pivot 无历史、最终总容量检查、无独立历史配额、一次解析、新节点无额外优待、直接子代不影响普通生成上下文、解析恢复、真实评价计数、修复网络错误恢复及六处中断恢复。未调用真实模型进行搜索，修复成功率与最终求解收益尚未验证。

共享监控注册为 `v10_10`；正式启动沿用已核验的现有模型服务。

```bash
.venv/bin/python -m experiments.traceaad_v10_6.monitor \
  --version v10_10 --session-prefix v1010 --port 8765
```

监控地址为 `http://127.0.0.1:8765/?version=v10_10`。页面默认展示 V10.10，也可在顶部切换到 V10.9 等历史版本。

遇错后，普通失败候选先按原规则落日志；候选代码提取或校验失败，以及导入、运行、超时或非法结果才触发下一候选的一次修复——响应缺标签、说明为空或说明过长不触发修复。修复保留同一父代、donor 和原操作标签，增加 `repair_of`，不再次消耗父节点的独立设计请求计数。修复失败或重复即结束本次修复链。反馈只含清除路径后最多 2000 字符的核心异常类型和消息，runtime/exec 错误附候选代码最深栈帧行号与函数名，超时附真实时限数字；不附调用栈或父代整份代码；不按本地提示长度跳过修复。评价准备故障或框架异常记录后终止。没有剩余评价预算时不发修复请求，未知评价仍阻断。

## 2026-09-10 正式启动

上下文修复后的续跑副本为 `results/runtime_20260910_v1010_formal_contextfix`。原冻结副本完整保留；迁移前检查点保存在 `results/before_contextfix`，逐路评价进度及迁移记录见 `results/contextfix_migration.json`。此次只修改输出额度分配，历史上下文材料不变，沿用原批次和评价收据续跑。

五份被原 16128 输入阈值拦截的提示实测如下。历史/其他材料分别按原始文本计数，合计与聊天输入之间相差模板开销。

| 实验 | 聊天输入 tokens | 形成历史 tokens | 其他材料 tokens |
| --- | ---: | ---: | ---: |
| OP rep1 | 16850 | 14957 | 1881 |
| VRPTW rep1 | 16478 | 14210 | 2256 |
| CVRP rep2 | 18511 | 15467 | 3032 |
| VRPTW rep2 | 19794 | 16773 | 3009 |
| VRPTW rep3 | 16223 | 13842 | 2369 |

V10.8 的 8192 配额只约束历史材料，且历史表示在 diff 和完整前代中择短。V10.10 取消配额并固定展示完整源代码、diff 和历史 donor，因此历史本身增至约 14–17K；长度增长与错误反馈无关。

正式批次为 `20260910_v1010_formal`，五任务 × 三重复共 15 路。源码冻结在 `results/runtime_20260910_v1010_formal`，调度会话为 `v1010_launcher`，逐路会话使用 `v1010_<task>_r<repeat>`。启动前停止 V10.8 TSP allocation 消融的运行会话和自动补位器，保留其检查点；V10.9 `20260909_v109_initaware` 未完成实验继续运行。V10.10 调度器只占用现有等价服务的空闲槽。

启动验收时 V10.10 为 15 路运行，五个任务的三次重复均已完成真实模型请求和至少一次评价，共记录 58 次模型调用、41 次真实评价。同期 V10.9 为 5 路完成、10 路运行；TSP allocation 消融没有残留运行会话或补位进程。这是启动状态记录，不是结果分析。
