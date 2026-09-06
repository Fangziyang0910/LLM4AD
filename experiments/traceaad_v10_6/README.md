# TraceAAD V10.6

实现[完整机制设计](../../docs/methods/TraceAAD-V10.6完整机制设计.md)。每次先选择父代、再选择算子，生成并评价一个子代，然后重新分配预算。一次模型调用先输出完整代码，再输出约500 words的实现摘要；提示直接要求改进、重新设计或结合算法。

| 项目 | 正式配置 |
| --- | --- |
| 规模 | 五任务 × 三重复，seed=0/1/2，每路1000次真实评价 |
| 初始化 | 从头生成8个有效根，评价计入1000次 |
| 分配 | 保留V10.5联合概率：m=.925p0+.075/N，父代选定后按条件概率选择R/P/F |
| 基础权重 | ESS目标min(N,max(.1N,2))，再除以sqrt(选择次数+1) |
| 算子 | 请求边际R/P/F=.50/.15/.35；Fuse无合法donor时执行Refine |
| 历史 | 最近最多8条真实形成事件，上限8192实际tokens |
| 摘要 | 每条进入上下文最多1024实际tokens；超长时保留完整前缀段落，原文留档 |
| 上下文 | 总32768，输出预留16384，余量256；完整输入最多16128 |
| 采样 | temperature=1、top_p=.95、top_k=20，thinking关闭 |

运行入口：

.venv/bin/python -m experiments.traceaad_v10_6.run --task tsp_construct --backend server3 --repeat 1 --seed 0 --run-name example_tsp_v106_rep1
.venv/bin/python -m experiments.traceaad_v10_6.launch --batch UNIQUE_BATCH --session-prefix v106 --watch
./experiments/traceaad_v10_6/start_monitor.sh [PORT] [--restart]
```

调度器放在持久tmux会话中。每一路实验另建会话，完成后由调度器更新`results/batch_BATCH.json`。`--dry-run`查看当时可启动的分配。同任务重复优先分散到不同后端，再按占用/容量分配；只使用空闲槽位，server1/server3/server3b/local逻辑容量为6/9/9/3。后端启动检查失败仅暂停该后端分配，下轮重试。恢复沿用原backend、run_dir和seed；每路最多启动5次，未知评价提交状态单独阻断。

实际模型服务使用现有配置：远端为Qwen3.8-27B-AWQ-INT4，本地为Qwen3.8-27B-UD-Q4_K_XL GGUF。两者量化及服务实现不同，分析需保留backend分组，不能仅凭共同别名视作完全相同。启动时进程参数保存为`results/backend_snapshot_20260906.json`。

完整候选模块直接进入统一评价器，保留辅助函数、装饰器、类和模块级语句。正常输出中的完整代码即使缺少摘要也接受；`finish_reason=length`仅在代码围栏已经闭合且通过语法、接口检查时评价，后置摘要整体记为不可用。代码未完成时不评价。每次真实评价（包括初始化、运行失败、超时）占一个预算；纯生成/解析失败单独记成本。

复用V10.5的持久化响应、评价提交及收据机制。已有响应不重复生成，已有收据不重复评价；无法确认是否执行过的评价保留unknown reservation并停止该路。V10.6 checkpoint版本106，保存机制及源文件指纹；正式运行期间保持相关源文件稳定。

每路产物：`run_config.json`保存配置；`tree_state.json`保存档案、预算和RNG；`pending_candidate.json`保存处理中操作；`llm_calls.jsonl`保存完整请求与响应；`events.jsonl`保存分配、上下文和评价结果；`evaluations.jsonl`保存评价收据；`tokenizer_calls.jsonl`保存精确计数；`logs/run_summary.json`保存终态。运行中以checkpoint及pending状态判断进度，不能将尚未生成终态summary误判为失败。

必要验证：

```bash
.venv/bin/python -m pytest -q tests/method/test_traceaad_v106.py tests/experiments/test_traceaad_v106_launch.py
.venv/bin/python -m pytest -q tests/method/test_traceaad_v105.py tests/experiments/test_traceaad_v105_launch.py tests/tools/test_openai_api.py tests/method/test_traceaad_v103_schema.py tests/method/test_traceaad_v104.py tests/experiments/test_traceaad_v104_launch.py
```

2026-09-06：V10.6针对性验证32项通过，旧版及共享接口回归55项通过；分别审查设计符合性和实现规范，后端故障隔离问题已修复并复核。真实冒烟使用`smoke_20260906_v106_*`独立目录，每任务2次真实评价、1个根，覆盖全部四个后端，不计入正式预算。正式批次的启动观察另见[启动记录](launch_20260906.md)。
