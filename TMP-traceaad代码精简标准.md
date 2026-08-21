# TraceAAD 代码精简标准（临时工作文档）

V9.7 → V9.14 的精简（1828 → 766 行）中，与机制无关的部分提炼为下述标准，应用到其余全部 traceaad 版本（v4、v5、v8、v9、v9_7、v9_7_co、v9_8–v9_13）。约束：**不改变各版本机制核心逻辑**。v9_14 不动，作为标准参照。

## 一、精简项

### 1. 观测输出裁剪（artifacts.py）

- 删 `best_curve.csv`（可从 evaluations.csv 推导；monitor.py 对缺失已优雅降级，v9_14 先例）。
- 删 `logs/errors.jsonl` 与 `record_error`（异常已由 summary.json 的 error 字段与控制台承载）。
- 删 `record_llm_call`、`record_decision` 空桩方法及调用点（含 route_selected / anchor_selected / best_updated / history_built 等全量打分表记录）。
- 控制台打印压到单行：每次评价一行 `[Eval N/1000] 结果 | Best: x`；结束摘要保留可压简。删分 stage 的花哨格式化函数。
- `best_program.py` 文件头压成一行 fitness 注释。
- `write_summary` 删 started_at / duration_seconds；payload 照旧透传。
- **evaluations.csv 列 schema 保持不变**（monitor.py 依赖 eval_count / child_fitness / status / best_fitness；一次性分析脚本可能依赖其余列）。

### 2. 健壮性基础设施删除

- LLM 传输重试（`TRANSPORT_RETRIES` 循环 + 逐次失败记录 + 中途 checkpoint）→ 单次 `draw_sample`，异常上抛。
- checkpoint 原子写（tempfile + `os.replace`）→ 直接 `write_text`。
- checkpoint 恢复时的 `search_configuration` 一致性校验与 config 块 → 删；checkpoint 只存恢复必需状态。
- 防御性构造检查（`use_numba_accelerate` / `use_protected_div` / `random_seed` 约束、`debug_mode` 全链路传参、模板函数 deepcopy）→ 删。
- 各版本自查同类模式：参数创建时校验一次以内、想象中的兜底默认值、空桩接口。

### 3. 上下文窗口适配删除（v9_7 系 7 版）

- `context_limit` 参数、token 计数探测（`_fits` / `_tokens`）、"prompt 超限就逐条丢最老历史事件重建" 的循环 → 固定取最近 `max_history` 条。
- 若 run.py / launcher 向该版本传 `context_limit`，同步删除传参处。

### 4. 由"存"改"算"（仅在行为不变时）

- `_best_id` 增量维护 + tie-break 理由记录 → 按需现算，**tie-break key 保持原样**（如 `(q, -length, -order)`）。
- 仅用于 CSV 列的字段可在写行时现算，不必先存成实体字段。

## 二、机制红线（不得动）

- 分配 / 选择公式与层级（route→anchor、各版本 priors / windows / half-lives / regions / Treatment 等）。
- 各版本导出常量与 `__init__` 导出面（run.py 按版本导入大量常量与类名）。
- bootstrap / 初始化协议；intent 混合与 `draw_intent` 的种子策略。
- prompt 全文与历史渲染内容（含 diff 摘录——v9_7 系历史条件含代码增删摘录，属生成条件）。
- 去重 / 缓存策略（v9_7 系五种 kind 的判定与跳过评价行为）。
- best 的 tie-break 规则。
- evaluations.csv 列 schema。
- 构造函数 kwargs 与 run.py / launcher 实际传参保持一致（`context_limit` 例外，两侧同步删）。
- Pending / Attempt / Forest 等实体中参与 prompt 条件、checkpoint 恢复或 CSV 输出的字段。

## 三、兼容面核对清单（每个版本动手前先 grep）

1. `experiments/runners/traceaad/run.py` — import 符号、构造 kwargs。
2. `experiments/runners/traceaad/launch_v9XX*.py`、`v913_stage_p.py`、各 probe 脚本。
3. `tests/` 下对应测试文件（断言了被删工程行为的，随行为删除同步改测试；机制断言不动）。
4. `experiments/analysis/analyze_v9XX*.py` — 是否引用将被删除的方法或列。
5. `experiments/plotting/monitor.py` — evaluations.csv 四列。

## 四、验证

- 每版本改完：`uv run pytest tests/ -q -k "<该版本测试标记>"` 全绿 + `uv run python -c "import ..."` 编译检查。
- 基线：改前全量 traceaad 测试已通过。
- 报告格式：行数（改前 → 改后）、删了什么、保留了什么机制红线、测试结果。

## 五、执行结果（2026-08-21 完成，v4–v9.13 共十一版）

| 版本 | 改前 | 改后 | 缩减 |
| --- | --- | --- | --- |
| v4 | 2075 | 1703 | −372 |
| v5 | 2666 | 2306 | −360 |
| v8 | 2664 | 1765 | −899 |
| v9 | 2803 | 1908 | −895 |
| v9_7 | 1828 | 1363 | −465 |
| v9_7_co | 1819 | 1280 | −539 |
| v9_8 | 2258 | 1890 | −368 |
| v9_9 | 2035 | 1739 | −296 |
| v9_10 | 2285 | 1977 | −308 |
| v9_11 | 2098 | 1658 | −440 |
| v9_12 | 2209 | 1747 | −462 |
| v9_13 | 2416 | 1959 | −457 |
| **合计** | **27156** | **21295** | **−5861（−21.6%）** |

（口径：`llm4ad/method/traceaad_*/` 内 Python 行数；v9_14 为参照未动，766 行。）

测试：全量测试套件 675 passed；traceaad 相关从基线 440 收敛到 425（净删 15 条只断言已删工程行为的测试：transport 重试、llm_calls 持久恢复、checkpoint config 校验、drop_oldest 等；机制断言全部保留）。

### 版本特定取舍

- v8/v9 的 token 适配循环（压参考历史→减直接分支）是 V9-Core 规范第 3 节明文机制，保留未删；v9_7 系的 context_limit 适配是工程设施，全删。
- v9_8–v9_13 的 events.jsonl（选择/体制/frontier 决策事件）是各版规范列明的机制工件且有分析脚本消费，保留；只蒸馏其中的全量打分表。
- v8/v9 的 candidates/edges/decisions/llm_calls jsonl 蒸馏到实际消费列（逐脚本核对：analyze_v92_scoring、analyze_v9_v95_generation_interface、analyze_tree_depth、analyze_process、monitor 等）。
- v4/v5 的 best 是机制状态（参与种群管理 / tie-break 断言），未改现算。
- `global_best_update_reason`（v8 系 edge 字段）因进 checkpoint round-trip 且被分析脚本用于重建 ImprovementEdge，保留。

### 已知后果

- 精简后代码不能恢复精简前的 checkpoint（config 块 / best_id 等键已删；旧 checkpoint 多余键可被新 loader 忽略，反向不行）。run.py 的 resume 校验按 expected 键投影旧 run_config，旧 run_config 里多出的 context_limit 不影响恢复。
- v9_7 系不再写 best_curve.csv、errors.jsonl、llm_calls.jsonl；monitor.py 对缺 best_curve 优雅降级（first_ts 为空，速率改用 mtime）。原始模型响应不再落盘（checkpoint 的 Pending.response 仍保留单条待处理响应）。
- v9_8 的 `_v98_implementation_identity` 哈希源码，未来 run 的 protocol_source_sha256 会变，属预期。
- 若将来分析全新 v9_12 运行，analyze_v912_progress.py 的 best_curve 对比段需改从 evaluations.csv 推导（历史批次工件冻结，不受影响）。
