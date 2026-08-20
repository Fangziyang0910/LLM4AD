# TraceAAD V9.13 代理标签盲评审计

V9.13 设计 §4.2 要求的正式 Stage P 前置审计:每任务按来源运行与代理区域分层抽取 24 个程序,审阅者在看不到 fitness、生成意图和终局位置的条件下,核对冻结的 `mechanism_tags` / `macro_family` 能否由实际代码支持。工作表由 [audit_v913_proxy.py](../../experiments/analysis/audit_v913_proxy.py) 生成(数据在 [traceaad_v913_proxy_audit/](traceaad_v913_proxy_audit/),冻结 seed 913401,剜除字段:fitness/q/intent/idea/order);每个标签附其词法触发证据,另附词表外机制关键词扫描标记候选漏标。

## 审阅结论

审阅覆盖全部 96 个样本的程序代码、触发证据与关键词标记。结论分三类;冻结规则不因审计修改。

### 系统性漏标

1. **TSP `two_opt` 拼写漏标(最大误差源)**。冻结模式 `2_opt|2opt|cost_original` 不匹配字面拼写 `two_opt` / `2-opt`。抽样 6 个含 2-opt 关键词的程序中 4 个确有真实 2-opt 局部搜索实现(`_two_opt_single_pass`、`_two_opt_iterative` 等)却未获 `two_opt` 标签,被归入 local_score 或 explicit_search。全量重放:2,382 个 TSP 程序中 599 个(25.1%)属于此类。影响:`completion_rollout` 家族低估、`local_score` / `explicit_search` 高估;TSP 的区域前沿与重访统计携带这一已知偏差。
2. **TSP `lookahead` 命名漏标**。24 个 TSP 样本中 14 个含 `lookahead` 字样但未触发 `one_step_lookahead`(规则只认 `dist_unvisited` / `forward_cost` / `connectivity` / `isolation` / `future_cost` 等具体标识符);抽样确认存在真实的 `lookahead_depth` 逻辑被 `tour_rollout` 覆盖的例子。属于"概念以通用英文命名而词表只认具体实现名"的结构性缺口。

### 弱触发(误标方向)

3. **OBP `piecewise_bands` 的 `np.where` 触发**。抽样 21 个 `piecewise_bands` 样本中 2 个仅由 `np.where` 触发(代码无 threshold 结构),语义支撑弱;发生率低。

未发现语义相反的误标(触发词存在但代码完全不含对应机制结构)。

### 不构成误差的标记

- 词表外关键词 `greedy` / `cluster` / `ratio` / `slack` / `penalty` / `margin` / `regret` 等多为构造式启发式的通用词汇,样本核对显示其机制已由对应任务的主导家族承载,不构成独立漏标。
- OBP 字面 `best_fit` / `first_fit` / `worst_fit` 未触发标签的样本默认落入 `best_fit` 家族,家族归属不受影响。

## 对解释的约束

代理是词法规则,不是语义真值。TSP 的区域级结论(前沿、重访、目的地分布)需注明 `two_opt` 漏标类约 25% 的规模;这正属于设计 §2.2 声明的"任务内静态代理"边界内,不改变 Stage P 的信息条件比较——各条件共用同一套带误差标签,条件间对比对该误差是配对不变的。
