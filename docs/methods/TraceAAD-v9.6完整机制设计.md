# TraceAAD V9.6 完整机制设计

> 版本：V9.6（`llm4ad/method/traceaad_v9_6/`），protocol id
> `traceaad-v9.6-anchor-history-context`。
> V9.6 在 V9.5 基础上只改两处：**锚点历史上下文**（挑法 + 写法，设计定案见
> [RQ-009](../research/RQ-009-锚点历史上下文.md)）和**预算单位**（按真实评价次数计）。
> 状态与事实模型、初始化、锚点选择、输出契约、去重与 cache、checkpoint 结构均与
> V9.5 相同，规范见 [TraceAAD-v9.5完整机制设计.md](TraceAAD-v9.5完整机制设计.md)
> 第 2、4、5、6 章与附录，本文不重复。

## 1. 历史上下文

选定锚点后，从它的父代轨迹（formation）和已有子代尝试（direct attempts）中挑最多
8 条历史，让模型看清三件事：我现在有什么算法 → 它之前怎么改过来的 → 从这里已经
试过什么。

### 1.1 挑法

实现在 `history.py` 的 `select_history`：

1. **子代尝试**：从当前锚点的 direct attempts 中，取最近的改进尝试至多 2 条、最近的
   退步尝试至多 2 条；完全相同的代码（按 `evaluator_input_hash`）只算一次。持平和
   无效不入选。没有子代就不取。
2. **父代轨迹**：剩余名额（8 减去子代条数）用 formation 链上最靠近锚点的修改填满，
   不足 8 条就全给，不凑数。
3. 全部入选事件按发生顺序（`candidate_order`）排列。

子代封顶 + 成功失败都露面，修正了 V9.5 子代优先、装满为止在 OP/OBP 上造成的
失败比例失真（实测数据见 RQ-009）。

### 1.2 写法

实现在 `history.py` 的 `render_history`。历史块只含入选事件，不放总览统计。每条事件
统一格式，父代标 `Formation step`，子代标 `Attempt from current algorithm`：

```text
[History 3] Attempt from current algorithm
Idea: Use adaptive candidate ordering
Change: +8/-5 lines; removed: `score = dist * w`; added: `score = dist * w * decay`
Result: regress
Fitness: 12.84 -> 13.20
```

- **Idea**：当时声明的想法，单行截断 300 字符；
- **Change**：确定性生成的紧凑修改描述——加删行数 + 每侧至多 2 行真实改动代码示例，
  整行截断 520 字符（第二轮识别实验验证过的形式），不再放整段压平的 diff；
- **Result / Fitness**：improve / regress / plateau 与 parent -> child 分数。

没有可展示事件（根锚点首次 bootstrap，或尝试全部为持平/无效）时写一句
`No history events are shown for this algorithm.`。

### 1.3 Prompt 结构

```text
[Task]
[Current Algorithm]  Fitness + 完整代码
[Recent Algorithm Improvement History]  最多 8 条
[Instruction]  与 V9.5 一字不动（含输出契约）
```

### 1.4 Context 回退

若 prompt 超出上下文预算，从最早的事件开始逐条丢弃并重新渲染；全部丢完仍超限则抛
`ConfigurationFailure`。V9.5 的 diff 截断阶梯
（1200/600/300/0）随 raw diff 一起删除。

### 1.5 审计

每次构建历史记录一条 `history_built` decision：formation/direct 完整池、入选 id、
实际展示 id、因上下文丢弃的条数、prompt tokens。分析历史失真时以池与入选的对比为准。

## 2. 预算单位

预算按**真实评价次数**计：`_has_budget()` 比较 `evaluation_count`（evaluator 实际
执行次数）与 `evaluator_call_budget`。解析失败、重复代码命中 cache 都不消耗预算。
固定 1000 次评价对所有任务与重复统一。配置键为 `evaluator_call_budget` +
`budget_unit: real_evaluator_call`，停止原因为 `evaluator_budget_exhausted`。

## 3. 实现与验证

- 包结构：`schema / source / forest / selection / history / prompt / checkpoint /
  traceaad`。`history.py` 替换 V9.5 的 `evidence.py`；`forest / selection / source /
  checkpoint` 与 V9.5 相同。
- 运行入口：`experiments.runners.traceaad.run --version v9_6`，默认 8 根、
  32768 context、8192 输出 tokens，`--budget` 即评价次数预算。
- 测试：`tests/test_traceaad_v96_integration.py` 覆盖挑法（封顶、去重、时间排序、
  formation 填充）、写法（事件格式、无历史）、context 回退、runner 冻结配置
  与预算口径。
