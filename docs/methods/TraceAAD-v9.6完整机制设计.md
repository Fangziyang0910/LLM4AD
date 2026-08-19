# TraceAAD V9.6：父代形成历史优先的锚点上下文

> 历史版本。V9.6 在 V9.5 基础上只改变历史上下文和预算单位；其余搜索状态、生成与分配见 [V9.5](TraceAAD-v9.5完整机制设计.md)。联合结果见 [V9.5–V9.6 机制诊断](../analysis/TraceAAD-V9.5-V9.6机制诊断.md)。

## 1. 改动目标

V9.5 的 direct-first 规则在 OP/OBP 上经常使 8 个位置被当前锚点的子代尝试占满，父代形成历史缺席。V9.6 把历史拆成两个来源：当前算法怎样形成，以及从当前算法已经试过什么；对子代尝试设上限，把剩余容量保留给 formation。

## 2. 历史选择

历史最多 8 条：

1. 从当前锚点的 direct attempts 中取最近的 improve 至多 2 条、最近的 regress 至多 2 条；完全相同代码只保留一条。plateau 与 invalid 不入选。
2. 剩余位置由 formation 链上最靠近当前锚点的事件填满。
3. 全部事件按真实发生顺序呈现。

这一定义保证 direct 不超过 4 条，并使非根锚点通常保留形成历史。它不声称 direct 的 improve/regress 选择具有独立价值。

## 3. 历史写法

每条事件统一为：

```text
[History i] Formation step | Attempt from current algorithm
Idea: <declared idea>
Change: +A/-R lines; removed: <up to 2 lines>; added: <up to 2 lines>
Result: improve | regress | plateau
Fitness: parent -> child
```

Change 由父子实际代码确定，包括增删行数和每侧至多两行真实改动；不再展示压平的长 diff。Idea 截断为 300 字符，Change 截断为 520 字符。提示结构为任务、当前完整算法、最近历史和与 V9.5 相同的生成指令。

若上下文超限，从最早事件开始逐条删除；任务契约和当前代码始终保留。没有可展示事件时明确写出无历史事件。

## 4. 预算单位

V9.6 把正式预算改为**真实 evaluator 调用次数**。解析失败、no-op、重复代码或命中确定性 cache 不消耗评价预算；evaluator 实际执行一次即消耗一次。正式比较统一为 1000 次真实评价。

这项变更使不同运行的搜索预算按真实评价对齐，但 LLM 调用数仍可能不同，不能把 1000 eval 解释为全部生成成本相同。

## 5. 解释边界

- V9.6 同时改变历史挑选、历史写法和预算单位，完整搜索结果不能识别三者的独立贡献。
- V9.5–V9.6 观察性对比显示 formation 恢复、提示缩短及生成分布变化，但锚点人群和分配尺度也同时变化。
- 随后的固定锚点三臂实验直接支持父代来时路的单步价值，没有支持已有子代尝试的稳定额外价值。V9.7 因而把默认上下文收缩为当前算法加父代来时路。
