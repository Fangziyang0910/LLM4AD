# TraceAAD V9.6：父代形成历史优先的锚点上下文

## 1. 研究问题

V9.5 将当前锚点的父代形成事件和直接子代尝试共同放入局部上下文。V9.6 进一步追问：当上下文容量有限时，哪些历史最能帮助模型理解当前算法，并完成下一步修改？

V9.6 的回答是：优先保留当前程序的父代形成历史，同时保留少量具有代表性的局部尝试。该版本还将正式预算改为真实 evaluator 调用次数，使不同运行按真实评价成本对齐。

## 2. 历史窗口

每个锚点最多展示 8 条事件：

1. 从当前锚点的 direct attempts 中取最近的 improve 至多 2 条、regress 至多 2 条；相同代码只保留一条；plateau 与 invalid 不进入 direct 配额。
2. 剩余位置由当前锚点父链上最靠近锚点的 formation 事件填充。
3. 所有事件按真实发生顺序呈现。

这种窗口让 direct attempts 表示“从当前程序已经试过什么”，让 formation history 表示“当前程序怎样形成”。两类事件保持同一时间顺序，模型可以同时读取局部边界和形成路径。

## 3. 历史写法

每条事件使用统一格式：

````text
[History i] Formation step | Attempt from current algorithm
Idea: <declared idea>
Change: +A/-R lines; removed: <up to 2 lines>; added: <up to 2 lines>
Result: improve | regress | plateau
Fitness: parent -> child
````

`Change` 由父子实际代码推导，展示增删行数和两侧至多两行真实改动；`Idea` 是模型生成时的声明；`Result` 与 `Fitness` 来自 evaluator。Idea 截断为 300 字符，Change 截断为 520 字符。

上下文超限时，从最早事件开始删除；任务契约和当前完整算法始终保留。没有可展示事件时明确写出无历史事件。

## 4. 预算单位

V9.6 把正式预算定义为真实 evaluator 调用次数。解析失败、no-op、重复代码和命中确定性 cache 不消耗评价预算；evaluator 实际执行一次即消耗一次。正式比较使用 1000 次真实评价。

每次完成的模型响应仍使起点锚点的机会计数 $n(a)$ 加一。评价预算与 LLM 调用数、token 数和墙钟时间分开计量。
