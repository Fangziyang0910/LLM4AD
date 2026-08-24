# MEMOIR

- 论文：*Memory-Guided Tree Search with Cross-Branch Knowledge Transfer for LLM Solver Synthesis*；本地来源：`../../../../papers/Memory_Guided_Tree_Search_with_Cross_Branch_Knowledge_Transfer/paper.pdf`；设计对象：完整组合优化 solver。

## 1. 核心问题与方法

MEMOIR 让每个树分支对应一种算法设计。branch-local memory 保留该路线的修改与执行故障；分支结束时 Reflect 将整条路线压缩为算法原则、失败模式和规避指令，写入 global memory，供后续新分支提案。

## 2. 论文宣称的机制贡献（逐项）

- 分离局部调试历史与可跨分支迁移的算法知识。
- 在分支终止点反思压缩，避免把低层日志污染全局上下文。
- 失败节点本身是约束修复所需的有效记忆。

## 3. 实验究竟支持了什么

|机制主张|论文证据|证据等级|判断|
|---|---|---|---|
|完整系统提高有效率和归一化成绩|Table 1、Figures 2–3|间接支持|七任务、匹配 16 次执行预算的联合结果。|
|global memory 有益|§4.3、Table 2 w/o Global Memory|直接支持|Avg 降 6.81 点、Valid 降 5.32 点。|
|branch-local memory 有益且更关键|Table 2 w/o Branch-Local Memory|直接支持|该消融下降最大，尤其约束密集任务。|
|失败记录有益|Table 2 w/o Failed Nodes|直接支持|只保留有效记录也下降，支持保留失败事实。|
|层级分离优于 flat memory|§4.2 flat-memory check|直接支持|同模型设置下 flat variant 的 Avg/Valid 分别下降 8.5/4.4 点。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

局部记录回答“当前实现下一步怎么修”，全局摘要回答“下一条路线不要再犯什么、可尝试什么”。二者混在一起会让新路线被旧实现细节锚定。失败记录提供了可行域边界，而不仅是负分。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 这是与 TraceAAD 最直接的对照：保留完整可重放 lineage，再在路线终止时生成可验证的跨路线知识。最小验证应比较 local-only、global-only、flat、hierarchical 四组，并固定上下文 token。

## 6. 证据边界

每题独立运行，尚未证明跨问题迁移；global entry 是 LLM 摘要，可能失真。branch budget 分配规则是启发式而非学习得到；更强 critic/reflect 模型也影响记忆质量。
