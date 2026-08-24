# AutoSND

- 论文：*AutoSND: From Execution Evidence to Structural Policies for Automated Network Dismantling Heuristic Discovery*；本地来源：`../../../../papers/AutoSND_Execution_Evidence_to_Structural_Policies/paper.pdf`；设计对象：完整网络拆解程序。

## 1. 核心问题与方法

AutoSND 分三阶段：Stage I 从简单启发式广搜并记录质量、运行时、状态和代码操作；Stage II 用 LLM-Struct 对齐执行结果与程序结构，编译成“复用/避免/限制”的 structural policy；Stage III 以该策略约束树搜索，并分别产生质量优先与速度优先程序。

## 2. 论文宣称的机制贡献（逐项）

- 把 scalar fitness 升级成可执行的结构级搜索约束。
- 用证据诱导局部信号、邻域访问和状态更新范围。
- 通过多目标 seed branches 同时搜索质量与可扩展性。

## 3. 实验究竟支持了什么

|机制主张|论文证据|证据等级|判断|
|---|---|---|---|
|完整系统在 12+3 网络上有竞争力|Table 1、Figure 3、Figure 5|间接支持|完整三阶段与专用网络启发式/AHD 的联合比较。|
|Stage II 策略改善跨图覆盖和质量|§5.3.1、Figure 4、Appendix Table 10|直接支持|w/o Stage II 使用对应完整搜索的固定候选；Full 覆盖 12/12，消融为 8/12。|
|structural family restriction 保持有效性|Figure 4|直接支持|移除限制后 Stage III validity 降至 76.5%。|
|最终结构机制可解释|Table 2、Figure 6、§5.4|部分支持|代码组件消融和 lineage 显示 residual degree backbone、局部修正与有界更新的作用；仅限最终候选与网络域。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

执行证据指出“什么结构在何种图上失败”，policy 将其压缩成后续代码的边界条件。相比自由文本反思，结构字段更易检查；但压缩仍由 LLM 完成，可能把相关性写成规则。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：把路线历史编译为带适用范围的结构政策，并保留来源候选。最小验证：policy-on/off 与无来源摘要对照，检查每条规则是否能被回放证据支持。

## 6. 证据边界

任务只覆盖 network dismantling，且论文标注 KDD 2027 稿件；Stage II 有 10 次 induction calls，额外预算需要计入。搜索期使用 proxy graphs 和 relaxed normalization，最终才做 strict full-permutation 检查。
