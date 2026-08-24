# Back to the Beginning of Heuristic Design

- 论文：*Back to the Beginning of Heuristic Design: Bridging Code and Knowledge with LLMs*；本地来源：`../../../../papers/Back_to_the_Beginning_of_Heuristic_Design_Bridging_Code_and_Knowledge_with_LLMs/paper.pdf`；设计对象：可解释知识状态及其实例化的启发式代码。

## 1. 核心问题与方法

论文把 code-first 的 bottom-up AHD 与 knowledge-first 的 top-down AHD 区分开：后者先在知识空间进化原则/假设，再生成代码验证；作者分别改造 ReEvo 与 MCTS-AHD，并提出同时维护代码与知识群体的 dual 版本。

## 2. 论文宣称的机制贡献（逐项）

- 将知识提升为主要搜索对象，使其可跨程序、轨迹和 solver 复用。
- 用 distortion–compression 视角解释知识抽象的收益与损失。
- top-down、bottom-up 与 dual 搜索可互补，稀疏执行下仍能演化知识。

## 3. 实验究竟支持了什么

|机制主张|论文证据|证据等级|判断|
|---|---|---|---|
|knowledge-first 能改善发现效率|Figure 2、Table 1|部分支持|在 ReEvo/MCTS-AHD 的 top-down 对照中表现改善，但提示与搜索对象一并改变。|
|知识产物可跨 solver/分布迁移|Tables 1–2|部分支持|固定知识被注入新 solver/NCO 分布，直接检验所测迁移；不能证明一般跨任务迁移。|
|dual code–knowledge 搜索互补|Figure 3、Table 3|部分支持|dual 在多项设置较好，但同时增加两类群体及交互，预算归因需谨慎。|
|稀疏执行仍可利用知识搜索|Figure 3、Table 3|部分支持|训练进度变弱而测试泛化可改善，呈混合而非单向证据。|
|理论界给出实际收益保证|§3 distortion–compression analysis|未验证|理论揭示条件与权衡，没有实证测量实际 distortion 或信息压缩量。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

知识状态压缩多份实现，降低路线之间重复探索；但抽象会丢掉实现细节。只有同一知识类中的程序质量相近时，这种压缩才保真。dual 的价值在于让代码纠正抽象、抽象跨代码迁移。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：历史不只存摘要，而存可再次实例化并由执行反证的设计假设。最小验证：同 token/评价预算比较代码历史、知识历史、双表示。

## 6. 证据边界

论文覆盖 CO、SCO、SR、PE，但不同任务实例化和 evaluator 不同；知识文本由 LLM 生成，真实性没有自动验证。部分 top-down 优势体现在泛化而非训练最好值，不能概括为全程支配。
