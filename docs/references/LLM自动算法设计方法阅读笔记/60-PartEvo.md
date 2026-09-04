# PartEvo

- 论文：*Partition to Evolve: Niching-enhanced Evolution with LLMs for Automated Algorithm Discovery*；本地来源：`../../../../papers/PartEvo_Partition_to_Evolve/paper.pdf`；正式发表于 NeurIPS 2025；设计对象：完整 meta-heuristic 算法。

## 1. 核心问题与方法

PartEvo 将代码相似度向量（CodeBLEU 两两相似度矩阵的行）或思想嵌入投影到特征空间，以 K-means 构造 language-space niches；跨 niche 均匀分配采样资源，niche 内精英保留的概率选择。四个算子（RE 反思演化、SE 摘要演化、CN 跨 niche 交叉、LGE 局部-全局引导）**在每个 niche 内都执行**：CN 随机选 k=2 个其它 niche 的代表与当前个体以当前者为框架融合，LGE 组装"个体 + 本 niche 最优 + 全局最优"三元组进 prompt——两者引用跨 niche 信息但运行位置在 niche 内。

## 2. 论文宣称的机制贡献（逐项）

- feature-assisted partition 让传统 niching 可用于无显式几何的语言搜索空间。
- niche collaboration 同时支持局部 exploitation 与跨区 exploration。
- prompt-centric 与 EC-inspired operators 互补。

## 3. 实验究竟支持了什么

|机制主张|论文证据|证据等级|判断|
|---|---|---|---|
|完整系统提高搜索效率|Tables 1–2、Figure 2|间接支持|四 benchmark、多个 runs 的联合结果。|
|niche 数影响探索—利用|§4.4、Table 4|直接支持|K=1/2/4/6 同设置比较，K=4 在所测任务较稳。|
|有意义 feature 优于随机 partition|Table 5、Appendix Figure 4|直接支持|Code Similarity/Thought Embedding 比 random 更稳、更快。|
|prompt-centric operators 有益|Tables 6/11，w/o RE、SE、PartEvo†|直接支持|逐项和成组移除均退化。|
|EC-inspired CN/LGE 有益|Tables 6/11，w/o CN、LGE、PartEvo‡|直接支持|逐项和成组移除均退化。|
|任意 clustering method 都等价|Appendix Table 14|部分支持|K-means/GMM/spectral 都优于 EoH，但任务少且差异仍存在。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

niche 的核心不是“代码看起来不同”，而是把有限 LLM 查询分给低相关路线。feature 若与有效行为无关，partition 只是随机切预算；BehaveSim 的后续工作正好提供更贴近行为的替代表示。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：按路线/行为构造 niche，再测每个 niche 的独立突破贡献。最小验证：random、code、thought、behavior 四种 partition，共享初始化与预算。
