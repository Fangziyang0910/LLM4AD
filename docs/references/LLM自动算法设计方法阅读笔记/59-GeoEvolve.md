# GeoEvolve

- 论文：*GeoEvolve: Automating Geospatial Model Discovery via Multi-Agent Large Language Models*；本地来源：`../../../../papers/GeoEvolve_Automating_Geospatial_Model_Discovery/paper.pdf`；设计对象：Kriging 与 geospatial conformal prediction 算法。

## 1. 核心问题与方法

GeoEvolve 以内层 OpenEvolve 生成/变异代码，外层 controller 评估并保留 global elite；四个组件为 code evolver（OpenEvolve）、evolved code analyzer（诊断缺失知识并提出查询，§3.2）、geospatial knowledge retriever（GeoKnowRAG，从 141 份地理文档检索：五类关键词、Wikipedia/arXiv/GitHub 三源、300 词 50 重叠分块、text-embedding-3-small + Chroma、RRF 重排）、geo-informed prompt generator（RAG-Fusion 形成下一轮领域提示）。

## 2. 论文宣称的机制贡献（逐项）

- 双层 evolution+agent loop 将领域诊断接到动态检索。
- GeoKnowRAG 注入空间异质性、邻域和不确定性等理论先验。
- 自动化、可扩展的地学建模流水线。

## 3. 实验究竟支持了什么

|机制主张|论文证据|证据等级|判断|
|---|---|---|---|
|完整系统改善 Kriging/GeoCP|Tables 1–2、Figures 4–5（相对原始 kriging RMSE 降 15.4%/21.2%/13.0% Cu/Pb/Zn；GeoCP interval score 55.37→46.12 即 −16.7%）|间接支持|两任务的联合系统结果；headline 对 OpenEvolve（52.37）的优势含 10 倍迭代差——OpenEvolve 系基线只跑 10 次迭代，GeoEvolve 系跑 10 外层 × 10 内层 = 100 次（§4 开头）。|
|动态结构化检索优于移除检索|§4、Tables 1–2，GeoEvolve w/o GeoKnowRAG|直接支持|相同进化预算（同为 100 次迭代）下完整方法总体更好——"identical budgets" 只适用于这一消融对。|
|仅静态加知识 prompt 足够|Tables 1–2，OpenEvolve+GeoKnowledge|反向或混合证据|静态知识有时改善、有时恶化（GeoCP 54.80 差于 OpenEvolve 52.37；kriging 上 Cu/Pb 变差、Zn 略好）。|
|code analyzer 的主动查询单独必要|§3.2–3.4|未验证|未分别关闭 analyzer、RAG-Fusion 或 outer-loop elite 控制。||

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

静态知识与当前失败无关时会稀释上下文；动态查询把知识注入绑定到 elite 的具体缺口，因此更可能成为有效变异方向。知识库本身仍由专家关键词决定，自动性位于检索与利用而非知识边界构建。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：检索应由当前路线的失败证据触发。最小验证：static-RAG、state-conditioned RAG、no-RAG，并审计检索块是否落实到代码 diff。
