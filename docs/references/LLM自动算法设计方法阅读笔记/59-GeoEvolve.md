# GeoEvolve

- 论文：*GeoEvolve: Automating Geospatial Model Discovery via Multi-Agent Large Language Models*；本地来源：`../../../../papers/GeoEvolve_Automating_Geospatial_Model_Discovery/paper.pdf`；设计对象：Kriging 与 geospatial conformal prediction 算法。

## 1. 核心问题与方法

GeoEvolve 以内层 OpenEvolve 生成/变异代码，外层 controller 评估 global elite；code analyzer 诊断缺失知识并提出查询，GeoKnowRAG 从 141 份地理文档检索，RAG-Fusion 形成下一轮领域提示。Code-to-Formula agent 负责把用户代码转成标准搜索接口。

## 2. 论文宣称的机制贡献（逐项）

- 双层 evolution+agent loop 将领域诊断接到动态检索。
- GeoKnowRAG 注入空间异质性、邻域和不确定性等理论先验。
- 自动接口转换降低领域模型进入代码进化的门槛。

## 3. 实验究竟支持了什么

|机制主张|论文证据|证据等级|判断|
|---|---|---|---|
|完整系统改善 Kriging/GeoCP|Tables 1–2、Figures 4–5|间接支持|两任务的联合系统结果。|
|动态结构化检索优于移除检索|§4、Tables 1–2，GeoEvolve w/o GeoKnowRAG|直接支持|相同进化预算下完整方法总体更好。|
|仅静态加知识 prompt 足够|Tables 1–2，OpenEvolve+GeoKnowledge|反向或混合证据|静态知识有时改善、有时恶化；GeoCP interval score 比 OpenEvolve 更差。|
|code analyzer 的主动查询单独必要|§3.3–3.4|未验证|未分别关闭 analyzer、RAG-Fusion 或 outer-loop elite 控制。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

静态知识与当前失败无关时会稀释上下文；动态查询把知识注入绑定到 elite 的具体缺口，因此更可能成为有效变异方向。知识库本身仍由专家关键词决定，自动性位于检索与利用而非知识边界构建。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：检索应由当前路线的失败证据触发。最小验证：static-RAG、state-conditioned RAG、no-RAG，并审计检索块是否落实到代码 diff。

## 6. 证据边界

只覆盖两个 geospatial tasks；knowledge base 141 文档、关键词人工选定。论文也以 LLM 辅助分析算法差异，定性机制解释不能替代代码组件消融。

## 7. 论文内定位

Figures 1–3；§3.1–3.4；Tables 1–2；Figures 4–5；§5；Appendix A.2–A.4、Figures 6–7。
