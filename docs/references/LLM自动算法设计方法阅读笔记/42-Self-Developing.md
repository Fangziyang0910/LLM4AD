# Self-Developing

- 论文：*Can Large Language Models Invent Algorithms to Improve Themselves?*；本地来源：`../../../../papers/Can_Large_Language_Models_Invent_Algorithms_to_Improve_Themselves/paper.pdf`；正式发表于 NAACL 2025 Long Paper；设计对象：可执行的模型合并算法与生成这些算法的 algorithm factory。

## 1. 核心问题与方法

方法让一个 LLM 生成模型合并代码，将候选算法应用于 seed model 并以数学推理成绩评价，再把高低分算法组成偏好对，用 DPO 更新 algorithm factory；下一轮继续生成算法，从而同时改进模型和算法生成器。

## 2. 论文宣称的机制贡献（逐项）

- 用可执行程序表示开放式模型改进算法，而非从固定合并规则中选参。
- 通过迭代 DPO 把算法评价反馈写回生成模型，形成“发现—训练—再发现”闭环。
- 发现的合并规则可迁移到生成时未使用的模型。

## 3. 实验究竟支持了什么

|机制主张|论文证据|证据等级|判断|
|---|---|---|---|
|完整系统能发现优于人工合并基线的规则|§3.2、Table 1|间接支持|联合包含生成、筛选、DPO、温度衰减与合并 evaluator。|
|迭代后候选质量提高|Table 2、Figure 3|部分支持|跨轮次最佳值和分布改善，但没有匹配预算的 DPO-off 对照，不能排除累积采样效应。|
|发现规则可迁移|§4 Transferability of Algorithms、Figure 6（GSM8k）、Figure 7（MATH）、Table 3、Appendix A Table 4（固定规则在域外 Mistral 系模型重测：迁移 78.8% vs 为新模型重优化的 Task Arithmetic 71.4）|部分支持|固定规则在域外模型上重测，直接支持所测迁移；任务仍限数学推理与同类 Mistral 模型。|
|温度衰减改善迭代生成|Appendix B、Figure 8|部分支持|比较了温度设置下的可执行性和多样性，但未隔离其对最终合并成绩的贡献。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

算法代码比单个权重向量更可复用，DPO 则把 evaluator 的排序压缩进下一轮生成分布。论文实现中生成的算法**恒定作用于固定的 seed model $M_0$**（跨轮评价环境不变，t≥2 仅把历史 top-3 算法加入偏好数据），因此跨轮改进在静态目标上成立；若改为把合并模型回灌为新 seed，评价环境会随轮漂移——这是该范式外推时的风险，非本文实现。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：将成功与失败算法组成可训练偏好对。前提：评价方向和任务分布固定。最小验证：同候选、同调用预算比较 DPO-on/off。
- 可学习点：训练后必须固定生成器或产物做 held-out 迁移。风险：把模板记忆当成算法能力。
