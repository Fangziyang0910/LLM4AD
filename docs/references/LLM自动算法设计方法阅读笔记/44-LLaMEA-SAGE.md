# LLaMEA-SAGE

- 论文：*LLaMEA-SAGE: Guiding Automated Algorithm Design with Structural Feedback from Explainable AI*；本地来源：`../../../../papers/LLaMEA_SAGE_Guiding_Automated_Algorithm_Design_with_Structural_Feedback/paper.pdf`；设计对象：完整黑盒优化算法代码。

## 1. 核心问题与方法

方法从 archive 中每个程序提取 AST/控制流等结构特征，以特征和真实 fitness 训练 surrogate，再用 SHAP 解释被选父代的哪些结构特征应增减，把解释翻成自然语言注入 LLaMEA 的变异提示。

## 2. 论文宣称的机制贡献（逐项）

- archive 不只存候选，也训练结构—性能 surrogate。
- SHAP 将相关结构信号转成 LLM 可执行的定向变异建议。
- 结构反馈提高早期样本效率且保持模板自由的代码搜索。

## 3. 实验究竟支持了什么

|机制主张|论文证据|证据等级|判断|
|---|---|---|---|
|结构反馈加快早期改进|§4–5、Figures 2–3、7–8|直接支持|LLaMEA-SAGE 与 vanilla LLaMEA 使用相同设置，差别是结构引导；末期优势并非所有设置都显著。|
|LLM 会按结构建议实际修改代码|Figure 5、Figures 6/9|部分支持|合规性分层："refine" prompt 下大体服从，"random new" prompt 下 LLM 完全忽略结构引导——探索型指令会冲掉方向引导；且方向几乎总是"increase"（benchmark 特异）。|
|机制跨 LLM 后端稳健|Appendix B、Figure 12|部分支持|主实验与消融后端为 GPT-5-mini；附录跨后端检查中论文自身不一致——正文措辞为 Gemini-2.0-flash-lite，Figure 12 图注却标 gpt-5-nano 与 gemini-flash-2.0-lite。仍限 MA-BBOB、5 runs。|
|surrogate 或 SHAP 各自必要|§5、Appendix B|未验证|没有分别关闭 surrogate 学习或替换 SHAP 的匹配消融。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

surrogate 把历史样本变成局部方向信号，SHAP 再将不可直接用于代码编辑的数值归因翻译成自然语言。它学习的是 archive 内相关性；若搜索分布移动，建议可能强化“复杂度越高越好”等偶然模式。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：从真实 lineage 提取结构变化与增益，训练辅助方向器。前提：按任务/阶段校准且保留执行验证。最小验证：同父代比较无反馈、随机反馈、学习反馈三组。

## 6. 证据边界

主文以 GPT-5-mini、SBOX-COST 与 MA-BBOB 为主；结构特征不含动态行为与交互效应。末期最好值、token 成本和统计显著性并非在所有设置一致占优。
