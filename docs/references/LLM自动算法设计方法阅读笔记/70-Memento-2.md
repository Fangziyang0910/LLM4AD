# Memento 2

- 论文：*Memento 2: Learning by Stateful Reflective Memory*；本地来源：LaTeX 源码目录 [`Memento_2_Learning_by_Stateful_Reflective_Memory/`](../../../../papers/Memento_2_Learning_by_Stateful_Reflective_Memory/)；研究对象：冻结参数、靠情景记忆 + 反思持续学习的 LLM 智能体的 RL 理论（纯理论，无新实验）。

## 1. 核心问题与方法

为"记忆驱动的免训练自改进"建立收敛理论。设定 SRDP 元组 $\langle\mathcal S,\mathcal A,\mathcal P,\mathcal R,\gamma,\mathfrak M,p_{LLM}\rangle$：每步两段动作——检索 $c_t\sim\mu(\cdot\mid s_t,M_t)$ 再行动 $a_t\sim p_{LLM}(\cdot\mid s_t,c_t)$，复合策略 $\pi^\mu(a\mid s,M)=\sum_c\mu(c\mid s,M)p_{LLM}(a\mid s,c)$。**Reflected MDP**：增广状态 $x=(s,M)$ 恢复马尔可夫性，动作空间即记忆 $\mathcal C(M)=M$，LLM 被吸收进环境（转移 $\mathcal P_{LLM}$、奖励 $\mathcal R_{LLM}(x,c)=\sum_a p_{LLM}(a\mid s,c)\mathcal R(s,a)$），唯一可控决策是检索策略 $\mu$。Read=策略改进、Write=策略评价：Parzen 窗先验 $\mu_0$ + 空案例 $c_\varnothing$（混合系数 $\lambda(x)$ 实现"检索 vs 由 LLM 内部知识发现"），KL 正则软策略迭代有闭式 $\mu^+(c\mid x)\propto\mu_0(c\mid x)e^{Q(x,c)/\alpha}$。

## 2. 论文宣称的机制贡献（逐项）

- 定理 1：固定记忆下软策略迭代收敛到 KL 正则最优。
- 定理 2：双时间尺度（$\rho_t/\eta_t\to 0$、Robbins-Monro、鞅差噪声、记忆紧吸引集）下 $(Q_t,\mu_t,M_t)$ 联合收敛。
- 定理 3 与推论：值差 $\|V^{\pi^\star}-V^{\pi_M}\|_\infty\le\frac{2R_{\max}}{(1-\gamma)^2}\Delta_M$，且 $\mathrm{TV}(\pi_M,\pi^\star)\le\varepsilon_{LLM}(r_M)+\delta_M$——记忆覆盖半径 $r_M\to 0$、检索误差 $\delta_M\to 0$ 则渐近最优。核心假设"LLM 局部一致性"：在 $d(s,s(c))\le r$ 内 $\mathrm{TV}(p_{LLM}(\cdot\mid s,c),\pi^\star(\cdot\mid s))\le\varepsilon_{LLM}(r)$，即 LLM 对参考案例的胜任半径。

## 3. 实验究竟支持了什么

|主张|论文证据|证据等级|判断|
|---|---|---|---|
|收敛与渐近最优保证|定理 1–3 及证明|直接支持（理论）|假设清单明确（双时间尺度、局部一致性、紧吸引集），均为渐近性、无样本复杂度。|
|记忆系统实践有效|引用 Memento（zhou2025）、CBR-LLM（数据科学/软件测试）、Agent K 的既有实验|间接支持|本文无新实验；实证全部来自同作者序列前置工作。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

值差被分解为"生成器局部胜任误差 $\varepsilon_{LLM}$ + 记忆/轨迹覆盖误差 $\delta_M$"——记忆的价值是缩小生成器需要胜任的邻域。这与"生成决定可达、分配决定兑现"的耦合主张同构：检索策略 $\mu$ 即分配（从哪份历史继续），$p_{LLM}$ 即生成。双时间尺度（快策略迭代 / 慢记忆写入）为"评价预算在生成与记忆巩固间的分配"提供稳定性论证框架。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：用"$\varepsilon_{LLM}(r)$ + 覆盖误差"的分解语言表述生成-分配耦合。前提：承认局部一致性假设未经验证、$c_\varnothing$ 的混合系数需手调。风险：理论是渐近的，不给有限预算结论。
- 可学习点：记忆写入与读取用不同时间尺度更新（写慢读快）。最小验证：在线搜索中对比每次写 vs 批量巩固的稳定性。

## 6. 证据边界

纯理论：无新实验、无样本复杂度（文中批注自认"开发较平凡"）；记忆增长的计算挑战与嵌入质量影响未处理；RAG 被归为静态特例、CBR-LLM 为无状态特例的定位是否公允未检验。
