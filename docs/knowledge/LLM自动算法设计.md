# LLM 自动算法设计：研究认识

## 一、LLM4AD 任务定义

自动算法设计（Automated Algorithm Design, AD）是在候选算法表示空间中搜索目标算法的优化任务：评价器提供执行反馈，设计预算限制搜索规模，目标是在指定实例分布上获得良好性能。LLM4AD 研究如何用大语言模型求解这一任务。LLM 参与理解任务、提出思想、生成与修改候选；候选选择、状态管理与搜索组织由设计系统完成。

本节先说明任务对象，再给出形式化定义，随后按「问题 → 候选 → 评价 → 约束与目标 → 求解器」展开各组成部分，最后概括任务性质。

### 1.1 任务对象

给定问题实例 $x$，目标算法 $A$ 产生输出 $y$：

$$
y=A(x),\qquad A:\mathcal{X}\rightarrow\mathcal{Y}.
$$

自动算法设计涉及三个层次：

1. **问题实例**：$A$ 运行时接收的输入 $x$；
2. **目标算法**：从一类实例到输出的映射 $A$；
3. **算法设计器**：接收任务规格 $\mathcal{T}_{\mathrm{AD}}$，在候选表示空间中构造最终表示 $r^*$。

记设计器为 $\mathfrak{D}$，实现映射为 $\Gamma_K$，则

$$
\mathfrak{D}:\mathcal{T}_{\mathrm{AD}}\rightarrow r^*,
\qquad
A^*=\Gamma_K(r^*).
$$

$A^*$ 是最终得到的目标算法。LLM4AD 关注由 LLM 参与的设计器

$$
\mathfrak{D}_{\theta,\mathfrak{S}}:
\mathcal{T}_{\mathrm{AD}}\rightarrow r^*,
$$

其中 $\theta$ 为 LLM，$\mathfrak{S}$ 为搜索与状态管理机制。任务本身由问题、表示、评价、约束、预算和目标定义；LLM 是求解器的组成部分，不是任务定义的一部分。

### 1.2 形式化定义

当前研究关注「固定框架内设计函数或程序组件」。在此设定下，一个自动算法设计任务写为：

$$
\boxed{
\mathcal{T}_{\mathrm{AD}}
=
\left(
d_{\mathcal{T}},
\mathcal{X},
\mathcal{D},
\mathcal{Y},
\mathcal{R},
K,
\Gamma_K,
\mathcal{E},
\mathcal{C},
\mathcal{B},
J
\right)
}
$$

| 符号 | 含义 |
|---|---|
| $d_{\mathcal{T}}$ | 任务与算法组件的语义说明 |
| $\mathcal{X}$ | 问题实例空间 |
| $\mathcal{D}$ | 目标实例分布 |
| $\mathcal{Y}$ | 算法输出空间 |
| $\mathcal{R}$ | 候选算法表示空间 |
| $K$ | 固定算法框架 |
| $\Gamma_K$ | 将候选表示嵌入 $K$ 后得到可执行算法的映射 |
| $\mathcal{E}$ | 自动评价环境 |
| $\mathcal{C}$ | 有效性、行为、安全与资源约束 |
| $\mathcal{B}$ | 外层设计预算与内层运行预算 |
| $J$ | 目标分布上的算法性能泛函 |

可实现算法空间由表示与实现映射共同决定：

$$
\mathcal{A}_{\mathrm{impl}}
=
\left\{
\Gamma_K(r)\mid r\in\mathcal{R},\ \Gamma_K(r)\neq\perp
\right\}.
$$

以下按任务逻辑依次说明各组成部分。

### 1.3 问题实例、目标分布与输出

$\mathcal{X}$ 是算法运行时可能接收的全部输入，例如图、序列、约束条件与规模参数。实例从目标分布采样：

$$
x\sim\mathcal{D}.
$$

$\mathcal{D}$ 规定算法需要服务的对象范围，可以是单一实例、固定规模集合、已知分布，或存在分布偏移的测试环境。算法性能必须相对于 $\mathcal{D}$ 解释：实例规模、结构与分布会改变同一算法的有效性。

给定 $x$，算法输出满足

$$
y=A(x),\qquad y\in\mathcal{Y}(x).
$$

若问题带有可行性约束，合法输出还须满足

$$
y\in\mathcal{F}(x),\qquad \mathcal{F}(x)\subseteq\mathcal{Y}(x).
$$

$\mathcal{Y}(x)$ 是语法与类型允许的输出；$\mathcal{F}(x)$ 是满足问题约束的可行输出；实例级质量 $q(x,y)$ 衡量输出相对于任务目标的优劣。目标算法的输出是解、决策或数值结果；程序代码属于设计器的候选表示，不是目标算法的输出。

实际设计使用有限实例集：

| 实例集 | 用途 |
|---|---|
| $S_{\mathrm{design}}$ | 候选评价与搜索更新 |
| $S_{\mathrm{validation}}$ | 算法选择、配置选择与停止判断 |
| $S_{\mathrm{test}}$ | 设计结束后的独立评价 |

若验证集反复进入提示构造、候选修改或搜索更新，它在功能上已成为设计数据。测试集须在算法设计与选择结束前保持不可见。

### 1.4 候选表示、固定框架与实现映射

设计器直接操作的是算法表示 $r\in\mathcal{R}$，而不是最终可执行算法。$\mathcal{R}$ 由编程语言、函数签名、可调用 API、可编辑位置、代码长度、依赖与沙箱共同界定。表示可以是参数、表达式、函数体、算子、模块或完整程序；粒度决定候选结构范围、有效性条件、评价成本与可获得的算法类型。

固定框架 $K$ 提供数据读取、状态初始化、主循环、可靠子程序、可编辑组件的调用点，以及测试与评价入口。$r$ 可以是函数、决策规则、更新公式、搜索算子或若干指定代码区域。$K$ 与 $r$ 共同确定执行结构：$K$ 固定已知可靠部分并限制可编辑范围，把设计自由度集中到指定组件。框架越具体，候选空间越小、评价越稳定，可创新范围也越窄。

实现映射写为

$$
\Gamma_K:\mathcal{R}\rightarrow\mathcal{A}\cup\{\perp\}.
$$

$\Gamma_K(r)=\perp$ 表示候选无法解析、构建、编译或嵌入框架；否则得到可执行算法 $A_r=\Gamma_K(r)$。执行路径为

$$
r
\overset{\Gamma_K}{\longrightarrow}
A_r\ \text{或}\ \perp
\overset{x,\xi}{\longrightarrow}
y
\overset{q}{\longrightarrow}
\text{实例级质量}.
$$

当前核心任务是搜索单个函数或程序组件，嵌入固定框架后得到目标算法：

$$
r^*\longrightarrow A^*=\Gamma_K(r^*).
$$

算法集合设计、实例条件化设计、联合算法—参数设计属于扩展形式。

任务规格 $d_{\mathcal{T}}$ 用自然语言、注释、接口文档、公式与示例说明设计要求：问题背景、输入输出语义、组件调用位置、参数与返回值契约、约束、优化方向、可用领域知识与接口。$d_{\mathcal{T}}$ 给出规范目标；约束与评价环境将其中可执行、可测量的部分操作化。二者一致时，评价结果才代表任务意图。

### 1.5 评价环境与性能目标

评价环境将候选与评价实例映射为反馈：

$$
\mathcal{E}(r,S)\rightarrow o(r;S).
$$

评价通常依次完成解析、构建、运行、验证、度量与聚合：

$$
r
\rightarrow
\mathrm{Parse}
\rightarrow
\mathrm{Build}
\rightarrow
\mathrm{Run}
\rightarrow
\mathrm{Validate}
\rightarrow
\mathrm{Measure}
\rightarrow
\mathrm{Aggregate}.
$$

设 $y_r(x,\xi)=A_r(x;\xi)$，实例级质量为 $q(x,y_r)$，实例级测量写为

$$
m(r,x,\xi)
=
\left(
q(x,y_r),
T(r,x,\xi),
M(r,x,\xi),
v(r,x,\xi),
\ldots
\right),
$$

其中 $T$ 为运行时间，$M$ 为内存或其他资源成本，$v$ 为有效性指标。可执行性与输出合法性通常是进入性能评价的前置条件；通过后，输出质量与资源消耗决定性能。

目标分布上的理想性能为

$$
J_{\mathcal{D}}(r)
=
\mathbb{E}_{x\sim\mathcal{D},\xi}
\left[
U\bigl(m(r,x,\xi)\bigr)
\right],
$$

其中 $U$ 将实例级测量映射为任务效用。有限评价集上的经验性能为

$$
J_S(r)
=
G\left(
\left\{
m(r,x_i,\xi_j)
\right\}_{i,j}
\right).
$$

$G$ 可以是平均、加权平均、最坏情况、分位数、风险敏感目标或多目标聚合。评价器返回

$$
o(r;S)
=
\left(
J_S(r),
m_S(r),
z_S(r),
e_S(r)
\right),
$$

其中 $m_S$ 为实例级测量集合，$z_S$ 为输出、日志或行为信息，$e_S$ 为错误信息。四个层次的关系是

$$
q
\longrightarrow
m
\longrightarrow
J_S
\approx
J_{\mathcal{D}}.
$$

$J_{\mathcal{D}}$ 是希望优化的目标；$J_S$ 是评价器在有限实例上实际提供的经验目标。若规范目标与操作目标不一致，搜索会优化评价器实现的目标，从而可能出现 evaluator exploitation 或 specification gaming。

### 1.6 约束、预算与优化问题

约束集合为 $\mathcal{C}$。静态约束只依赖表示：

$$
C_j^{\mathrm{static}}(r)=1.
$$

动态约束依赖实现后的算法、实例与运行随机性：

$$
C_j^{\mathrm{dynamic}}(r,A_r,x,\xi)=1.
$$

在评价集 $S$ 上，有效候选空间为

$$
\mathcal{R}_{\mathrm{valid}}(S)
=
\left\{
r\in\mathcal{R}\ \middle|\
\begin{array}{l}
\Gamma_K(r)\neq\perp,\\
C_j^{\mathrm{static}}(r)=1,\\
C_j^{\mathrm{dynamic}}(r,A_r,x,\xi)=1,\\
\forall(x,\xi)\in S
\end{array}
\right\}.
$$

静态约束包括语言、签名、长度、可编辑区域与依赖；动态约束包括异常、超时、超内存、输出可行性与功能要求；安全约束包括文件系统、网络、进程权限与评价环境访问边界。

有效性可具有实例条件性与随机性。对仅在部分实例上成功的候选，可用有效概率

$$
P_{\mathrm{valid}}(r)
=
\Pr_{x\sim\mathcal{D},\xi}
\left[
C^{\mathrm{dynamic}}(r,A_r,x,\xi)=1
\right]
$$

并要求 $P_{\mathrm{valid}}(r)\geq1-\delta$。有效性既可作为进入评价的硬门控，也可作为成功率进入目标。

自动算法设计的核心优化问题是

$$
r^*
\in
\arg\max_{r\in\mathcal{R}}
J_{\mathcal{D}}(r),
$$

满足

$$
\Gamma_K(r)\neq\perp,
\qquad
C_j(r,A_r,x,\xi)=1,
$$

以及预算

$$
\operatorname{Cost}_{\mathrm{design}}
\leq B_{\mathrm{design}},
\qquad
\operatorname{Cost}_{\mathrm{execution}}(A_r,x)
\leq B_{\mathrm{execution}}.
$$

$B_{\mathrm{design}}$ 是发现算法的外层资源：LLM 调用、生成 token、候选数、评价次数、设计时间与计算资源。$B_{\mathrm{execution}}$ 是算法解决实例的内层资源：单实例时间、搜索节点、迭代次数、函数调用、随机重启与内存。时间与内存可同时作为硬约束与软目标，例如

$$
J_{\mathcal{D}}(r)
=
\mathbb{E}
\left[
q(x,y_r)
-\lambda_tT(r,x,\xi)
-\lambda_mM(r,x,\xi)
\right].
$$

多目标设定可用向量 $\mathbf{J}(r)$，并返回非支配候选。

### 1.7 LLM 驱动的求解过程

任务规定可设计对象、评价标准与资源边界；如何选择、生成、修改与保留候选由求解器决定。LLM 驱动的设计器为

$$
\mathfrak{D}_{\theta,\mathfrak{S}}:
\mathcal{T}_{\mathrm{AD}}\rightarrow r^*.
$$

第 $t$ 步生成候选

$$
r_t\sim\pi_{\theta}\left(r\mid c_t\right),
\qquad
c_t=\Phi_{\mathfrak{S}}(s_t),
$$

其中 $c_t$ 由任务规格与求解器允许使用的状态构成，$\Phi_{\mathfrak{S}}$ 为上下文构造。这属于求解器机制，不是任务定义的必要组成部分。

任务层面的候选执行链是

$$
r
\rightarrow
\Gamma_K(r)
\rightarrow
\mathcal{E}(r,S)
\rightarrow
o(r;S).
$$

求解器依据有限反馈继续选择、生成或修改候选，最终得到 $A^*=\Gamma_K(r^*)$。具体的候选组织、搜索方向与状态更新由方法决定。

### 1.8 任务性质

由上述定义可直接看出 LLM4AD 的优化性质：

- **黑盒**：评价器主要返回执行结果，设计器需从有限反馈推断改进方向；
- **昂贵**：单次评价常涉及多实例、多种子、编译或完整仿真；
- **有噪声**：算法随机性、实例采样与非确定性执行使测量波动；
- **有约束**：可解析、可构建、可运行且满足约束的程序只占表示空间的一部分；
- **需泛化**：设计集上的 $J_S$ 与目标分布上的 $J_{\mathcal{D}}$ 存在差距；
- **生成式结构化搜索**：候选的结构、长度与语义由模型持续生成，空间难以显式枚举。

因此，LLM4AD 是**具有生成式候选构造能力的超大结构化程序搜索任务**。在语言、API 与预算有限时，理论候选空间可以有限，但规模通常极大。
