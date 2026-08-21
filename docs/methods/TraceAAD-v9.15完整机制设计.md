# TraceAAD V9.15 完整机制设计

本文档定义 TraceAAD V9.15 的完整机制规范。V9.15 以 V9.14 为直接基线，保持单父树拓扑、生成提示接口与根初始化协议不变，重点重构 L2 预算分配机制。

---

## 1. 科学定位与核心主张

### 1.1 实验事实与经验出发点
V9.14 的实验结果揭示了显著的任务画像偏移：
1. **平滑景观与局部精炼收益（TSP / OP）**：在 TSP 任务上取得全规模历史最优（TSP50/100/200 相对 V9.7 降低 3.5%–4.4%），在 OP 任务上全面好于 V9.7/V9.12。这证实了沿高质量轨迹持续深挖（Refinement Trajectory）的有效性。
2. **复杂景观与探索过度折损（CVRP / OBP）**：在需要跨越崎岖景观、进行大幅度算法结构迁移的任务上，V9.14 的大尺度泛化显著落后于 V9.7。其直接原因在于简单的质量贪心分配使得低初始质量的探索节点（Explore Child）因初生结构折损被即时淘汰，导致搜索被困在早期发现的局部盆地中（Over-exploitation）。

### 1.2 核心科学问题
> **Research Question**: 有限评价预算下，进化搜索如何兼顾对高潜上升期轨迹的深入兑现（Exploitation / Investment），与对低初始质量但具结构突破潜力的新探索萌芽的必要保护（Protected Exploration）？

### 1.3 核心设计原则：三组件协同
V9.15 彻底消除传统启发式多项拼凑打分，将分配机制收敛为三个各具独立因果作用的核心组件：
1. **跨簇探索步阶保护（Explore Bridge with Gap Clipping）**：为初生探索节点提供有限步数、有界幅度的质量宽限，允许其完成初期局部调优，防止结构迁移入口被即时饿死；
2. **轨迹段延续价值（Trajectory Segment Continuation Value）**：结合改进稳定度与边际潜力，以来时路局部斜率识别“持续上升期”与“成熟饱和期”，将主要投资倾斜给高边际产出分支；
3. **有效样本数（ESS）全局广度控制与自适应算子调度**：由全局停滞状态驱动探索/利用呼吸节奏，替代局部人工加分项。

---

## 2. 状态表示与分配主方程

### 2.1 算子优先与自适应算子调度
两级决策流首先确定生成意图，再在特定算子条件下分配计算资源。

#### 1. 自适应算子概率
生成算子 $o_t \in \{\text{Refine}, \text{Explore}\}$ 的先验分布根据全局搜索状态自适应调节：
$$
o_t \sim \pi(o \mid \mathcal{H}_t)
$$

设当前距离上一次产生全局新 best 的停滞步数为 $n_{\text{stag}}$，探索算子概率定义为：
$$
p_E(t) = \operatorname{clip}\left(p_0 + \alpha_{\text{stag}} \cdot \frac{n_{\text{stag}}}{B}, \, p_{\min}, \, p_{\max}\right)
$$
其中默认基础探索率 $p_0 = 0.25$，停滞增益 $\alpha_{\text{stag}} = 0.50$，下限 $p_{\min} = 0.15$，上限 $p_{\max} = 0.50$。
- **前沿推进期（$n_{\text{stag}} \to 0$）**：$p_E \approx 0.15 \sim 0.25$，搜索将约 80% 的预算集中在 Refine 深挖；
- **长期停滞期（$n_{\text{stag}} \gg 0$）**：$p_E$ 自然升高至最高 0.50，主动激发跨盆地跳跃。

#### 2. 条件化父节点选择
确定算子 $o_t$ 后，在有效节点池 $\mathcal{T}_t$ 中采样父节点：
$$
a_t \sim \mu(a \mid o_t, \mathcal{H}_t)
$$

---

### 2.2 节点分配主方程
对于节点 $a \in \mathcal{T}_t$ 与算子 $o \in \{\text{Refine}, \text{Explore}\}$，分配分数定义为：

$$
S(a, o) = Q_{\text{bridge}}(a, o) + C_{\text{traj}}(a)
$$

该方程清晰区隔两类职责：$Q_{\text{bridge}}$ 负责初生探索节点的保护期存活，$C_{\text{traj}}$ 负责成长中轨迹段的追加投资。

```
                       ┌──> ① 高质量但饱和停滞节点 ──> 基础利用 (Exploit)，C_traj 归零让出预算
                       │
S(a, o) 分配主方程 ────┼──> ② 中质量但持续上升节点 ──> 重点投资 (Invest)，高 C_traj 驱动深挖
                       │
                       └──> ③ 初生低质量探索节点   ──> 受保护探索 (Protect)，Q_bridge 宽限试错
```

---

### 2.3 机制一：有界跨簇初生保护 $Q_{\text{bridge}}(a, o)$
当使用 Explore 算子发现新结构时，子节点 $a$ 常因未经局部微调而出现初始适应度下降（$q(a) < q(\text{parent}(a))$）。若直接参与全局质量排序，该节点会立即被饿死。

$Q_{\text{bridge}}$ 仅对后续的 Refine 算子生效，为其提供有上限、可衰减的保护宽限：

对于 $o = \text{Refine}$：
$$
Q_{\text{bridge}}(a, \text{Refine}) = 
\begin{cases}
q(a) + \gamma(n_R(a)) \cdot \min\Big(\max(0, q(p) - q(a)), \, \delta_t\Big), & \text{若 } a \text{ 由 Explore 产生且 } p = \text{parent}(a) \\
q(a), & \text{其它节点}
\end{cases}
$$

其中：
- $n_R(a)$ 为节点 $a$ 已经被 Refine 采样的次数；
- 衰减系数为 $\gamma(n) = \frac{1}{\sqrt{n + 1}}$；
- 最大保护跨度 $\delta_t = 2.0 \cdot s_t$（$s_t$ 为当前搜索池中有效改进幅度的经验尺度），防止严重破损的 Explore 节点占用过度预算。

当 $n_R(a) = 0$ 时，$\gamma(0) = 1$，节点获得最大宽限补偿；随着 Refine 次数增加，保护迅速衰减至其自身真实质量 $q(a)$。若数次微调后仍无改善，该节点自然沉降淘汰。

对于 $o = \text{Explore}$：
$$
Q_{\text{bridge}}(a, \text{Explore}) = q(a)
$$
节点发起跨簇探索时不享受自身保护宽限。

---

### 2.4 机制二：轨迹段延续价值 $C_{\text{traj}}(a)$
算法发现过程的边际潜力不属于孤立节点，而属于局部轨迹段。为避免将低水平区域的小步自刷误判为高潜力，延续价值由**改善稳定度**与**相对改进潜力**共同决定。

设节点 $a$ 的祖先链上最近 $k$ 步（$k = \min(6, \text{depth}(a))$）的状态序列为 $(x_0, x_1, \dots, x_k = a)$，各步的质量增益为 $\Delta q_i = q(x_i) - q(x_{i-1})$。

#### 1. 改善稳定度（Success Frequency）
$$
f_{\text{succ}}(a) = \frac{1}{k} \sum_{i=1}^{k} \mathbb{I}(\Delta q_i > 0)
$$

#### 2. 正向边际斜率（Positive Mean Yield）
$$
\overline{\Delta q}_+(a) = \frac{1}{k} \sum_{i=1}^{k} \max(0, \Delta q_i)
$$

#### 3. 头部空间与潜力调节
设当前搜索池全局最佳质量为 $q_{\text{best}}$，节点 $a$ 的头顶剩余空间为：
$$
h(a) = \max(q_{\text{best}} - q(a), \, 0)
$$

轨迹段延续价值定义为：
$$
C_{\text{traj}}(a) = f_{\text{succ}}(a) \cdot \frac{\overline{\Delta q}_+(a)}{h(a) + s_t}
$$

- **上升期轨迹（高 $C_{\text{traj}}$）**：最近连续几步稳定取得质量提升，且距离最优解仍有成长空间的分支，将获得高额投资加分，驱动快速突破；
- **低水平自刷分支（低 $C_{\text{traj}}$）**：虽然产生微小增益，但因 $h(a)$ 巨大，分母放大，无法获得夸大加分；
- **饱和停滞轨迹（$C_{\text{traj}} \to 0$）**：若处于高分（$h(a) \approx 0$）但最近连续 $k$ 步无增益（$f_{\text{succ}} = 0, \overline{\Delta q}_+ = 0$），$C_{\text{traj}}$ 归零，仅保留自身基础质量 $q(a)$，不再垄断全局预算。

---

## 3. 概率采样与有效样本数（ESS）控制

探索广度的控制完全交由全局有效样本数（Effective Sample Size, ESS）进行调控，替代在打分中加入人工 UCB/不确定性项。

### 3.1 玻尔兹曼分布与温度求解
给定算子 $o_t$，对所有有效候选 $a \in \mathcal{T}_t$，分配概率由逆温度 $\beta$ 决定：

$$
P_\beta(a \mid o_t) = \frac{\exp(\beta \cdot S(a, o_t))}{\sum_{j \in \mathcal{T}_t} \exp(\beta \cdot S(j, o_t))}
$$

有效样本数定义为：
$$
\operatorname{ESS}(\beta) = \frac{1}{\sum_{a \in \mathcal{T}_t} P_\beta(a \mid o_t)^2}
$$

### 3.2 基于树规模的动态 ESS 目标
为适应不同搜索阶段的树规模增长，目标有效样本数与当前有效节点数 $|\mathcal{T}_t|$ 成比例：

$$
\operatorname{ESS}^*(t) = \rho(t) \cdot |\mathcal{T}_t|
$$

比例系数 $\rho(t)$ 由停滞状态决定：
$$
\rho(t) = \operatorname{clip}\left(\rho_0 + \alpha_{\rho} \cdot \frac{n_{\text{stag}}}{B}, \, \rho_{\min}, \, \rho_{\max}\right)
$$
其中默认基础比例 $\rho_0 = 0.05$，增益 $\alpha_{\rho} = 0.20$，范围为 $[0.05, 0.25]$。

- **前沿推进期（$n_{\text{stag}} \to 0$）**：$\rho \approx 0.05$，求解得到高逆温度 $\beta$，采样高度聚焦于前 $5\%$ 的高潜与前沿节点；
- **停滞重探索期（$n_{\text{stag}} \gg 0$）**：$\rho \to 0.25$，分布展平，有效采样宽度拓宽至前 $25\%$，自动激活具有 $Q_{\text{bridge}}$ 保护的次优分支。

通过一维二分搜索即可在每次决策时精确求解满足 $\operatorname{ESS}(\beta) \approx \operatorname{ESS}^*$ 的 $\beta_t$。

---

## 4. 完整原子运行协议

### 4.1 初始化
1. 创建虚拟根节点（ID = 0）；
2. 独立调用模型生成 $K = 8$ 个代码互异的有效种子程序并完成评估，作为深度 1 节点挂载于根节点；
3. 初始化各节点的 $n_R(a) = 0, n_E(a) = 0$ 以及祖先轨迹段记录；
4. 初始化停滞计数器 $n_{\text{stag}} = 0$。

### 4.2 单步搜索循环（消耗 1 次真实评价预算）
在预算 $t = 1, \dots, 1000$ 内：
1. **计算自适应算子概率**：根据 $n_{\text{stag}}$ 计算 $p_E(t)$，抽取算子 $o_t \sim \operatorname{Bernoulli}(1 - p_E(t))$；
2. **计算节点分数**：遍历有效树节点 $\mathcal{T}_t$，计算 $S(a, o_t) = Q_{\text{bridge}}(a, o_t) + C_{\text{traj}}(a)$；
3. **求解逆温度**：计算目标 $\operatorname{ESS}^* = \rho(t) \cdot |\mathcal{T}_t|$，二分搜索求解 $\beta_t$，得到采样分布 $P_{\beta_t}(\cdot \mid o_t)$；
4. **选择锚点**：采样选定父节点 $a_t \sim P_{\beta_t}(\cdot \mid o_t)$；
5. **组装提示与生成**：
   - 提取节点 $a_t$ 的完整源码；
   - 提取 $a_t$ 祖先链上最近至多 8 条匹配的真实父代来时路作为上下文；
   - 组装 $o_t$ 对应的指令提示，请求 LLM 生成 `Idea + Code`；
6. **执行评价**：调用 Evaluator 评估生成程序，消耗 1 次预算；
7. **原子状态写回**：
   - 累加采样计数：若 $o_t = \text{Refine}$ 则 $n_R(a_t) \leftarrow n_R(a_t) + 1$，否则 $n_E(a_t) \leftarrow n_E(a_t) + 1$；
   - 若生成成功且通过评估，创建新子节点 $c$，标记其产生算子为 $o_t$，继承父代历史并插入树中；
   - 若产生全局新 best，重置 $n_{\text{stag}} = 0$；否则 $n_{\text{stag}} \leftarrow n_{\text{stag}} + 1$。

### 4.3 终局输出
搜索结束后，读取全局评估库中真实 Objective 最优的算法代码，输出为最终工件。

---

## 5. 机制因果链与设计承诺

1. **三组件职责正交**：
   - $Q_{\text{bridge}}$ 仅负责为跨簇 Explore 萌芽提供免饿死保护；
   - $C_{\text{traj}}$ 仅负责识别成长型轨迹段并追加投资；
   - ESS 控制仅负责全局探索/利用宽度的自适应呼吸。
2. **无冗余加分项**：彻底废弃孤立的 UCB 探索 bonus 与手工多目标加权，探索行为完全由算子调度与 ESS 温度产生。
3. **失败可见**：若出现采样分布全零或 NaN，立即中断并暴露异常，不采用静默兜底。
