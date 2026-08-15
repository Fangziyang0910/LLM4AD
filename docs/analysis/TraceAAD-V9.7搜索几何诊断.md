# TraceAAD V9.7 搜索几何诊断：Allocation 与 Proposal

> 分析对象：正统 V9.7 批次 `20260814_150927` 的四任务三重复，共 12 次完整运行、96 个初始根和 12,152 次正式生成决策；历史条件分析复用第三轮固定锚点三臂实验的 72 个锚点。
> 时点：2026-08-15。本文是面向 V9.8 之前的机制尸检，只回答 V9.7 实际形成了怎样的搜索几何，不提出新的在线分数。
> 复现入口：[`analyze_v97_search_geometry.py`](../../experiments/analysis/analyze_v97_search_geometry.py)；机器可读结果见[`summary.json`](traceaad_v97_search_geometry/summary.json)。

## 核心判断

V9.7 的真实运行形态已经可以比“路线几乎塌缩、Explore 改得更大”描述得更精确。

1. **路线层表现为带吸收态的历史最好门控，并未持续估计多路线的延续价值。** 由于 $q^*(r)$ 只升不降，而欠投入路线的乐观项有固定上界，一条路线一旦落入不可达区就不能靠等待重新获得预算。CVRP 与 OP 几乎从第一次决策起只剩一条可存活路线；TSP 最迟在第 40 次决策进入同一状态。
2. **拓扑路线不保持稳定算法簇。** 每次运行的 8 个根只有 2–3 个静态机制宏簇，但最终最好程序有 9/12 次已经离开其根程序的宏簇。路线保存来源历史，却可以在内部多次换簇；root lineage 分配因而没有等价为 algorithm family 分配。
3. **Explore 同时承担稀疏结构跃迁。** 在静态机制标签下，Explore 的宏簇切换率为 37%–54%，Refine 为 2%–12%；Explore 还贡献了各任务 20%–36% 的搜索期全局增益。它的单次命中率低，但正尾更重。
4. **当前分配几何会系统性截断 Explore。** 新锚点出生时若满足 $q_{mathrm{child}}+s<q^*(r)$，即使保有最大锚点乐观项也永远无法再被选择。Explore child 的出生即淘汰率为 36%–87%，明显高于 Refine；跨宏簇 child 的相应比例为 34%–85%。
5. **最终谱系显示的是“稀疏换簇，随后开发”的分工。** 9 次根簇到终局簇发生变化的运行中，6 次由 Explore 首次进入终局宏簇、3 次由 Refine 完成；但 12 个最终最好程序本身全部由 Refine 生成。固定 0.7/0.3 是否合适仍未被识别。

因此，V9.7 最值得保留的是一条更清楚的机制事实：proposal 已经能在同一路线内部制造算法结构跃迁，allocation 仍按拓扑来源和即时质量筛选这些跃迁。V9.8 面对的主要错位是**投资单位、延续价值与跨簇提议的存活条件没有对齐**。

## 1. 分析对象与测量边界

### 1.1 两种几何

Allocation geometry 描述当前状态集合怎样获得后续预算。V9.7 在路线层使用

$$
S_t^{\mathrm{route}}(r)=q^*_t(r)+\frac{s}{\sqrt{N_t(r)+1}},
$$

在已选路线内使用

$$
S_t^{\mathrm{anchor}}(a)=q(a)+\frac{s}{\sqrt{n_t(a)+1}}.
$$

本文直接测量该分数在实际质量差、计数和候选集合上产生的可达区域、淘汰边界与预算集中。

Proposal geometry 描述在给定锚点、历史和意图后，下一程序落到哪里。除 immediate $\Delta q$、invalid 和修改行数外，本文增加三个过程量：

- **全局突破**：候选 $q$ 严格超过该运行此前全部程序；
- **静态机制宏簇切换**：父子可执行代码的任务机制标签映射到不同宏簇；
- **观察到的后代价值**：当前 child 在真实分配器下是否继续长出后代，以及其后代是否越过原父代。

### 1.2 算法簇只使用可审计代理

代码差异不能直接定义算法簇。本文先删除可执行代码中的注释、docstring 和其他字符串，再按实际标识符与运算结构提取任务机制标签，文本 diff 不参与 family switch 判定：

| 任务 | 静态机制宏簇示例 |
| --- | --- |
| TSP | local score、one-step lookahead、completion rollout、explicit search |
| CVRP | distance/capacity、angular/radial、spatial cluster、savings、sweep partition |
| OP | prize-density、return feasibility、neighborhood potential、target direction |
| OBP | best fit、fragmentation、future-gap model、distribution model |

宏簇切换要求父子主机制类别改变；仅增加一个局部标签不计。12 条最终谱系进入或重返终局宏簇的事件另行逐条核对了实际代码标签、Idea 与 diff。该标签仍然只是**静态、任务内、解释型代理**，没有达到执行行为簇或人工双盲语义标注的构念强度。本文据此写“静态宏簇切换”，不把它升级成真实算法盆地的已验证标签。

### 1.3 因果边界

- 路线访问由 V9.7 自身策略选择。被放弃路线的后续结果缺失不是随机缺失，而是策略导致的右删失；不能从现有日志恢复它们若继续投入会达到的 ceiling。
- Refine 与 Explore 在完整搜索中面对的是分配器选出的不同锚点，后代也继续受同一分配器筛选。意图统计描述联合系统行为，不是固定锚点意图消融。
- 静态宏簇、全局突破与后代统计都是搜索过程证据，不替代 held-out。正式最终质量仍以[实验总汇](../experiments/实验总汇.md)为准。

## 2. Allocation geometry

### 2.1 路线分数存在精确的不可恢复区

对一条当前未被选择的路线 $r$，只要它不获得生成机会，$q^*_t(r)$、$N_t(r)$ 和路线分数就全部保持不变。当前全局路线前沿记为

$$
Q_t=\max_j q^*_t(j).
$$

若某一时刻

$$
q^*_t(r)+\frac{s}{\sqrt{N_t(r)+1}}<Q_t,
$$

那么领先路线的分数始终至少为 $Q_t$，路线 $r$ 当时无法入选；不入选又使它无法更新质量或计数。随着其他路线的历史最好质量只升不降，这条路线以后也无法自行返回候选前沿。这是现行确定性规则的吸收态，不是事后对曲线形状的命名。

初始化后每条路线都完成一次 bootstrap，因此 $N(r)=1$。一条不再访问的路线此后可保留的最大补偿只有

$$
\frac{s}{\sqrt{2}}\approx 0.707s.
$$

若初始化结束时它与领先路线的质量差已经超过这个上界，就从第一次正式决策起永久退出。这里的 $s$ 又来自 8 次 Refine bootstrap 的一步绝对变化中位数，并不估计根之间的质量差或路线未来潜力。

### 2.2 实际塌缩很早进入吸收态

下表中的“可存活路线”按上述精确不等式逐决策重放；“单路线吸收”是可存活集合第一次只剩 1 条的正式生成决策序号。“稳定集中”是最终主路线从此以后再未失选的序号。决策数与真实评价数分开，因为 invalid、重复和缓存可能不消耗评价。

| 任务 | 第一次决策可存活路线，rep1/2/3 | 单路线吸收，rep1/2/3 | 稳定集中，rep1/2/3 | 顶路线份额 | bootstrap 后零次正式访问 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TSP | 5 / 4 / 3 | 5 / 40 / 7 | 4 / 37 / 1 | 99.4%–100% | 19/24 |
| CVRP | 1 / 2 / 1 | 1 / 2 / 1 | 1 / 1 / 1 | 100% | 21/24 |
| OP | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 100% | 21/24 |
| OBP | 7 / 8 / 8 | 28 / 176 / 147 | 24 / 168 / 87 | 85.9%–98.0% | 3/24 |

这给路线干预率只有 1.07%、且全部发生在前 1/3 一个更强的解释：TSP、CVRP 与 OP 只在最初 1–40 次正式生成内短暂比较路线，随后多数路线进入不可恢复区，系统长期运行路线内锚点搜索。锚点乐观项改变 64.8% 的选择，与这一运行形态一致。

OBP 是唯一较晚进入吸收态的任务，但原因首先是评价几何。三次运行在第一次决策分别有 6、6、7 条路线并列最高质量，初始可存活路线为 7、8、8；其他任务每次都只有一个初始最高值。OBP 的多路线访问主要由离散分数平局和较小质量跨度维持，不能据此认领更强的路线语义异质性。

### 2.3 根之间有浅层差异，路线身份却不保持算法簇

每次运行的 8 份根代码按协议都不同，但静态机制结构远没有形成 8 个独立算法簇。

| 任务 | 初始最高质量并列数，rep1/2/3 | 根宏簇数 | 根机制签名数 | 最大根宏簇份额 | 初始质量跨度，以 $s$ 为单位 | 根簇到终局簇改变 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TSP | 1 / 1 / 1 | 2 / 2 / 2 | 4 / 3 / 3 | 75% / 87.5% / 87.5% | 3.76 / 6.06 / 6.17 | 2/3 |
| CVRP | 1 / 1 / 1 | 3 / 2 / 3 | 4 / 3 / 4 | 50% / 75% / 62.5% | 7.24 / 5.48 / 5.96 | 3/3 |
| OP | 1 / 1 / 1 | 2 / 2 / 2 | 4 / 3 / 4 | 62.5% / 75% / 75% | 12.44 / 19.33 / 16.46 | 1/3 |
| OBP | 6 / 6 / 7 | 2 / 2 / 2 | 7 / 6 / 5 | 50% / 50% / 62.5% | 1.97 / 0.54 / 0.54 | 3/3 |

这里同时出现两件事：

- 根具有有限异质性。特别是 CVRP，初始代码覆盖 capacity-distance、angular-radial 等不同静态宏簇；“根完全没有差异”不能作为统一解释。
- 路线不保持簇身份。12 次运行中有 9 次，最终最好程序的宏簇已不同于所在路线的根宏簇；CVRP 与 OBP 是 3/3。算法簇转换主要发生在路线内部。

因此，现行 route 的科学含义只能是**共同初始来源的形成历史**。它可以作为 provenance unit，却还不是经验证的 investment unit。若 V9.8 想按算法簇分配，不能继续默认一个 root lineage 等于一个 basin。

### 2.4 被放弃路线的潜力在现有工件中不可识别

96 条路线中有 64 条在 bootstrap 后一次正式生成都没有得到；仅看 TSP、CVRP 与 OP，比例是 61/72。它们的“最终最好质量”实际上只是 root 与一次 Refine bootstrap 的最大值，不是路线 ceiling。

现有日志因此不能回答“被放弃路线后来是否可能更好”。任何把这些路线的观测终点与主路线终点比较的分析都会把**投资不足**误写成**潜力不足**。同理，下列早期量目前只能解释选择，不能验证延续价值：

- 初始化 $q^*$ 直接进入在线分数，当然会预测谁先得到预算；
- bootstrap $\Delta q$ 同时决定该路线的初始质量和共享尺度，存在机械耦合；
- 事后最终谱系、最终宏簇和最终突破数只有在获得大量预算后才能观测，不能作为当时可用的前瞻信号。

真正需要估计的量更接近有限后续预算下的新突破价值。例如，对额外 $H$ 份路线预算，可定义

$$
V_H(r,t)=\mathbb E\left[
\max_{1\le h\le H}\left(q^*_{t+h}(r)-q^*_t(r)\right)_+
\mid \mathcal H_t,\ r\text{ receives compute}
\right].
$$

V9.7 没有估计 $V_H$，现有选择性日志也没有足够 overlap 离线验证它。继续把 recent gain、stagnation 或 family label 塞入信用之前，必须先通过强制覆盖取得各路线的可比较续段。

### 2.5 Allocation 五个问题的当前答案

| 问题 | 当前答案 | 仍缺的证据 |
| --- | --- | --- |
| 路线是否异质 | 根有 2–3 个静态宏簇和 3–7 个机制签名，但差异远少于 8 条独立算法簇 | 行为簇、跨种子复现和强制等预算后的 ceiling |
| 何时塌缩 | CVRP/OP 在第 1–2 次、TSP 在第 5–40 次决策进入单路线吸收态；OBP 为 28–176 | 无；这是现行分数与日志的直接事实 |
| 放弃路线是否可能更好 | 不可识别；64/96 条路线没有 bootstrap 后续段 | uniform/random 或强制前缀覆盖 |
| 哪些早期量预测 continuation | 当前只能证明 $q^*$ 预测选择，不能证明它预测边际计算价值 | 具有 overlap 的路线续段和跨运行验证 |
| 任务异质性是否对应 allocation opportunity | OBP 的多路线行为主要对应质量平局；其他任务的质量跨度远大于共享尺度 | 固定 proposal 的 Single、Uniform/Random、Adaptive 对照 |

## 3. Proposal geometry

### 3.1 Refine 与 Explore 的差异已经超出修改行数

下表只统计正式搜索。improve 是相对父代改善；全局突破要求超过运行此前全部程序。宏簇切换率的分母是形成了新锚点的候选。全局增益份额按每次严格刷新 best-so-far 的增量求和，任务间不合并原始 $q$ 尺度。

| 任务 | improve，Refine / Explore | 静态宏簇切换，Refine / Explore | 全局突破，Refine 次数/尝试；Explore 次数/尝试 | Explore 的全局增益份额 |
| --- | ---: | ---: | ---: | ---: |
| TSP | 14.6% / 4.2% | 7.4% / 37.2% | 43/2,082；10/915 | 36.5% |
| CVRP | 36.7% / 2.2% | 2.3% / 48.9% | 79/2,124；8/928 | 22.1% |
| OP | 6.3% / 1.3% | 12.0% / 53.9% | 27/2,142；5/930 | 20.2% |
| OBP | 29.1% / 6.3% | 5.3% / 36.9% | 27/2,109；11/922 | 21.9% |

Explore 在四任务上都更少产生普通 improve，也更少产生每次尝试意义上的全局突破；同时它的宏簇切换率稳定高于 Refine，且相差约 4–22 倍。

Explore 的正尾也更重。四任务中，Explore 全局突破的中位增量均大于 Refine：TSP 为 0.074 对 0.014，CVRP 为 0.157 对 0.041，OP 为 0.158 对 0.052，OBP 为 0.75 对 0.50。不同任务单位不能横向比较；任务内结果将 Explore 描述为**低命中、较大步长**的提议分布。

这些结果支持“Explore 在完整搜索里经常改变静态机制结构”，但还不能支持“给定相同锚点时 Explore 的换簇概率因指令而提高”。后一个因果问题仍需要固定锚点、相同来时路和配对采样的意图实验。

### 3.2 最终谱系显示 Explore 与 Refine 的角色互补

从根程序到最终最好程序，有 9/12 次发生宏簇变化。对这 9 次逐条沿最终父链定位第一次进入终局宏簇的事件：

- 6 次由 Explore 产生，3 次由 Refine 产生；
- CVRP 三次全部由 Explore 从 capacity/angular 类机制进入 sweep、spatial cluster 或 savings 类终局宏簇；
- OBP 三次有两次由 Explore 进入 distribution/future-gap 类终局宏簇；
- TSP 两次换簇中，一次由 Explore 进入 completion rollout，一次由 Refine 进入 explicit search。

与此同时，12 个最终最好程序的出生事件全部是 Refine。这个组合比单独看 improve rate 更接近 proposal geometry：

$$
\text{Explore: sparse structural relocation}
\quad\longrightarrow\quad
\text{Refine: within-region development and final hit}.
$$

它不证明固定 30% Explore 最优，也不证明所有 Explore 换簇都有用；它只否定了“Explore 只是 destructive large mutation、可以仅凭普通 improve rate 删除”的读法。

### 3.3 同一个锚点尺度会使大量跨簇 child 出生即淘汰

新 child 锚点的访问次数为 0，因此它能获得的最大锚点乐观项是 $s$。若出生时

$$
q_{\mathrm{child}}+s<q^*(r),
$$

那么该路线内至少有一个锚点分数始终不低于 $q^*(r)$；新 child 当时无法入选，不入选又使其乐观项不变。只要路线前沿不下降，它以后也永远无法获得第一次继续生成。本文把这类 child 记为“出生即淘汰”。

| 任务 | 出生即淘汰，Refine / Explore | 有观察后代，Refine / Explore | 后代越过原父代，Refine / Explore |
| --- | ---: | ---: | ---: |
| TSP | 13.6% / 55.9% | 75.3% / 29.1% | 28.8% / 8.7% |
| CVRP | 3.8% / 86.9% | 88.8% / 7.0% | 43.7% / 3.9% |
| OP | 48.4% / 85.7% | 37.5% / 10.8% | 10.0% / 3.4% |
| OBP | 5.6% / 35.9% | 85.5% / 32.7% | 34.6% / 10.3% |

跨静态宏簇 child 的淘汰更集中：TSP 73.5%、CVRP 81.8%、OP 84.8%、OBP 33.9%；不换宏簇的 child 分别为 17.1%、17.5%、51.5%、11.1%。这说明当前联合系统对结构跃迁有明确的即时质量门槛。

Explore 的 regress child 最终被后代“救回并超过原父代”的比例也很低：TSP 3.1%、CVRP 1.8%、OP 0.8%、OBP 4.3%。相应 Refine 比例为 15.8%、21.8%、2.4%、16.5%。这些是实际分配下的 realized value，不是强制继续若干步后的潜力。尤其在 CVRP，Explore child 一旦真的得到后代，后代越过原父代的条件比例并不低；但只有 7.0% 的 Explore child 获得了这种机会。按已获后代条件化会产生严重的后选择偏差。

所以“Explore 后代价值低”与“当前分配器不让 Explore 长出后代”不能从完整搜索日志中彻底拆开。现有证据能确认的是：**proposal 产生的跨簇质量退步经常大于由 Refine bootstrap 标定的 $s$，锚点分配随后机械地将其截断。**

### 3.4 Formation history 的作用没有显示单调适用区

第三轮固定锚点实验已证明 parent path 相对 code-only 在四任务上改善 conditional $\Delta q$。进一步按实验预先使用的锚点质量三等层拆分，配对均值差如下：

| 任务 | 低质量层 | 中质量层 | 高质量层 |
| --- | ---: | ---: | ---: |
| TSP | +1.042 [−1.022, 4.272] | +1.492 [0.609, 2.804] | +1.165 [0.197, 2.228] |
| CVRP | +3.724 [2.211, 5.487] | +5.249 [2.345, 8.057] | +3.920 [−0.351, 7.615] |
| OP | +0.105 [−0.230, 0.480] | +0.406 [0.152, 0.777] | +0.241 [0.088, 0.398] |
| OBP | +88.6 [−67.3, 318.5] | +56.6 [−56.6, 192.0] | +190.4 [11.0, 392.8] |

12 个任务乘质量层的点估计全部为正，7 个区间不跨 0；收益并不只出现在低质量锚点，也不随锚点质量单调增加。TSP、CVRP、OP 的中质量层点估计最大，OBP 则是高质量层最大。

本文还事后检查了深度、发生时点、历史 improve/regress 比例、路径净增益、正负累计变化和最后一步 $\Delta q$ 与 parent-path 单步效应的 Spearman 相关。每任务只有 18 个锚点，多个指标的方向跨任务不一致；没有一个指标形成可复用、单调的 effect modifier。OP 上较晚锚点效应更大这一相关是事后探索，不能单独据此设计成熟度开关。

当前最稳妥的表述仍是：formation history 在多种质量状态上提供局部约束，主要减少灾难性偏离；现有数据没有找到“只有某种深度、质量或改进历史才有用”的可靠边界，也没有识别它对 Refine 与 Explore 各自的交互作用。

### 3.5 Proposal 五个问题的当前答案

| 问题 | 当前答案 | 仍缺的证据 |
| --- | --- | --- |
| Refine / Explore 谁贡献 global best | Refine 突破更多且 12/12 生成终局；Explore 仍贡献 20%–36% 累积全局增益 | 固定锚点意图消融 |
| Explore 是否更多换簇 | 静态宏簇代理上为 37%–54%，稳定高于 Refine 的 2%–12% | 行为簇或人工语义标签验证 |
| 换簇后长期收益如何 | 多数跨簇 child 出生即淘汰；被真实展开的少量样本受强选择偏差 | 每个换簇 child 的强制短续段 |
| Formation history 何时最有效 | 三个质量层都可出现正效应，没有稳定单调条件 | 更大锚点样本与 history × intent 因子实验 |
| 四任务 proposal geometry 有何差异 | TSP/CVRP 显示“换簇后 Refine”链；OP 换簇多但存活与突破少；OBP 结构变化多但最终版本差小 | 独立 proposal 与 allocation 实验 |

## 4. 两种几何的耦合

### 4.1 Route 不保持 family，跨族搜索主要发生在 route 内

V9.7 的路线层按初始来源分配，proposal 却可以在同一路线内把 local score 改成 rollout、把 angular heuristic 改成 sweep partition、把 best fit 改成 online distribution。根到终局有 9/12 次换宏簇，说明路线边界没有把算法空间分割成稳定盆地。

因此目前更准确的运行抽象是：

$$
\text{early provenance selection}
\rightarrow
\text{within-route anchor search}
\rightarrow
\text{occasional family transition}
\rightarrow
\text{Refine development}.
$$

这解释了为什么 TSP/CVRP 即使路线层几乎没有持续探索，仍能在最终谱系中出现实质算法结构变化：proposal 在主路线内部制造了这些变化，八条路线没有分别保持八个固定 family。

### 4.2 一个由 Refine 标定的 $s$ 同时承担了三种不同比较

$s$ 来自初始化 Refine 的一步绝对变化中位数，但随后同时进入：

1. 不同根来源之间的路线竞争；
2. 同一路线内常规 Refine 锚点竞争；
3. Explore 大幅换簇后的新锚点存活。

经验上，这三个尺度并不一致：

- TSP、CVRP、OP 初始化路线质量总跨度分别达到约 3.8–6.2、5.5–7.2、12.4–19.3 个 $s$，路线补偿很快失去作用；
- Explore 的修改和质量变化远大于 Refine，大量 child 的退步超过一个 $s$，又在锚点层出生即淘汰；
- OBP 的初始跨度只有 0.54–1.97 个 $s$ 且最高值大量并列，因此路线竞争维持更久。

这些证据没有给出一个可直接迁移的 $s$ 倍率。共享尺度把路线覆盖、局部锚点回访和跨簇过渡三个不同决策问题压成了同一个一步变化单位。

### 4.3 当前瓶颈位于 proposal 与 allocation 的接口

只看 immediate improve，会低估 Explore 的结构迁移和正尾；只看 Explore 的低后代率，又会把 allocation 的即时淘汰算到 proposal 头上。反过来，路线层没有持续多路线比较，也不意味着 proposal 缺少算法簇多样性，因为簇可以在线路内部变化。

现有联合系统暴露出的因果链是：

$$
P(x'\mid x,h,o)
\rightarrow
\mathrm{child}(x')
\rightarrow
\left[q(x')+\mathrm{bonus}\right]
\rightarrow
\mathrm{realized\ search\ distribution}.
$$

第一步产生局部或跨簇 child，第二步决定它能否获得继续生成，最终形成有限预算下实际观察到的搜索分布。这条链支持“生成与分配理论上耦合、实验上拆开”的研究主线，也给 V9.8 的开发顺序设置了硬约束：在评价新的 proposal 时不能继续让现行锚点分数决定它是否获得最小续段；在评价新的 allocation 时必须冻结 proposal 分布。

## 5. 对 V9.8 之前实验程序的约束

本文不据此选择 V9.8 公式，只确定下一轮实验必须补齐什么。

### 5.1 Allocation 线

1. 固定 V9.7 parent path、Refine/Explore 和锚点规则，比较 Single/Greedy、Uniform 或 Random over $K$ routes、Adaptive allocation。Single 到 Uniform/Random 测池效应，Uniform/Random 到 Adaptive 测分配效应。
2. 在正式自适应分配前给予每条路线相同的强制前缀预算，使至少一个早期续段可观测。没有 overlap 时，不训练 continuation predictor。
3. 同时报告静态机制标签与行为距离；若根仍集中于同一宏簇，应先修改初始化或投资单位，再评价 allocator。
4. 评价量分为 best-at-budget、held-out、路线覆盖、进入好簇概率和成本；干预率不是目标。

### 5.2 Proposal 线

1. 固定锚点、parent path、采样 seed 与 evaluator，直接比较 Refine 与 Explore 的有效率、$\Delta q$、宏簇切换和行为切换。
2. 对每个有效 child，特别是跨簇 regress child，强制给予相同的短续段预算，再测是否越过原父代。这样才能把 proposal 的 continuation potential 与现行分配器的 realized value 分开。
3. 将 history × intent 作为因子设计，检验来时路是在帮助 Refine 聚焦，还是同时压缩 Explore 的跨簇 proposal mass。
4. 先测固定 0.7/0.3 两臂各自的 transition kernel，再讨论比例或状态依赖切换。

### 5.3 组合线

只有在 proposal 的换簇与续段价值、allocation 的池效应与自适应效应分别被识别后，才把两者组合成 V9.8。最终版本可以同时重构两层，但证据链必须保留两条独立对照；否则性能变化仍无法回答究竟是“能到达什么”改变，还是“预算把什么兑现出来”改变。

## 6. 结论边界

现有证据足以把 V9.7 定义为一个干净但几何错位的 baseline：

- 路线层实际是早期来源门控，且具有不可恢复淘汰；
- route identity 不保持稳定 family identity；
- Refine 是高命中的局部开发分布；
- Explore 是低命中、较大步、更多静态换簇的分布；
- 大量 Explore 与跨簇 child 的潜力在获得第二步之前就被当前锚点几何截断；
- parent path 的单步价值跨质量层存在，但尚无可靠适用条件或 intent 交互结论。

现有证据仍不足以判断：被放弃路线的真实 ceiling、静态宏簇是否等于行为盆地、Explore 的强制续段价值、0.7/0.3 的最优性，以及任何新的 continuation-value 公式。以上空白应由分离后的 Stage A 与 Stage B 实验填补；当前日志的事后相关不承担这些结论。

## 相关文档

- [V9.7 完整机制设计](../methods/TraceAAD-v9.7完整机制设计.md)
- [V9.7 机制有效性与任务异质性](TraceAAD-V9.7机制有效性与任务异质性.md)
- [V9.7 路线级分配诊断](TraceAAD-V9.7路线级分配诊断.md)
- [固定锚点单步生成识别实验](../experiments/TraceAAD-固定锚点单步生成识别实验.md)
- [L2 预算分配](../research/L2-预算分配.md)
- [L3 单步生成](../research/L3-单步生成.md)
- [L4 诊断](../research/L4-诊断.md)
- [BaSE 阅读笔记](../references/LLM自动算法设计方法阅读笔记/28-Compute-Allocation-BaSE.md)
