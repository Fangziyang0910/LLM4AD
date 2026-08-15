# TraceAAD 研究问题

TraceAAD 的两个核心研究对象是：

- **轨迹条件生成** $P(x_{t+1}\mid x_t,h_t,o_t)$：轨迹上下文和搜索意图如何改变 LLM 在算法空间中的下一步提议。
- **轨迹感知的计算分配** $\mu(a_t\mid\mathcal H_t)$：有限预算下，下一份计算投给哪条路线、哪个历史状态。

二者实验上拆开、理论上耦合。有效分配建立在提议异质性上；好的跨族提议若不被维持，会被调度浪费。当前判断见[研究认识](../knowledge/研究认识.md)。当前可运行规范为 [V9.7](../methods/TraceAAD-v9.7完整机制设计.md)；[V9.8](../methods/TraceAAD-v9.8完整机制设计.md) 是尚未实现的 hypothesis-centered 机制设计稿。

控制层次是这两个对象的操作分解，不是第三套科学问题。层次顺序是运行顺序，不是研究优先级。V9.7 的静态簇与搜索几何已经读出；下一步按 V9.8 设计分别验证 hypothesis boundary、区域级与 child-chain 短续段、History × Intent、discovery source 和 structured lane schedule。

## 层次

| 层次 | 服务哪个对象 | 当前状态 |
| --- | --- | --- |
| [L0 状态表示](L0-状态表示.md) | 两个对象的共同状态 | V9.7 route 是来源；V9.8 提议用 Explore-defined hypothesis 作为操作单位，真实性待测 |
| [L1 初始化](L1-初始化.md) | 分配的投资单位 | V9.8 保留 8 roots 但给每个 root hypothesis 等长 probe；尚无独立实验 |
| [L2 预算分配](L2-预算分配.md) | 轨迹感知分配 | V9.8 设计拆成 discovery、continuation 与 exploitation 三通道；source 与固定配额均待验证 |
| [L3 单步生成](L3-单步生成.md) | 轨迹条件生成 | 来时路已有单步证据；V9.8 将 Refine/Explore 定义为发展/创建 hypothesis，仍需固定锚点验证 |
| [L4 诊断](L4-诊断.md) | 任务几何与成本 | 静态代理已有描述，真实簇稀有度、族内可改进性与簇间方差仍需受控读出 |

## 开放问题

1. Explore-defined hypothesis 是否比 root route 更接近可利用的算法区域；Refine/Explore 意图边界与真实 family switch 的偏差有多大。
2. Hypothesis-level probe 与沿 Explore child 的 descendant-chain continuation 在 0/3/5 步下分别显示怎样的 parent recovery；局部重选改变了什么。
3. Single、Uniform 与三通道 allocation 能否分别识别 pool value、投资单位、discovery source、protected schedule 和 routing value。
4. Refine regress child 的延迟改进有多少会被 hypothesis 内局部选择截断。
5. Explore 创建的新 hypothesis 有多少只是在重访已有静态/行为机制代理；当前局部 parent path 是否缺少全局 explored-hypothesis awareness。
6. Recent realized gain 能否预测额外计算的边际价值，还是主要度量容易进步区间的短期 momentum。
7. 四个任务的 effective discovery-development share、簇稀有度、族内可改进性与簇间质量方差能否解释 V9.8 的任务依赖收益。

## 旧编号对照

| 旧文件 | 归入 |
| --- | --- |
| [RQ-008](RQ-008-V9.2底层反思与下一版轨迹闭环.md) | L0 历史索引 |
| （原先无独立 RQ） | L1 |
| [RQ-006](RQ-006-轨迹中心预算分配.md)、[RQ-007](RQ-007-V9.2锚点与局部轨迹窗口.md) | L2 历史记录 |
| [RQ-003](RQ-003-轨迹上下文与搜索评分.md) | 已拆入两个核心对象 |
| [RQ-009](RQ-009-锚点历史上下文.md) | L3 上下文，已收口 |
| [RQ-001](RQ-001-程序膨胀.md) | L4 |
