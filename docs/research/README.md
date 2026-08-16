# TraceAAD 研究问题

TraceAAD 的两个核心研究对象是：

- **轨迹条件生成** $P(x_{t+1}\mid x_t,h_t,o_t)$：轨迹上下文和搜索意图如何改变 LLM 在算法空间中的下一步提议。
- **轨迹感知的计算分配** $\mu(a_t\mid\mathcal H_t)$：有限预算下，下一份计算投给哪条路线、哪个历史状态。

二者实验上拆开、理论上耦合。有效分配建立在提议异质性上；好的跨族提议若不被维持，会被调度浪费。当前判断见[研究认识](../knowledge/研究认识.md)。[V9.7](../methods/TraceAAD-v9.7完整机制设计.md) 是当前已有正式结果的版本；[V9.8](../methods/TraceAAD-v9.8完整机制设计.md) 已实现并完成 Stage P，未完成的正式搜索不构成性能证据。

控制层次是这两个对象的操作分解，不是第三套科学问题。层次顺序是运行顺序，不是研究优先级。V9.7 的静态簇与搜索几何已经读出；V9.8 将在线循环收缩为“状态与评分 → operator/hypothesis/anchor 选择 → 上下文 → 单步生成与评价”。Stage P 已识别 History × Intent 与两种强制续段的局部行为；下一步是用 Stage A 分开验证 hypothesis 聚合、边界宽限与历史发展收益。

## 层次

| 层次 | 服务哪个对象 | 当前状态 |
| --- | --- | --- |
| [L0 状态表示](L0-状态表示.md) | 两个对象的共同状态 | V9.7 route 是来源；V9.8 用 Explore-defined hypothesis 作为轨迹分段标记，真实性待测 |
| [L1 初始化](L1-初始化.md) | 初始起点与任务尺度 | V9.8 保留 8 roots 与每 root 一次 bootstrap，不增加三步 probe |
| [L2 预算分配](L2-预算分配.md) | 轨迹感知分配 | V9.8 每步按 operator-specific 的 $Q/U/C/M$ 分数选择 hypothesis，再选锚点；无未来预算承诺 |
| [L3 单步生成](L3-单步生成.md) | 轨迹条件生成 | Stage P 已确认 Refine/Explore 的行为差异及 parent path 对 Refine 的局部价值；真实 family 与 Explore 专用上下文仍待验证 |
| [L4 诊断](L4-诊断.md) | 任务几何与成本 | 静态代理已有描述，真实簇稀有度、族内可改进性与簇间方差仍需受控读出 |

## 开放问题

1. Explore-defined hypothesis 是否比 root route 更接近可利用的算法区域；Refine/Explore 意图边界与真实 family switch 的偏差有多大。
2. 已观察的 Explore child 有限视野可恢复性能否预测边界宽限的任务依赖收益；该事后解释不能代替 Stage A 对照。
3. Single、Uniform、root route 聚合与逐步 hypothesis 分数能否分别识别 pool value、聚合单位、观察不足、边界宽限与历史发展收益。
4. Refine regress child 的延迟改进有多少会被 hypothesis 内局部选择截断。
5. Explore 创建的新 hypothesis 有多少只是在重访已有静态/行为机制代理；当前局部 parent path 是否缺少全局 explored-hypothesis awareness。
6. 历史平均 realized gain 能否预测额外计算的边际价值，还是主要度量容易进步区间的短期 momentum。
7. 四个任务的 operator 实际份额、簇稀有度、族内可改进性与簇间质量方差能否解释 V9.8 的任务依赖收益。

## 旧编号对照

| 旧文件 | 归入 |
| --- | --- |
| [RQ-008](RQ-008-V9.2底层反思与下一版轨迹闭环.md) | L0 历史索引 |
| （原先无独立 RQ） | L1 |
| [RQ-006](RQ-006-轨迹中心预算分配.md)、[RQ-007](RQ-007-V9.2锚点与局部轨迹窗口.md) | L2 历史记录 |
| [RQ-003](RQ-003-轨迹上下文与搜索评分.md) | 已拆入两个核心对象 |
| [RQ-009](RQ-009-锚点历史上下文.md) | L3 上下文，已收口 |
| [RQ-001](RQ-001-程序膨胀.md) | L4 |
