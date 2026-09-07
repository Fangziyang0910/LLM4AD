# E1-A.1：质量条件下的行为增量检验

## 1. 研究问题

E1-A 没有支持“行为近邻可以直接共享 Refine 响应”，但留下了两个不能混为一谈的现象：直接行为核的概率误差略低于 M1，而质量核又优于行为核；同时，行为静态特征的 AP 和 top-20% lift 略有改善。因此 E1-A.1 只回答一个更窄的问题：

$$
\boxed{\text{控制 parent fitness 相近后，BehaveSim 是否仍提供额外的 Refine 响应信息？}}
$$

本实验是 E1-A 后验发现驱动的独立分析，不能反过来改变 E1-A 的主结论或原筛选门槛。它不新增 LLM 生成、不新增正式 evaluator 调用，也不修改 V10.6 调度器。

## 2. 数据与时间协议

复用 E1-A 在 `experiments/_logs/refine_e1_20260907/` 冻结的 15 路 V10.6、行为距离矩阵和逐次 Refine 结果。保持：

- 每个 run 独立，严格执行 $H_{t^-}\rightarrow\hat y_t\rightarrow y_t\rightarrow H_t$；
- 每路前 50 次 Refine 作为 warmup，不进入主指标；
- 当前 parent、未来节点和未来结果不可进入邻域；
- invalid generation 与 evaluation failure 计为未改善；
- 主分析排除当前 parent 自身历史，并保留 first-parent-use 子集；
- 几何与响应池仍限于此前已经成为 Refine parent 的节点。后者是既有画像覆盖带来的协议边界，不代表完整 archive geometry。

## 3. 三个固定估计器

在时刻 $t$，先从历史可见、不同于当前 parent 的节点中取 fitness 距离最近的 $k=10$ 个节点。质量带宽 $h_q$ 为这组中最远节点的质量距离；行为带宽 $h_B$ 为同一质量局部组中最大的 BehaveSim 距离。带宽下限固定为 $10^{-6}$。

质量权重为：

$$
w_i^Q=\exp\left[-\frac12\left(\frac{|f_i-f_n|}{h_q}\right)^2\right].
$$

联合权重为：

$$
w_i^{QB}=w_i^Q\exp\left[-\frac12\left(\frac{d_B(i,n)}{h_B}\right)^2\right].
$$

若同一历史 parent 已执行多次 Refine，每次历史动作都是一个响应样本并继承该 parent 的权重。预测继续使用 E1-A 固定的收缩强度 $\alpha=10$ 和当时的全局平滑 prior：

$$
\hat p(n)=\frac{\alpha\pi_t+\sum_i w_i y_i}{\alpha+\sum_i w_i}.
$$

比较三项：

1. `Q_kernel`：只用 $w_i^Q$；
2. `QB_kernel`：使用 $w_i^{QB}$；
3. `QB_permuted`：在每个 fitness-local 邻域内部随机置乱行为权重与 parent 的对应关系，再计算联合核。

置乱固定为 1000 次，seed=`20260907`。它保留每次预测的质量邻居、行为权重分布、历史样本量和标签，只破坏“哪个行为距离属于哪个 parent”的对应关系。当前 parent 无行为画像时，`QB_kernel` 回退到相同时间点的 `Q_kernel`，并另报完整画像子集。

## 4. 指标与事先固定的判断规则

主比较为 `QB_kernel − Q_kernel` 的逐 run 配对 Brier 差，并报告 run 宏平均、run bootstrap 95% 探索性区间、任务方向和 log loss。置乱对照使用 1000 个 `QB_permuted` 的 run 宏平均 Brier 分布。

只有同时满足以下条件，才把结果记为“观察到行为在质量之外的预测增量”：

1. `QB_kernel` 相对 `Q_kernel` 的 run 宏平均 Brier 至少降低 2%；
2. 至少 3/5 个任务的 pooled Brier 同向改善；
3. run 宏平均 log loss 不变差；
4. 真实 `QB_kernel` 的 run 宏平均 Brier 优于至少 95% 的 `QB_permuted`。

AP、run 宏平均 top-20% lift、top 组正部增益用于控制器相关的后验探索，不参与上述判定。另报告 first-parent-use 和完整行为画像子集。由于 E1-A.1 来源于 E1-A 的后验发现，所有区间和置乱分位只作探索性证据，不表述为预先注册的确认性显著性检验。

## 5. 决策含义

- 若通过：说明 BehaveSim 在质量局部性之外仍有增量，再设计独立数据或固定锚点确认。
- 若不通过：停止将 BehaveSim 用作 Refine evolvability 的直接共享坐标；质量核也只有在独立回放中同时改善校准与预算排序后，才考虑进入控制器。
- 无论结果如何，BehaveSim 仍可转向更符合外部行为含义的问题，例如 stagnation/revisit detection、Pivot switching 和 Fuse complementarity。

