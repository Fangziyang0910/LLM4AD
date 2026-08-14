# TraceAAD V9-Core：真实匹配历史与完整树搜索

> 状态：历史版本，不定义当前方法。正式结果见[实验总汇](../experiments/实验总汇.md)，机制诊断见[版本谱系与实验事实](../analysis/TraceAAD-版本实验事实与机制诊断.md)。当前方法为 [V9.7](TraceAAD-v9.7完整机制设计.md)。

## 1. 科学目标

V9-Core 研究的问题是：在保持既有树搜索骨架基本不变时，把与当前程序严格匹配的真实改进历史交给模型，能否形成更好的下一步生成条件？

每次评价闭环为：选择一个程序节点，构造它的形成历史和已测试分支，生成 `Idea + Code`，真实评价，并把新程序及结果写回树。完整树、算子与预算分配是支撑机制；核心信息是当前程序怎样形成以及从这里已经尝试过什么。

V9-Core 有意接近 V8.2，删除趋势、路线信用、算子信用、固定深度和额外反思调用。该设计使历史输入成为可讨论的主要变量，但独立版本仍不是受控消融。

## 2. 搜索表示

搜索状态是一棵带虚拟根的完整单父代程序树。初始化后虚拟根连接 10 个有效根程序，不再新增根。

每个程序节点保存完整代码、原始 fitness、有向质量、唯一结构父节点、子节点、访问与扩展次数，以及当前子树最好程序。最大化任务取 $y(n)=fitness(n)$，最小化任务取 $y(n)=-fitness(n)$，统一为越大越好。

每条父子边记录本轮 Idea、算子、实际父子质量变化、`improve / plateau / regress` 结局和发生顺序。参考程序只提供生成信息，不构成第二结构父代。因此每个节点唯一确定一条形成路径

$$
\tau(n)=(n_0,e_1,n_1,\ldots,e_k,n_k=n).
$$

树与边保存已经发生的事实，不把局部改善解释为思想的长期因果信用。

## 3. 真实匹配历史

对当前节点 $n$，生成上下文只使用与该节点真实匹配的信息：

1. 当前完整程序和原始 fitness；
2. 从所属根到 $n$ 的最近 8 条形成边；
3. 从 $n$ 已测试的最多 8 个直接分支；
4. 双轨迹算子所选参考程序及其最近 8 条形成边。

形成事件包含 Idea、父子 fitness、结果分类、是否刷新当时全局最好，以及 LOC 与代码变化比例。当前程序完整展示，祖先代码不重复展示。

直接分支先取 subtree-best 最高的 4 个，其余位置按最近创建补足，最后按真实创建顺序呈现。摘要同时显示直接结果和该分支后来达到的 subtree-best。后者只表示到达事实，不证明直接边上的 Idea 导致了后续最好结果。

历史不从其他节点随机借用，不由模型重新总结，也不包含未发生的推断。若上下文超限，先缩短参考历史，再减少直接分支；双轨迹仍无法容纳时退化为单轨迹算子。任务契约、当前完整代码和当前形成历史不为满足长度限制而伪造或删除。

## 4. 初始化与生成

初始化生成 10 个评价有效且代码不同的根程序。每个 child slot 只进行一次模型调用，输出一条 Idea 和一份完整可执行程序。一次 expansion batch 最多生成两个 sibling；二者共享选中节点、算子、参考节点和批前全局最好。

每轮从可用算子中等概率抽取：

| 算子 | 生成意图 | 额外参考 |
| --- | --- | --- |
| `trace_ideate` | 从当前来时路提出新方向 | 无 |
| `trace_refine` | 深化当前设计 | 无 |
| `trace_synthesize` | 综合当前与另一根分支的历史 | 另一根分支 |
| `trace_transfer` | 把另一根分支机制适配到当前程序 | 另一根分支 |

若不存在其他合格根分支，只使用前两个算子。双轨迹参考从其他根分支的 subtree-best 中抽取；若候选质量的全树中秩百分位为 $Z_i$，温度为 $\tau=0.2$，则

$$
P(i)=\frac{\exp((Z_i-\max_j Z_j)/\tau)}
{\sum_j\exp((Z_j-\max_k Z_k)/\tau)}.
$$

算子只约束本轮语义，不是对生成代码的静态分类，也不根据短期命中率在线调权。

## 5. 子树价值与递归选择

节点的乐观子树价值为

$$
G(n)=\max\left(y(n),\max_{c\in Children(n)}G(c)\right).
$$

选择前，把全树有向质量转为中秩百分位。若严格小于 $x$ 的数量为 $L$，等于 $x$ 的数量为 $E$，节点数为 $M>1$，则

$$
Z(x)=\frac{L+(E-1)/2}{M-1}.
$$

只有一个值或全部同分时定义 $Z=0.5$。该变换只用于调度，不修改原始 fitness。

总评价预算为 $T$，已启动评价数为 $t$，剩余预算比例为

$$
r_t=\operatorname{clip}\left(\frac{T-t}{T},0,1\right).
$$

从父节点 $p$ 进入已有子节点 $c$ 的分数为

$$
S_{down}(c\mid p)=Z(G(c))+\lambda_0r_t
\sqrt{\frac{\log(1+N(p))}{N(c)}},\qquad \lambda_0=0.1.
$$

每个程序节点另有一个内部 `new_child` 选项，用于决定继续开直接分支还是沿已有子树深入。设节点 $n$ 已完成 $B(n)$ 个扩展批，第 $b$ 批的有效子节点集合为 $C_b$：

$$
R_b(n)=\max_{c\in C_b}Z(G(c)),
$$

空批回报为 0。以当前节点质量作为先验，$\beta=1$：

$$
Q_{new}(n)=\frac{\beta Z(y(n))+\sum_{b=1}^{B(n)}R_b(n)}{\beta+B(n)},
$$

$$
S_{new}(n)=Q_{new}(n)+\lambda_0r_t
\sqrt{\frac{\log(1+N(n))}{1+B(n)}}.
$$

在每个节点，`S_new` 与所有 `S_down` 直接竞争。`new_child` 获胜则在当前节点生成新 batch；某个子节点获胜则继续递归。没有 child 数或树深上限。

## 6. 预算、更新与最终程序

正式预算为 1000 次 evaluator 调用，包含初始化。解析失败和传输失败不消耗评价预算；一旦 evaluator 启动即消耗预算。只有得到有限数值 fitness 的候选进入树。失败批仍计入该节点的扩展经验，回报为 0。

每个有效新节点写入后，沿祖先路径更新 $G$、batch 回报和全树最好程序。搜索正常停止于评价预算耗尽；空树、持续生成失败或基础设施中止属于失败运行，不是算法提前收敛。

最终程序只按任务原始目标选择；完全同分时依次偏好非空 LOC 更少、发现更早的程序。树分数、访问次数、算子和历史长度不参与最终排序。

## 7. 算法

```text
Generate 10 unique valid roots.

While evaluator budget remains:
    Compute subtree values G and midrank qualities Z.
    Recursively compare S_down and S_new from the virtual root.
    Stop at the node whose new_child option wins.
    Draw one available semantic operator.
    If needed, sample one reference root branch.
    Build matched current and reference histories.
    Generate at most two Idea + Code siblings.
    Evaluate each parsable program.
    Add every finite-fitness program as a single-parent child.
    Update subtree values, batch returns, and global best.

Return the best unique program by the true objective.
```

## 8. 解释边界

- 完整树保存历史，不证明模型利用历史。
- 直接分支摘要把 immediate result 与 subtree-best 并列，存在把后续成功误归因于直接 Idea 的风险。
- max-backup 与自适应 `new_child` 用历史最好结果代理继续投入价值；一次幸运后代可能长期抬高祖先。
- V9-Core 的正式结果评价整套联合协议。历史的单步作用后来由固定锚点实验识别，完整搜索净收益仍需与预算分配分开判断。
