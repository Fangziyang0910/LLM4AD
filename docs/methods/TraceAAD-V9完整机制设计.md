# TraceAAD V9：真实匹配历史与完整树搜索

## 1. 研究问题

自动算法设计把 LLM 作为算法变异算子：给定问题、当前程序和评价器，模型提出新的算法思想并实现为可执行代码，评价器返回质量，搜索器再决定下一次从哪里继续。V9 研究其中的生成条件问题：

> 当前程序怎样形成，以及从这个程序出发已经尝试过什么，能否改善 LLM 的下一步算法生成？

因此，V9 将算法改进轨迹作为生成条件，同时用完整树保存所有已经评价过的程序，使任一历史状态都能重新获得生成机会。

## 2. 搜索状态

搜索状态是一棵带虚拟根的单父程序树。初始化时，虚拟根连接 10 个有效根程序；正式搜索不再新增根。

每个程序节点保存完整代码、原始 fitness、有向质量、唯一结构父节点、子节点、访问与扩展次数，以及当前子树的最好质量。最小化任务取 $y(n)=-fitness(n)$，最大化任务取 $y(n)=fitness(n)$。

每条父子边保存本轮 Idea、生成算子、父子质量变化、`improve` / `plateau` / `regress` 结果和发生顺序。节点的形成路径为：

$$
\tau(n)=(n_0,e_1,n_1,\ldots,e_k,n_k=n).
$$

每个节点由当前程序及其唯一形成路径共同确定生成状态。

## 3. 轨迹条件生成

当前节点的上下文包含四类信息：

1. 当前完整程序和原始 fitness；
2. 从根到当前节点最近 8 条形成边；
3. 从当前节点发起的最多 8 个直接分支；
4. 双轨迹算子选中的参考分支及其最近 8 条形成边。

形成事件包含 Idea、父子 fitness、结果分类、是否刷新当时全局最好，以及代码变化摘要。当前程序完整展示，祖先代码不重复展示。

直接分支先取子树最好质量最高的 4 个，再按创建时间补足。摘要同时显示直接结果和该分支后来达到的子树最好质量，使模型能够看到局部尝试与后续发展之间的关系。

历史只使用真实发生的事件，不由模型重新总结，也不混入其他节点的随机信息。上下文超限时先压缩参考历史，再减少直接分支；任务契约、当前代码和当前形成历史保持不变。

## 4. 生成意图

V9 使用四类语义算子：

| 算子 | 生成任务 | 参考信息 |
| --- | --- | --- |
| `trace_ideate` | 从当前来时路提出新方向 | 无 |
| `trace_refine` | 深化当前设计 | 无 |
| `trace_synthesize` | 综合当前分支与另一条路线的历史 | 另一条路线 |
| `trace_transfer` | 将另一条路线的机制适配到当前程序 | 另一条路线 |

若不存在合格的其他路线，只使用前两个算子。双轨迹参考从其他根路线的子树最好程序中抽取。若候选质量的全树中秩百分位为 $Z_i$，温度为 $\tau=0.2$：

$$
P(i)=\frac{\exp((Z_i-\max_j Z_j)/\tau)}
{\sum_j\exp((Z_j-\max_k Z_k)/\tau)}.
$$

算子规定本轮的语义任务，不根据短期命中率在线调权。

每个 child slot 生成一条 Idea 和一份完整可执行程序。一次 expansion batch 最多生成两个 sibling；二者共享父节点、算子、参考节点和批前全局最好。

## 5. 递归预算分配

子树最好值为：

$$
G(n)=\max\left(y(n),\max_{c\in Children(n)}G(c)\right).
$$

全树质量转为中秩百分位。若严格小于 $x$ 的节点数为 $L$，等于 $x$ 的节点数为 $E$，总节点数为 $M>1$：

$$
Z(x)=\frac{L+(E-1)/2}{M-1}.
$$

只有一个值或全部同分时，定义 $Z=0.5$。该变换只用于调度。剩余预算比例为：

$$
r_t=\operatorname{clip}\left(\frac{T-t}{T},0,1\right).
$$

从父节点 $p$ 进入已有子节点 $c$ 的分数为：

$$
S_{down}(c\mid p)=Z(G(c))+\lambda_0r_t
\sqrt{\frac{\log(1+N(p))}{N(c)}},
\qquad \lambda_0=0.1.
$$

节点还拥有 `new_child` 选项。第 $b$ 个扩展批的回报为：

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

在每个节点，`S_new` 与所有 `S_down` 直接竞争。`new_child` 获胜时生成新 batch；子节点获胜时沿树递归向下。树不设深度和 child 数上限。

## 6. 预算与更新

正式预算为 1000 次 evaluator 调用，包含初始化。evaluator 启动即消耗一次预算；只有得到有限 fitness 的候选进入树。失败批计入扩展经验，回报为 0。

有效节点写入后，沿父链更新 $G$、批回报和全局最好程序。最终程序按任务原始目标选择，完全同分时偏好代码更短、发现更早的程序。

````text
Generate 10 unique valid roots.

While evaluator budget remains:
    Compute subtree values G and midrank qualities Z.
    Recursively compare S_down and S_new from the virtual root.
    Draw one available semantic operator.
    Sample a reference route when the operator requires it.
    Build matched current and reference histories.
    Generate at most two Idea + Code siblings.
    Evaluate each parsable program.
    Add every finite-fitness program as a single-parent child.
    Update subtree values, batch returns, and global best.

Return the best unique program by the true objective.
````
