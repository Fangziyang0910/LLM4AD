# TraceAAD V9.14：单根算法树上的轨迹条件进化

> V9.14 在一棵带虚拟根的单父纯算法树上运行。两条机制主线是轨迹条件的单步生成与算法节点上的预算分配；固定协议为 8 个初始算法、Refine 0.7 / Explore 0.3 与最近 8 条父代来时路。每次成功生成并评价的候选形成一个独立 Algorithm 节点。

## 1. 状态与运行单位

搜索状态只包含 Algorithm 节点。节点保存代码、fitness、父节点、Idea 和被选为父代的累计次数。节点的父链构成改进来时路；每次评价成功的候选向树中增加一个节点。

## 2. 搜索状态：带虚拟根的单父纯算法树

搜索开始时建立虚拟根（ID=0），其下生成 $K=8$ 个初始算法。此后每个成功候选按生成时选中的算法建立唯一父子关系。

节点保存完整代码、真实 fitness、父节点引用 `parent_id`、被选为父代的累计次数 $c(a)$ 和模型声明的 Idea。自增 ID 记录创建次序，用于平局判定。maximize 任务取 $q(a)=\mathcal E(P_a)$，minimize 任务取 $q(a)=-\mathcal E(P_a)$。质量、相对父节点的结果和全局最优算法都在使用时由这些事实直接计算。

父节点的 $c(a)$ 在 evaluator 完成一次调用后增加。评价成功时创建子节点；评价失败时不创建节点，但该次预算已计入 $c(a)$。

节点的形成路径即其祖先链上的算法节点序列：

$$
\tau(a)=(a_1,a_2,\ldots,a_k=a).
$$

## 3. 生成即评价：线性的预算口径

每次 LLM 返回响应后，程序文本立即交给 evaluator 真实运行并计入评价预算。预算口径为：

- 每个模型响应消耗一次真实评价；
- 每次成功评价向树中增加一个独立节点；
- 评价失败不创建节点，并记录 evaluator 返回的失败原因。

正式预算为每次运行 $B=1000$ 次真实评价。evaluator 调用次数是运行中唯一的进度计数。

## 4. 算法节点上的预算分配

分配机制 $\mu(a_t\mid\mathcal H_t)$ 决定下一次生成从哪个算法出发。所有有效算法使用同一分数：

$$
S_t(a)=q(a)+\frac{1}{\sqrt{c_t(a)+1}}.
$$

$q(a)$ 是算法自身质量，$c_t(a)$ 是它已获得的评价预算数。同分时优先选择 $c_t(a)$ 更小、创建更早的算法。

初始化完成 8 个根节点后直接进入正式搜索。每个根节点的初始访问次数为 $0$。

## 5. 轨迹条件的单步生成

单步生成的条件分布为

$$
P(x_{t+1}\mid x_t,h_t,o_t),
$$

其中 $x_t$ 是选中算法的完整代码，$h_t$ 是其父链上最近 8 个祖先节点的形成元数据，$o_t$ 是本轮意图。每条事件用两行表达：当时的 Idea，以及合并了定性结果和 Fitness 变化的 Result。Refine 以 0.7 概率抽取，聚焦当前设计方向的局部修改；Explore 以 0.3 概率抽取，寻求结构性不同的改进方向。

提示词只要求输出一句 500 字符以内的 Idea 和一份完整 Python 程序。提取时优先使用代码围栏，其次使用 `Code:` 之后的文本，否则使用整个响应。每个模型响应都进入 evaluator，程序是否有效由真实运行决定。上下文固定使用最近 8 条形成事件。

来时路在单根树上取得直接定义：选中节点的父链即来时路，形成路径不再经过额外的索引表。

## 6. 完整运行协议

````text
Input: task, evaluator, LLM, real evaluator budget B = 1000

Generate K = 8 valid initial algorithms under the virtual root;

While evaluator budget remains:
    Score every algorithm by q(algo) + 1 / sqrt(count(algo) + 1).
    Select the highest-scoring algorithm.
    Build the selected algorithm's parent improvement path, at most 8 ancestor events.
    Draw Refine with probability 0.7, otherwise draw Explore.
    Request one concise Idea and one complete program.
    Extract code text and evaluate it for real; this consumes one budget unit.
    Increment the selected algorithm's count.
    If evaluation succeeds:
        create and insert a child Algorithm node into the tree.
    Reselect from the updated tree.

Return the best algorithm by the true objective.
````

停止条件是评价预算耗尽，不因连续无改善提前停止。最终最好算法只由真实质量决定；质量相同时保留当前结果。

## 7. 研究定位

V9.14 以 Algorithm 节点数描述有效搜索状态，以 evaluator 调用数描述真实评价预算。当前质量—访问次数分配是后续完善预算分配机制的基点：后续任务是从现有实验读出算法簇结构与来时路的发展事实，再定义用于下一份原子预算的分配信号。

> **V9.14 在统一的单父纯算法树上，用改进来时路条件化单步生成，并逐步完善有限评价预算的分配机制。**
