# TraceAAD V9.17：竞争质量门控的假设发现—发展进化

TraceAAD V9.17 研究一个任务无关的在线过程：面对未知的算法搜索空间，先让新方向获得一段可比较的发展，再由已经达到的真实质量决定哪些方向进入竞争集合；对竞争集合中仍能推进前沿的方向持续投入，当它们全部停滞时重新发现新方向。

完整闭环为：

$$
Discovery \rightarrow Maturation \rightarrow Competition \rightarrow Development \rightarrow Reopening.
$$

V9.17 使用两类在线事实：

- 假设前沿质量 $Q(z)$ 表示该方向已经兑现到什么水平；
- 固定发展块增益 $g(z)$ 表示该方向在当前一小段预算内是否仍有响应。

$Q(z)$ 决定假设能否进入竞争集合，$g(z)$ 只决定竞争集合中的假设是否连续获得下一个发展块。二者不相加，也不形成永久信用。

## 1. 搜索状态

### 1.1 Algorithm 节点

搜索状态是一棵带虚拟根的单父算法树。每个有效 Algorithm 节点 $v$ 保存：

- 完整程序与真实 fitness；
- 统一为越大越好的质量 $q(v)$；
- 唯一父节点；
- 生成该节点时的 Idea、实际 Change、Result 与 Fitness；
- 生成意图 Refine 或 Explore；
- 所属算法假设 ID；
- 创建次序与作为 Refine 父节点的尝试次数。

maximize 任务直接使用原始目标，minimize 任务取原始目标的相反数。所有在线质量判断都来自真实 evaluator。

### 1.2 算法假设

算法假设 $z$ 是在线分配使用的轨迹段：

- 每个有效初始根建立一个 root hypothesis；
- 每个有效 Explore child 建立一个新 hypothesis；
- Refine child 继承父节点的 hypothesis；
- 无效候选不创建 Algorithm 节点或 hypothesis。

算法假设是来源与发展过程的操作边界。它不承担真实算法簇标签；同一假设可以在后续 Refine 中发生较大变化，不同假设也可能产生相近算法。

假设前沿质量定义为：

$$
Q(z)=\max_{v\in z}q(v).
$$

达到 $Q(z)$ 且创建最早的节点记为前沿节点 $f(z)$。每个假设保存 `origin_node`、`source_hypothesis`、`primary_slots`、`last_block_gain` 和生命周期状态。

生命周期只有三种状态：

- `maturing`：新假设正在完成首次发展块；
- `active`：位于当前竞争集合；
- `reserve`：保留全部事实，不再获得本次运行的常规预算。

### 1.3 竞争集合

活跃集合 $A_t$ 最多包含 $K=8$ 个假设。假设按以下顺序排名：

1. $Q(z)$ 更高；
2. 创建更早。

记该顺序为：

$$
R(z)=(Q(z),-id(z)).
$$

当 $|A_t|=K$ 时，最低活跃前沿形成竞争线：

$$
L_t=\min_{z\in A_t}Q(z).
$$

竞争线表示有限预算下，一个新方向需要达到的当前质量水平。投入多少、历史改善多少和谱系多深都不改变竞争排名。

## 2. 轨迹条件生成

V9.17 使用 Refine 和 Explore 两种生成意图。

### 2.1 Refine

Refine 在已有算法假设内部发展。模型获得：

- 选中 Algorithm 节点的完整代码；
- 该节点父链上最近 $L=8$ 条形成事件；
- Refine 意图。

每条形成事件包含当时的 Idea、相对父代实际发生的 Change、真实 Result 与 Fitness。提示要求模型诊断当前算法的限制，保留已经形成的有效结构，提出一个连贯的下一步改进。

### 2.2 Explore

Explore 从当前排名最高的 active hypothesis 的前沿节点出发。模型获得相同的当前代码、形成历史和任务信息，并被要求提出具有不同核心机制或决策逻辑的算法方向。

有效 Explore child 建立新的 hypothesis，随后立即进入 maturation。Explore 的语义目标是改变提议分布；新 hypothesis 只记录这次结构性提议的来源，不在线认证语义新颖性。

### 2.3 输出、评价与 repair

每次模型输出一句 Idea 和一份完整程序：

````text
Idea: <one coherent algorithmic change>

```python
<complete program>
```
````

一次分配产生一个 primary candidate，并消耗一个 primary evaluator slot。有效结果创建 Algorithm 节点；无效结果保留失败记录。

一个 primary candidate 的初始程序无效时最多进行两次 repair。每次 repair 都把完整失败信息和当前程序返回模型；成功后以最终有效程序创建原父节点、原意图下的 child。repair 的 LLM 与 evaluator 调用单独记录，不产生新的分配决策，也不增加 primary slot。

## 3. 假设内部的 Refine 父节点选择

一个假设获得 Refine slot 后，在该假设内部按以下分数选择父节点：

$$
S_R(v)=q(v)+\frac{s_R}{\sqrt{n_R(v)+1}}.
$$

$n_R(v)$ 是以 $v$ 为父节点发起的 Refine primary slots 数；候选无论有效与否都增加一次。$s_R$ 是本次运行初始化后冻结的一步质量尺度。

初始化成熟结束后，收集其中所有有效 Refine 边的绝对质量变化：

$$
D_R=\{|q(v_{child})-q(v_{parent})|:\ v_{child}\text{ is a valid Refine child}\}.
$$

若 $D_R$ 非空，取：

$$
s_R=\operatorname{median}(D_R).
$$

若 $D_R$ 为空，取 $s_R=0$。随后保持不变。

选择 $S_R$ 最大的节点；同分时依次选择 $n_R$ 更小、创建更早的节点。该分数只在同一 hypothesis 内使用，用于在当前前沿与尚未充分尝试的局部状态之间重选，不参与 hypothesis 之间的竞争。

## 4. 固定发展块

一次发展块包含 $H=3$ 个 Refine primary slots。对假设 $z$ 执行一个 block：

1. 记录 $Q_{before}(z)$；
2. 连续三次执行假设内部父节点选择、Refine 生成和真实评价；
3. 每次评价后立即更新树、$n_R$ 与 $Q(z)$；
4. 记录 $Q_{after}(z)$。

block gain 为：

$$
g(z)=Q_{after}(z)-Q_{before}(z).
$$

当且仅当 $g(z)>0$ 时，该 block 被记为成功。退化节点、无效候选和没有超过既有前沿的结果都保存在轨迹事实中，不产生正增益。

$H=3$ 同时用于新假设成熟与活跃假设发展，使两个阶段使用相同的短视界观察单位。

## 5. 初始化

初始化建立八个可比较的起始假设：

1. 在虚拟根下持续生成初始算法，直到得到 $K=8$ 个有效根或预算耗尽；
2. 每个有效根建立一个 root hypothesis；
3. 令 $s_R=0$，按创建顺序为每个 root hypothesis 执行一个 $H=3$ 的 maturation block；
4. 根据初始化中的有效 Refine 边计算并冻结 $s_R$；
5. 八个 root hypotheses 全部进入 active set。

初始化失败和 Refine 失败都消耗对应 primary slot。预算在八个有效根形成前耗尽时，返回已经得到的最好有效算法。

## 6. 新假设的发现、成熟与竞争

### 6.1 Discovery

当前活跃假设完成一轮发展且全部停止推进后，触发一次 Discovery：

1. 选择 $R(z)$ 最高的 active hypothesis；
2. 以其前沿节点 $f(z)$ 为父节点执行一次 Explore；
3. 有效 child 建立新 hypothesis，状态设为 `maturing`；
4. 无效 Explore 不创建新 hypothesis，本轮 Discovery 结束。

### 6.2 Maturation

有效新 hypothesis 立即获得一个 $H=3$ 的 Refine block。成熟期间的所有有效后代归入该 hypothesis。完成后只依据当前 $Q(z)$ 参加竞争。

### 6.3 Competition

把成熟假设与当前 active set 合并，按 $R(z)$ 排序：

- 保留前 $K$ 个并设为 `active`；
- 其余假设设为 `reserve`；
- 被新假设替换的旧 active hypothesis 同样进入 `reserve`。

成熟 block 的 $g(z)$ 不改变准入结果，也不提供额外成熟预算。一个方向即使在低位有所进步，仍需达到竞争质量才能进入常规发展。

## 7. 活跃假设的发展调度

Development phase 由若干 sweep 构成。

### 7.1 全体 sweep

初始化完成后，以及每次 Discovery 与 Competition 完成后，当前全部 active hypotheses 构成首轮 eligible set：

$$
E_1=A_t.
$$

在 sweep 开始时按 $R(z)$ 冻结执行顺序，每个 eligible hypothesis 依次获得一个完整发展块。当前 sweep 中的新结果不改变尚未执行的顺序。

### 7.2 成功方向连续发展

一轮结束后，仅保留本轮 $g(z)>0$ 的假设：

$$
E_{j+1}=\{z\in E_j:g_j(z)>0\}.
$$

若 $E_{j+1}$ 非空，按最新 $R(z)$ 冻结下一轮顺序，并为这些假设各执行一个新 block。一个假设连续成功多少轮，就连续获得多少个发展块。

当某轮没有任何成功假设时，Development phase 结束并触发下一次 Discovery。没有成功的假设仍保持 active；新方向完成竞争后，它们会再次参加下一轮全体 sweep。

该调度让实际探索—利用节奏由本次运行产生：

- 竞争方向持续改善时，预算集中于发展；
- 少数方向改善时，连续发展集合自动收缩；
- 竞争方向普遍停滞时，Discovery 更频繁发生。

## 8. 完整运行协议

````text
Input:
    task, evaluator, LLM
    primary budget B = 1000
    initial and active capacity K = 8
    block horizon H = 3
    history length L = 8

Create a virtual root.
Generate valid roots until K roots exist or the budget is exhausted.
If the budget ends before K valid roots exist:
    return the best valid root, if one exists.

Create one root hypothesis for each root.
Set s_R = 0.
For each root hypothesis in creation order:
    run one Refine block of H primary slots;
    if the budget is exhausted:
        return the best valid algorithm.

Compute s_R from valid initialization Refine edges and freeze it.
Set all root hypotheses to active.
eligible = all active hypotheses.

While primary budget remains:
    # Development phase
    while eligible is non-empty and primary budget remains:
        freeze this sweep's order by R(z);
        successful = empty list;

        for hypothesis in the frozen order:
            if fewer than H slots remain:
                spend all remaining slots refining this hypothesis;
                return the best valid algorithm;

            run one Refine block for this hypothesis;
            if its block gain is positive:
                append it to successful;

        eligible = successful;

    # Discovery, maturation, competition
    if fewer than 1 + H slots remain:
        spend all remaining slots refining the highest-ranked active hypothesis;
        break;

    source = highest-ranked active hypothesis;
    run one Explore primary slot from its frontier node;

    if a valid Explore child is created:
        create a maturing hypothesis from that child;
        run one Refine block for the new hypothesis;
        keep the top K hypotheses from active set plus the new hypothesis;
        set every other hypothesis to reserve;

    eligible = all active hypotheses;

Return the valid Algorithm node with the highest true q;
break final ties by earlier creation.
````

## 9. 预算与终止

全局预算为 $B=1000$ 个 primary evaluator slots。以下动作各消耗一个 primary slot：

- 一次初始根候选；
- 一次 Explore 候选；
- 一个 Refine block 中的一次 Refine 候选。

所有阶段共享同一个 primary slot 计数器。新的 Discovery 只有在剩余预算足以完成一次 Explore 和一个 maturation block 时启动。Development 中剩余预算不足一个完整 block 时，把全部剩余 slots 用于当前执行顺序中的假设，然后终止。

在八个根都首次有效的情况下，初始化最少使用 $8+8H=32$ 个 slots。若一轮全体 Development 没有任何成功 block，且随后 Explore 有效，则一个“停滞—发现—成熟—重新比较”周期使用 $KH+1+H=28$ 个 slots。因而 V9.17 把数百个一次性 Explore proposal 收缩为数十个获得统一成熟机会的候选方向；实际数量随成功 block、无效候选和根初始化失败自适应变化。

最终输出由整棵树中真实 $q$ 最高的 Algorithm 节点决定，不受 hypothesis 当前状态影响。

## 10. 固定参数

| 参数 | 固定值 | 机制作用 |
| --- | ---: | --- |
| primary budget $B$ | 1000 | 正式搜索候选评价数 |
| initial roots $K$ | 8 | 初始方向数 |
| active capacity $K$ | 8 | 同时参加质量竞争的假设上限 |
| block horizon $H$ | 3 | 成熟与发展使用的短视界 |
| history length $L$ | 8 | Refine/Explore 提示中的最近形成事件数 |
| generation intents | Refine, Explore | 假设内部发展与新假设出生 |

V9.17 没有固定 Refine/Explore 概率。实际 Explore 份额、连续发展深度和有效活跃宽度由 block 响应在线形成。

## 11. 机制不变量

实现必须保持：

1. 每个有效 Algorithm 节点只有一个父节点和一个 hypothesis ID；
2. Refine child 继承 hypothesis，Explore child 新建 hypothesis；
3. $Q(z)$ 只由该 hypothesis 中真实评价过的节点计算；
4. hypothesis 竞争只读取 $Q(z)$，平局只按创建次序解决；
5. block gain 只等于 block 前后真实前沿之差；
6. block gain 只控制 active hypothesis 是否进入下一 sweep；
7. active hypotheses 数量不超过 $K$，每个 hypothesis 只有一个持久状态；
8. reserve hypothesis 保留全部树节点，但不再获得 primary slots；
9. 每个 primary slot 恰好属于初始化、Explore 或 Refine 候选；
10. repair 不改变 primary budget、sweep、hypothesis 或父节点分配状态；
11. 中断恢复后的下一次假设、父节点、意图和预算位置与不中断运行一致。

## 12. 过程记录

checkpoint 需要保存：

- 全部 Algorithm 节点、父子关系、hypothesis ID 与形成事件；
- 每个 hypothesis 的 $Q$、前沿节点、状态、投入和最近 block gain；
- 当前 active set、竞争线和 reserve；
- 当前 phase、sweep、eligible set、冻结顺序与 block 步数；
- $s_R$、全局 primary slot、随机状态和全局最好节点。

每次 Discovery 记录来源假设、Explore 出生质量和成熟后质量。每个 block 记录 $Q_{before}$、$Q_{after}$、$g$、三次父节点选择、有效结果数和下一 sweep 状态。每次 Competition 记录候选排名、当时竞争线和状态变化。

全局最好节点每次刷新时，把当时 primary slot 数、fitness、节点 ID 和完整程序追加进 `best_history.jsonl`，使任意预算处的最优程序可直接取用；`best_program.py` 始终保存当前全局最优程序。

## 13. 可证伪的机制预期

V9.17 的关键预期为：

1. maturation 后进入 active set 的新假设，应比 reserve 假设更常形成后续新前沿；
2. active hypothesis 的正 block gain，应提高其下一 block 再次推进的概率；
3. 不同任务和运行应形成不同的 Development 长度、Discovery 间隔和 Explore/Refine 实际比例；
4. 深发展任务应出现少量长成功序列，入口稀缺任务应出现较短 Development 与更多 Discovery；
5. 新假设成熟、质量竞争和成功续投的联合机制，应改善固定预算下的最差任务表现。[待验证]

若第 1 条不成立，operator-defined hypothesis 不是有效投资单位；若第 2 条不成立，block response 不能承担连续发展调度；若前两条成立但最终结果没有改善，固定 $K$、$H$ 或 Discovery 来源仍与任务搜索几何不匹配。

## 14. 核心含义

V9.17 把任务自适应限制在一个清晰闭环内：Explore 负责产生候选方向，统一 maturation 让其短期实现质量可见，竞争线保留已经兑现出高质量的方向，block response 决定这些竞争方向能连续发展多久，全体停滞重新打开 Discovery。

算法假设是可审计的轨迹投资单位，真实算法簇仍是待识别对象。V9.17 首先检验：在不声称已经理解簇边界的条件下，来源一致的轨迹段是否足以支持比固定探索率更合理的运行内预算节奏。
