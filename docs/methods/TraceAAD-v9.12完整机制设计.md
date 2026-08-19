# TraceAAD V9.12：轨迹视图切换下的去锚探索

> V9.12 以 [V9.11](TraceAAD-v9.11完整机制设计.md) 为直接基线。设计依据是 [V9.11 机制诊断](../analysis/TraceAAD-V9.11机制诊断.md) 与[研究认识](../knowledge/研究认识.md)。本文是下一轮实现规范。

## 1. 核心判断

V9.11 已经让 Explore 在停滞时稳定发生，并让每个有效 Explore child 获得一次紧邻 Landing。当前主要瓶颈已经从“是否给探索机会”前移到“探索实际提出了什么方向”：Explore 经常改变静态机制宏簇，但即时质量很低，一次 Landing 虽经常改善 child，却很少恢复到 Explore 前父代。

这说明增加探索次数或盲目延长着陆都不够直接。若 Explore 没有进入高价值算法簇，更多发展预算只会更深地开发错误方向。V9.12 因此只改变生成侧的一个条件：

**Refine 继续读取父代改进来时路；Explore 暂时不读取显式来时路，只根据任务、当前算法和真实 fitness 提出替代方向；Explore child 的紧邻 Landing 再恢复读取包含这次结构变化的完整来时路。**

完整节律为：

```text
有全局进展
    -> 带父代来时路 Refine

连续 H=8 个响应无严格全局突破
    -> 不展示父代来时路的 Explore

Explore 形成有效 child
    -> 恢复父代来时路的紧邻 Refine Landing

Landing 完成
    -> 回到 V9.11 的常规选择与体制切换
```

一句话概括为：**忘记来时路以离开当前方向，恢复来时路以形成新方向。**

## 2. 直觉与反直觉

父代改进来时路已经显示出稳定的 Refine 单步价值。它告诉模型当前算法如何形成，哪些结构应被延续，以及下一处局部修改应与什么既有决策保持一致。因此它是一个有效的发展先验。

同一份信息对 Explore 未必仍是帮助。Explore 被要求改变核心决策原则，但提示同时反复展示旧方向如何成功形成，会把生成分布拉回旧方向附近。模型在语言上声称“探索”，代码上仍可能只做较大的同族改写。历史越清楚，局部延续越容易；这对 Refine 是优势，对跨簇提议可能成为设计固着。

V9.12 的直觉是：不同生成意图需要不同信息。它的反直觉是：**轨迹感知不等于始终展示更多轨迹；主动隐藏与当前决策冲突的历史，也是一种轨迹政策。**

V9.12 并不让 Explore 从零设计算法。当前完整代码已经包含历史形成的结果，任务与 fitness 仍然提供明确边界。去掉的是显式形成叙事，而不是当前算法事实。

## 3. 唯一新增机制：按意图切换轨迹视图

### 3.1 轨迹事实保持完整

V9.12 继续保存全部在线事实：程序、父子锚点、形成意图、声明 Idea、实际修改、真实评价、形成顺序以及 invalid、no-op、重复和祖先返回响应。

设锚点 $a$ 的最近 8 条父链形成事件为

$$
\tau_8(a)
=
\operatorname{Tail}_8(\tau(a)).
$$

V9.12 不删除或改写这些事实，只根据本轮生成意图构造不同的模型可见视图。

### 3.2 Refine 视图

普通 Develop 与 Landing 都使用 Refine。其生成条件为

$$
h_R(a)=\tau_8(a).
$$

模型读取：

1. 任务定义与执行契约；
2. 当前程序的真实 fitness；
3. 当前完整代码；
4. 最近 8 条父链形成事件，每条包含 `Intent + Idea + Compact Actual Change + Result`；
5. Refine 意图与统一输出契约。

Refine 负责延续当前核心原则、利用形成历史中的有效结构，并完成一次聚焦改进。上下文超限时仍从最早形成事件开始删除。

### 3.3 Explore 视图

Explore 的显式轨迹视图为空：

$$
h_E(a)=\varnothing.
$$

模型只读取：

1. 任务定义与执行契约；
2. 当前程序的真实 fitness；
3. 当前完整代码；
4. Explore 意图与统一输出契约。

Explore 提示要求模型识别当前算法所依赖的一个核心决策原则，并用不同原则、搜索结构或信息利用方式替换它。最终仍只输出一条简短 `Idea` 和一份完整 `Code`，不额外输出分析、反思、算法簇标签或证据。

Explore 不读取：

- 父代形成事件；
- 当前锚点已有子代尝试；
- 其他路线代码或结果；
- 全局 Idea Bank；
- 在线算法簇标签；
- 模型生成的失败总结或局限摘要。

因此 V9.12 不新增摘要器、judge、第二次模型调用或额外 token 阶段。视图切换是确定性的上下文选择，不是新的生成动作。

### 3.4 Landing 重新形成记忆

若 Explore 从 $a_p$ 形成有效 child $a_c$，形成事件

$$
e_E=\langle a_p,E,a_c,\mathrm{Idea},\mathrm{Change},\Delta q,\mathrm{Outcome},t\rangle
$$

立即进入 $\tau_8(a_c)$。紧邻 Landing 满足

$$
a_{t+1}=a_c,
\qquad
o_{t+1}=R,
\qquad
h_{t+1}=h_R(a_c).
$$

因此 Landing 能看到 Explore 实际改变了什么、结果如何，并在新形成的事实基础上继续修正。Explore 阶段对旧历史的去锚不会破坏新方向自己的来时路；一旦新方向产生，它立即重新成为可被后续生成利用的轨迹事实。

## 4. 保持不变的 V9.11 骨架

V9.12 除轨迹视图外，完整继承 V9.11。

### 4.1 初始化

1. 独立生成 8 个有效且代码互异的根；
2. 每个根执行一次带父代来时路的 Refine bootstrap；
3. 用有效 bootstrap 的一步有向质量变化绝对值中位数估计共享尺度 $s_0$；
4. 根与 bootstrap 的真实 evaluator 调用计入 1000 次正式预算。

### 4.2 常规路线与锚点选择

路线仍只表示根来源，不被解释为算法簇。路线优先级为

$$
S_t^{route}(r)
=
q_t^*(r)
+
\frac{s_0}{\sqrt{N_t(r)+1}}.
$$

在选中路线内，锚点优先级为

$$
S_t^{anchor}(a)
=
q(x(a))
+
\frac{s_0}{\sqrt{n_t(a)+1}}.
$$

分数仍只读取已经达到的质量和已经获得的生成机会，不加入趋势、成熟度、算法簇价值或长期信用。

### 4.3 停滞触发

记已完成正式搜索响应数为 $m$，最近一次严格全局突破和最近一次 Explore 的序号分别为 `last_progress_order` 与 `last_explore_order`。下一轮意图保持：

$$
o_{m+1}
=
\begin{cases}
R, & \mathrm{landing\_anchor}\neq\varnothing,\\
E, & m-\max(\mathrm{last\_progress\_order},\mathrm{last\_explore\_order})\ge 8
     \ \land\ B_{\mathrm{remain}}\ge 2,\\
R, & \text{otherwise}.
\end{cases}
$$

严格全局突破才重置进展时钟；plateau 和同分择简不重置。每次完整 Explore 响应都会重置 Explore 时钟，避免失败响应触发连续 Explore。

### 4.4 一次 Landing

有效 Explore child 仍固定获得一次紧邻 Refine。Landing 完成后无条件清除资格，无论它形成改进、持平、退步、invalid、no-op 或重复响应。Landing 不修改路线或锚点分数，也不产生长期特殊身份。

## 5. 为什么当前不延长 Landing

V9.11 中 Landing 经常改善 Explore child，却很少恢复到 Explore 前父代。这说明一次 Landing 不是充分条件，但尚不能推出“固定三步”或“固定五步”就是正确机制。

多步 Landing 需要回答何时停止。固定长度会继续用同一个 horizon 处理不同任务和算法簇；恢复父代才停止会在坏提议上消耗大量预算；只要局部改善就继续则可能长期发展一个上限较低的簇。三种规则都引入了新的分配判断，而当前最上游的问题仍是 Explore 是否提议了值得发展的方向。

V9.12 因此先提高替代方向的提议质量，同时保留一次 Landing 作为最小兑现机会。若去锚 Explore 能产生更好的 child 或更有发展性的 child，一次 Landing 的价值也会随之改变。只有在该生成条件下仍稳定出现“命中好方向但一步无法兑现”，才有理由设计新的 episode 终止规则。

## 6. 为什么不加入“失败与局限摘要”

“当前代码 + 已知失败或局限的压缩视图”是一个合理候选，但不是 V9.12 的首选。

第一，现有父链事实只记录已经发生的修改和评价，并不直接给出算法为何受限。把它压缩成“局限”需要模型推断或手工规则，摘要可能把错误解释重新写入生成条件。

第二，从当前锚点发出的 sibling failures 在受控单步实验中没有显示稳定的额外价值。把它们重新加入 Explore 会同时改变信息来源与压缩方式。

第三，V9.12 需要一个清楚的机制边界：显式来时路对 Refine 保留，对 Explore 隐藏。加入失败摘要后，无法区分收益来自去锚还是来自新型反思信息。

因此 V9.12 使用最小视图。未来若需要限制 Explore 重复失败，应把失败摘要作为独立生成机制，而不是本版的补丁。

## 7. 完整原子循环

```text
Initialize 8 code-unique roots.
Refine each root once with Parent Improvement Path.
Estimate the shared scale s0.

Set last_progress_order = 0.
Set last_explore_order = 0.
Set landing_anchor = null.
Set completed_search_responses = 0.

While real evaluator budget remains:
    If landing_anchor exists:
        Select landing_anchor.
        Set intent = Refine.
        Set history_view = Parent Improvement Path.
    Else:
        Select one route by q_best + s0 / sqrt(N + 1).
        Select one anchor in that route by q + s0 / sqrt(n + 1).

        If completed_search_responses
           - max(last_progress_order, last_explore_order) >= 8
           and at least 2 real evaluator calls remain:
            Set intent = Explore.
            Set history_view = Empty.
        Else:
            Set intent = Refine.
            Set history_view = Parent Improvement Path.

    Build Task + Current Fitness + Current Code + history_view + intent.
    Generate one Idea + Code response.
    Parse, evaluate or reuse, and record all facts.
    Increment completed_search_responses and set t to its new value.

    If a strict global best is formed:
        Set last_progress_order = t.

    If intent is Explore:
        Set last_explore_order = t.
        If a valid child anchor is formed:
            Set landing_anchor = that child.

    Else if this was a landing response:
        Clear landing_anchor.

Return the globally best unique program by the true objective.
```

每份模型响应仍只产生一个 `Idea + Code`，随后立即评价或复用并更新事实。V9.12 没有规划器、反思器、长期 rollout 或离线知识注入。

## 8. 预期搜索行为

### 8.1 Refine 行为应基本保持

初始化、普通 Develop 和 Landing 的锚点、历史视图、提示职责与 V9.11 相同。V9.12 不以牺牲已知有效的局部发展能力换取全局多样性。

### 8.2 Explore 的改变应发生在提议分布

去掉显式形成历史后，Explore 应更少复述父链 Idea 或继续沿最近修改做扩大版局部调整。期望变化是替代核心决策原则的概率和有效替代方向的概率提高，而不是单纯追求更大的代码 diff 或更高的静态换簇率。

### 8.3 轨迹在两阶段承担相反职责

Explore 前，轨迹通过停滞状态决定何时暂时忘记旧形成叙事；Explore 后，新形成事件立即进入父链，轨迹又用于稳定刚产生的方向。轨迹同时参与生成条件与计算分配，但不再被当作所有生成意图的统一模板。

### 8.4 任务差异由提议自然体现

V9.12 不设置 task-specific 的 Explore 比例、簇标签或 Landing 长度。任务定义、当前程序、真实 fitness 和在线突破节律共同形成任务条件；模型在这一条件下提出替代机制。不同任务的优势算法簇仍可能具有不同提议概率，V9.12 的目标是减少由旧轨迹叙事额外造成的概率偏置，而不是宣称消除模型本身的算法先验。

## 9. 主动删除的候选机制

V9.12 明确不加入：

- Explore 专用总结器、critic、reflection 或第二次模型调用；
- sibling failures、全局 Idea Bank、其他路线代码和语义检索；
- 在线算法簇分类、embedding、novelty reward 或多样性配额；
- task-specific 停滞窗口、探索比例和着陆长度；
- 固定三步或五步 Landing rollout；
- 恢复父代阈值、局部改善阈值和 episode 终止器；
- Thompson Sampling、后验、父链信用、延迟 pending credit；
- 对 V9.11 路线—锚点分数的同步修改。

这些对象有各自可能解决的问题，但同时加入会破坏 V9.12 的主要叙事：生成意图是否应决定模型看到哪一种轨迹视图。

## 10. 设计假设、预测与反证

### 10.1 设计假设

1. 父代形成历史对 Refine 是发展先验，对 Explore 可能是方向锚点。
2. 当前代码与任务已经足够约束一次结构性 Explore，不需要显式父链保证可执行性。
3. 更好的跨方向提议能提高后续一次 Landing 的有限预算价值。

这些是机制设计假设，不是 V9.11 中期数据已经证明的事实。

### 10.2 期望观察

- Refine 的有效率、即时改善率和轨迹深度大体保持 V9.11 水平；
- Explore 的静态宏簇切换率可以提高或保持，但更重要的是 Explore--Landing 事件更常接近或超过 Explore 前父代；
- Explore 或其后代对全局突破的贡献提高；
- OP 等当前困难任务若主要受历史锚定影响，应比单纯增加 Explore 次数更容易出现新的有效方向。

### 10.3 会否定或削弱本机制的观察

- Explore 只产生更大、更差、不可修复的改写，说明父链主要提供任务相关约束而非锚定；
- 宏簇切换增加，但 Landing 恢复与后续突破不变，说明新颖性不是当前瓶颈；
- Explore 行为几乎不变，说明当前代码本身已足以锚定模型，显式来时路不是主要中介；
- Explore child 更有潜力但仍普遍需要多步才能竞争，说明下一瓶颈确实转向发展 horizon；
- 任务分化保持原样，说明主要限制更可能来自 LLM 对优势算法簇的基础提议概率或任务评价几何。

## 11. 与 V9.11 的唯一协议差异

| 决策位置 | V9.11 | V9.12 |
| --- | --- | --- |
| Develop / Refine | 当前代码 + 最近 8 条父代来时路 | 不变 |
| Explore | 当前代码 + 最近 8 条父代来时路 | 当前代码，不展示父代来时路 |
| Landing / Refine | Explore child + 包含 Explore 事件的来时路 | 不变 |
| 触发与冷却 | 全局停滞 `H=8` | 不变 |
| 常规分配 | V9.7 路线—锚点选择 | 不变 |
| 着陆预算 | 一次紧邻 Refine | 不变 |
| 额外模型调用 | 无 | 无 |

因此 V9.12 不是新的分配器，也不是多步探索框架。它是 V9.11 上一个明确、单一的生成条件修改。

## 12. 实现不变量

1. 轨迹事实完整保存；Explore 只改变模型可见视图，不改变落盘事实。
2. Refine bootstrap、Develop 和 Landing 最多展示最近 8 条父链形成事件。
3. Explore 提示中不出现父链形成事件或由其生成的替代摘要。
4. Explore 仍看到完整当前代码、真实 fitness、任务契约和 Explore 指令。
5. Explore child 的形成事件必须进入紧邻 Landing 的 parent path。
6. 其余初始化、选择、停滞时钟、一次 Landing、评价、去重、checkpoint 和最终程序选择与 V9.11 一致。
7. 正式预算仍为 1000 次真实 evaluator 调用；模型响应数、token 和墙钟成本单独记录。
8. 协议标识必须明确区分 V9.11 与 V9.12，checkpoint 不跨版本恢复。

## 13. 首轮实验策略

V9.12 的实现可以在 V9.11 后半程继续运行时开始，不需要等待其 held-out 结果。开始新正式批次前只要求一个小预算 smoke，确认：

1. Explore 的 prompt 工件确实不包含父链事件；
2. Develop 与 Landing 仍包含正确锚点的父链；
3. Explore child 仍只获得一次 Landing；
4. checkpoint 恢复后轨迹视图与体制状态不漂移。

smoke 通过后直接运行四任务三重复的完整版本，不先展开 `H`、历史长度、Landing 长度或任务配置消融。V9.11 继续完成，用作最终完整版本比较与过程参照；V9.12 的主结果仍需等待三重复搜索和 held-out 全部完成后进入权威结果页。

首轮过程分析只关注三个大问题：

1. Explore 是否提出不同且有效的替代方向；
2. Explore child 经一次 Landing 后是否更接近已有竞争前沿；
3. 任务分化是否从“频繁换簇但不兑现”转向更多实际突破。

## 14. 两句话方法说明

TraceAAD V9.12 认为同一条改进来时路对不同生成意图具有不同作用：它帮助 Refine 延续并修复当前方向，也可能把 Explore 锚定在已经形成的机制附近。方法在停滞时暂时隐藏显式来时路以提出替代算法，随后让新方向立即恢复由真实形成事件条件化的 Landing，从而以“去锚探索—重新成轨”的最小循环连接算法簇迁移与局部发展。
