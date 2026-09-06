# TraceAAD V10.4 完整机制设计

## 1. 核心变化

V10.4 保留 V10.3 的搜索树、父节点分配、donor 选择、算子采样和真实评价预算，只改变候选生成过程：形成历史恢复为 V9.16 式事件轨迹，并将一次候选生成拆为独立的算法设计与代码实现两次 LLM 调用。

一次扩展为：

```text
选择 parent、operator 与可选 donor
→ 基于完整设计上下文生成 Idea
→ 基于 Idea 生成 Code
→ 真实评价
→ 写入一个新节点并重新分配
```

每个成功候选使用两次 LLM 调用和一次 evaluator 调用。正式预算仍只按 evaluator 调用计数。

## 2. 形成事件轨迹

当前算法的形成路径取最近至多 8 条父子转变，按从旧到新的顺序展示。每条事件包含当时生成的 Idea、定性结果以及父子 fitness：

```text
[History i] Formation step
Idea: ...
Result: improve | regress | plateau (Fitness: parent -> child)
```

历史包含形成当前节点的最近事件。根节点没有父子形成事件，因此显示无历史。轨迹只进入 Idea 阶段，不进入 Code 阶段。

## 3. 以算法改进为中心的设计方向

- **Refine**：持续发展当前算法思想，从当前算法及其形成过程出发，提出最有希望使算法变得更好的发展。
- **Pivot**：理解当前算法及其形成过程，发现当前设计的重要不足或尚未兑现的机会，由此发展更强的算法思想。
- **Fuse**：理解当前算法与参考算法各自的设计优势，将其中有用的认识发展为一个统一且更好的算法思想。

这些方向用于帮助模型从不同视角进行算法设计，不规定必须保留、删除或禁止采用的机制。设计可以包含实现新算法所需的计算结构、公式、参数和程序细节。

## 4. 两阶段生成

### 4.1 Idea 阶段

Idea 阶段看到 Task Contract、当前算法及 fitness、完整形成事件轨迹、本次设计方向，以及 Fuse 时的参考算法。它只生成算法设计文本，输出上限为 1024 tokens；提示词不规定句数或字符数。

### 4.2 Code 阶段

Code 阶段看到 Task Contract、当前算法、第一阶段的完整 Idea，以及 Fuse 时的参考算法。它不再看到形成轨迹、算子名称或算子指令，只负责把已经提出的设计落实为完整可执行函数。节点的 `idea` 来自第一阶段，`code` 来自第二阶段。

## 5. 搜索与预算

以下机制与 V10.3 相同：

- 8 个有效初始化根；
- fitness-only ESS-Boltzmann 父节点概率；
- inverse-sqrt 父节点访问次数修正；
- Refine、Pivot、Fuse 均匀采样；
- Fuse 从跨谱系 top-5 候选中均匀选择 donor；
- 单父程序树和评价后立即重新分配；
- 1000 次真实 evaluator 调用预算。

Idea 为空或 Code 无法解析时不调用 evaluator。解析成功后的正式 evaluator 调用无论成功或失败都消耗一个 primary slot，与 V10.3 口径一致。

## 6. 恢复与记录

Idea 生成后立即把 parent、donor、operator、完整 Idea、Prompt 和响应写入 `tree_state.json` 的 `pending_idea`。若 Code 调用期间中断，恢复时直接继续该 Idea 的 Code 阶段，不重新选择状态，也不重新生成 Idea。

`llm_calls.jsonl` 按 `idea` / `code` 阶段记录完整 Prompt、响应、耗时及错误；`events.jsonl` 继续记录一次候选的解析、评价和入树结果。
