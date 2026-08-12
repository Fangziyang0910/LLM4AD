# TraceAAD V9 与 V9.5 生成接口审计

> 审计日期：2026-08-12。V9 正式批次为 `20260807_123753`，V9.5 正式批次为
> `20260811_171029`。范围为 TSP、CVRP、OP 的已完成运行；OBP 未纳入 prompt 抽样。
>
> 本文比较生成接口和伴随生成行为。它不是 prompt 的因果消融，也不把联合版本的
> 最终差异归因于某一个组件。
>
> 2026-08-12 记录：本文原含 V9/V9.5 预算计数口径的比较与"一级混杂因素"论述，该议题
> 已裁定关闭——预算统一为固定 1000 次评价即为公平口径，相关论述已从本文及各文档删除，
> 不再讨论。

## 结论

V9 是当前应优先解释的强 TraceAAD 参照。单版本与四个固定外部基线同场时，V9 的平均名次是
**1.867**，同场 MCTS-AHD 为 **1.933**；此前把 V9 写成 1.933 是把两者数值对调了。用户提供的
V9.5 暂定平均名次为 2.600，但 TSP/CVRP 各缺一个正式 repeat，不能视为完成的三重复排名。

本次审计得到两个主要事实：

1. **V9.5 的主要额外输入负担来自 correction diff，不是核心指令。** 随机 prompt 样本中，
   V9 历史部分约 0.99--1.03K token，V9.5 为 3.13--3.42K；两者指令部分仅约 122 对
   139 token。V9.5 平均历史事件数反而略少，因此差异主要是每条 actual diff 的表达长度。
2. **“V9.5 prompt 更重”目前是可信机制风险，不是已证实性能原因。** V9.5 在 TSP/CVRP 的
   parent improvement rate 高于 V9，在 OP 则显著更低；其影响明显随任务变化。Anchor、选择器、
   operator、一次生成数量均同时变化，现有观察不能识别 prompt 的净效应。

因此，下一步应先做固定 anchor 的配对生成接口实验，再做 allocation 消融；不应根据单个任务
继续加入 adaptive `s`、formation/direct 配额或 task-specific rescue。

## 1. 基准参照

当前正式结果支持以下判断：

- V9 是现有 TraceAAD 版本中最强的完整系统参照，而不是 V9.3。
- V9 的优势主要来自 OP 与 OBP capacity=100；TSP/CVRP 并非全面领先。
- V9 的整版结果不能证明 matched history、四 operator 或 UCT 中任一单项有效。
- V9.5 的 TSP 是正信号，OP 明确弱于 V9；CVRP/OBP 并未形成一致升级。

已完成 held-out 均值如下。V9 为三重复；V9.5 的 TSP/CVRP 仅两重复，均为暂定值。

| Task | Scale | V9 | V9.5 | 当前方向 |
| --- | --- | ---: | ---: | --- |
| TSP ↓ | 50 / 100 / 200 | 6.362 / 8.833 / 12.364 | **5.907 / 8.273 / 11.980** | V9.5 更好 |
| CVRP ↓ | 50 / 100 / 200 | **9.083 / 15.498** / 28.107 | 9.432 / 15.828 / **27.966** | V9 多数更好 |
| OP ↑ | 50 / 100 / 200 | **15.162 / 30.632 / 54.905** | 14.936 / 29.811 / 52.891 | V9 全部更好 |

这些是联合系统的描述性结果，不是生成接口的因果证据。

## 2. 两个版本给模型构造了什么问题

两版均是一个 user message、无独立 system prompt，并要求输出 `Idea + Full Code`。主要差异如下。

| 维度 | V9 | V9.5 |
| --- | --- | --- |
| 当前状态 | 当前完整程序与 fitness | 当前 executable anchor 与 fitness |
| Formation | 最近至多 8 条；Idea、结果、fitness、breakthrough、change ratio、LOC | 与 direct 共用 8 条总预算；Idea、actual diff、结果与 fitness |
| Direct trials | 最多 8 条代表分支；立即结果、subtree best、后续深度与 Idea | exact-state direct corrections；outcome coverage、去重、近期补足；actual diff |
| 跨路线输入 | synthesize/transfer 时给另一 root 的代表程序及 formation | 无 reference 程序 |
| 生成方向 | `ideate/refine/synthesize/transfer` 四种显式 operator | 一个固定的隐式历史条件生成指令 |
| 一次选择 | 同一 anchor 生成两个 sibling | 生成一个 candidate 后全局重选 anchor |
| 修改表示 | 历史中使用简短 Idea/change summary | 历史中使用最多 1200 字符的单行 actual diff excerpt |

V9 的决策问题是“在一个指定 operator 下，结合主程序历史，必要时参考另一程序，生成两个
sibling”；V9.5 的问题是“阅读当前代码与 correction evidence，自行决定一个 coherent
modification”。二者不只差一段 history 文本。

### 2.1 信息类型

按模型实际可利用的信息，而非树拓扑分类：

| 信息 | V9 | V9.5 |
| --- | --- | --- |
| 当前程序做什么 | 完整代码 | 完整代码 |
| 以前意图改什么 | 简短 implemented Idea | declared Idea |
| 实际改了什么 | change ratio、LOC；无 diff | actual parent-child diff excerpt |
| 改完结果怎样 | outcome、fitness、breakthrough；direct 还有 subtree best | outcome、parent/child fitness |
| 哪些尝试无效 | 不作为正常历史 correction 展示 | invalid 类别与精简 failure feedback |
| 另一候选代码 | 双轨迹 operator 约半数 prompt 含 reference program | 无 |
| reflection/LLM summary | 无 | 无 |

V9.5 增加的核心信息不是更多 event，而是更细的实际代码变化与无效反馈；V9 则保留更强的
operator 指向和偶发的另一完整程序。

## 3. Prompt token 审计

### 3.1 方法

- 对 TSP、CVRP、OP，每个 task × version 从全部已完成正式 run 中按固定随机种子抽取 20 个
  search prompt，共 120 个；不选择 best lineage。
- V9.5 prompt 从 `llm_calls.jsonl` 原样读取。V9 prompt 使用 `decisions.jsonl` 保存的 parent、
  operator、reference 及 edge-id snapshot，在当时树状态上重建。
- 使用实验模型的真实 tokenizer 与完整 chat template 重新计数，不使用字符数近似。
- 60 个 V9 重建样本的 raw-token 绝对误差均值为 **0**；chat template 相对 raw prompt 固定
  增加 12 token，说明重建与计数一致。

### 3.2 随机样本分解

下表均为每格 20 个 prompt 的 token 均值；Total 为 raw user-prompt token。完整 chat template 在
每条 prompt 上固定再增加 12 token。

| Task | Version | Total | Current code | History | Instruction | Operator | Reference code/history | 其他固定部分 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TSP | V9 | 4189 | 2592 | 989 | 122 | 32 | 221 | 234 |
| TSP | V9.5 | **5572** | 2189 | **3131** | 139 | 0 | 0 | 113 |
| CVRP | V9 | 4352 | 2336 | 995 | 122 | 33 | 528 | 338 |
| CVRP | V9.5 | **6646** | 2914 | **3421** | 139 | 0 | 0 | 171 |
| OP | V9 | 3983 | 1882 | 1030 | 122 | 33 | 594 | 322 |
| OP | V9.5 | **4961** | 1363 | **3299** | 139 | 0 | 0 | 160 |

“其他固定部分”包括 task、direction hint、target signature 等。不同版本抽到的 current program
并不相同，因此 current-code token 差不能解释为模板开销，也不能用于比较程序复杂度。

全量日志中的平均 raw prompt token 同样显示 V9.5 更长：

| Task | V9 | V9.5 | V9.5 / V9 |
| --- | ---: | ---: | ---: |
| TSP | 3737 | 5410 | 1.45× |
| CVRP | 4414 | 6027 | 1.37× |
| OP | 3660 | 4669 | 1.28× |

### 3.3 Event 数量与表达密度

| Task | V9 mean history events | V9.5 mean history events | V9.5 direct / formation |
| --- | ---: | ---: | ---: |
| TSP | 8.96 | 7.88 | 0.40 / 7.49 |
| CVRP | 8.88 | 7.94 | 0.13 / 7.81 |
| OP | 8.78 | 7.90 | 4.41 / 3.49 |

V9 的 prompt 约 49% 含 reference program，但抽样期 reference formation 平均为 0，说明参考项
通常来自尚未发展的 root program。V9.5 不含 reference program。

V9.5 用更少的 event 消耗了约三倍的 history token。直接证据指向 **actual diff 的表达密度**，
而非 EvidenceBuilder 塞入更多事件。真实样本中还可见长 diff 被压成单行并附
`[diff truncated]`；它提供了真实修改，却同时要求模型从高密度代码差异中恢复语义重点。

### 3.4 Instruction 是否过度复杂

V9.5 instruction 比 V9 基础 instruction 平均仅多约 17 token。它确实同时要求 preserve useful
mechanisms、consider history、允许 materially different revisit，但没有要求输出 reasoning、
evidence analysis 或 operator label。当前证据不支持把主要负担归因于 instruction 文案；优先应
审计每条 correction 的信息密度和必要性。

### 3.5 真实 prompt 个案核查

审计同时打开了随机样本的完整文本，而不只看 token 表。两个可复查的 TSP 样本是：

| Version | Run / sample | SHA-256 | Raw tokens | 看到的结构 |
| --- | --- | --- | ---: | --- |
| V9 | `tsp_rep3 / 887` | `08a38290...f52b6a` | 6449 | 8 条简洁 formation、无 direct；current code；单一 operator direction |
| V9.5 | `tsp_rep2 / 163` | `fb4b4164...01c339` | 5996 | current code；多条 actual diff correction；统一 generation instruction |

V9 样本的历史以“fitness 变化 + 一段 Idea + change ratio/LOC”呈现，模型可以快速看到每步语义，
但无法核对实际实现。V9.5 样本保留真实代码变化，部分 excerpt 很长或出现
`[diff truncated]`，且代码注释也会进入 diff；它更忠实，但语义重点更分散。两个样本都没有
发现互相矛盾的输出合同。完整 120 条原文通过脚本 `--dump-prompts` 本地导出，不进入 Git，避免
把正式原始工件复制进文档。

## 4. Anchor 选择规则

### 4.1 V9

V9 从 10 个 roots 建树，使用全局 directed-fitness midrank percentile 作为质量尺度。选择从虚拟
root 开始递归进行 UCT：在每个程序处比较继续进入已有 child 与从当前程序再打开一个新 child
batch。已有 child 使用 subtree-best quality，扩展选项使用 anchor quality prior 与历次 batch
subtree-best reward；exploration 随剩余预算衰减。选定 anchor 后，从四个 operator 中抽取一个，
synthesize/transfer 再选择其他 root 的代表程序，并一次生成两个 sibling。

### 4.2 V9.5

V9.5 初始化 8 个 roots，并对每个 root 做一次 bootstrap。optimism scale 是 bootstrap 有效
绝对变化的中位数。之后对每个 valid AnchorState 计算：

\[
S(a)=q(a)+\frac{s}{\sqrt{n(a)+1}}
\]

每次选全局最高分 state，依次用更小的 `n`、更早创建和更小 state id 确定性打破并列；生成一个
candidate 后重新比较所有 anchors。它没有 operator、reference 或 subtree value backup。

这意味着 V9 与 V9.5 同时改变了选择空间、价值定义、探索项、operator、cross-program context、
sibling 数和重选频率。仅看最终分数无法判断哪一个差异起主要作用。

## 5. 生成结果的伴随统计

只统计正式批次中已完成 run；parent improvement 仅在有效 parent-child transition 上计算。

| Task | Version | invalid | parent improve | line change mean / median | unique valid | parse failure / response |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TSP | V9 | 3.87% | 10.30% | 0.720 / 0.758 | 99.05% | 0.30% |
| TSP | V9.5 | 10.42% | **23.61%** | 0.588 / 0.641 | 94.84% | 0% |
| CVRP | V9 | 4.88% | 12.60% | 0.672 / 0.719 | 98.51% | 0.64% |
| CVRP | V9.5 | **0.76%** | **40.88%** | 0.246 / 0.091 | 96.00% | 0% |
| OP | V9 | 4.07% | **9.72%** | 0.686 / 0.718 | 99.23% | 0.20% |
| OP | V9.5 | 3.69% | 1.65% | 0.576 / 0.617 | 96.09% | 0% |

可支持的描述是：V9.5 的生成在 CVRP 上更局部且更常立即改善，在 TSP 上也提高了一步改善率，
但在 OP 上明显失效；V9.5 的有效程序唯一率略低。不能据此说 actual diff 必然有利或有害：
selection 改变了 parent 分布，V9 的两个 sibling 共享一次选择，而 V9.5 每次重选，operator 与
reference 也不同。

这组结果反驳了一个过强解释：V9.5 prompt 并非在所有任务上都让模型无法修改。更合理的待验证
风险是：它用更多 token 提供了更精细但未必更易用的 evidence，净效应依赖 anchor 与任务。

## 6. 拓扑解释边界

Tree、root、clade 和 depth 是可靠的 provenance：它们说明程序如何生成、哪些修改按何种父子关系
发生。它们不是算法语义类别：

\[
\text{tree topology}\ne\text{semantic search space}
\]

不同 root 可能生成近似算法；同一 root 的相邻程序也可能发生大幅语义重构。因此，旧复盘中的
“deep within-clade continuation 是 TSP 成功机制”应收缩为：**TSP 的优秀程序具有较长的连续
修改历史。** 长历史是否带来成功、同一 clade 是否提供独特价值，当前均没有受控证据。

后续仍可用 lineage 还原 modification facts、计算形成深度与定位真实 parent；不能用 clade
share 直接替代语义多样性、路线覆盖或 causal mechanism。

## 7. 当前支持与不支持的判断

### 已支持

- V9 是当前更强的完整版本参照；V9.5 不是全面升级。
- V9.5 的 prompt 更长，增长主要来自 actual diff evidence。
- V9.5 instruction 本身只占很小增量。
- V9.5 的 generation/allocation 界面更简洁可分解，但“设计清楚”不等于“模型更易使用”。
- optimism 活跃程度与最终质量没有跨任务单调关系，继续按 changed-rate 调 `s` 缺乏依据。
- V9.5 已有实验不能按共同 1000 evaluator budget 表述。

### 尚不支持

- prompt 更长导致 V9.5 总体弱于 V9；
- actual diff 比 Idea + outcome 更好或更差；
- V9 的 operator portfolio 或双程序输入是优势来源；
- V9.5 allocation 是主要退化原因；
- long lineage 或 same clade 是 TSP 成功原因；
- 为 OP/CVRP 定制 context 或 allocation 会提高通用方法。

## 8. 后续调研与实验顺序

### 第一优先：固定 anchor 的配对生成接口实验

在同一批真实 anchors 上固定 task、current code、输出预算、temperature 和随机种子，仅改变输入：

1. `Current Code Only`；
2. V9 式 concise `Idea + outcome` history；
3. V9.5 式 `Idea + actual diff + outcome` correction evidence。

每个条件生成一个 candidate，并用同一 evaluator 测 valid、parent improvement、\(\Delta q\)、
duplicate 与 response token。该实验直接回答“历史是否有用”以及“细粒度 diff 是否增加可用信息”，
比先重跑完整搜索更能识别 generation interface。

若预算有限，可先在 TSP/CVRP/OP 分层抽取高、中、低质量 anchor 各若干，配对测试。结论应按
task 与 anchor quality 分层报告，同时给总体效应；不能只挑 best lineage。

### 第二优先：固定生成接口的 allocation 消融

选定同一 generation interface 后，再比较：

- V9 搜索选择；
- V9.5 `q+s/sqrt(n+1)`；
- 最小 pure-`q` 基线。

统一 roots、一次生成数量、evaluator budget 和 best selection，避免将 operator/sibling/reference
差异重新带入 allocation 结论。

在这些识别实验完成前，不在完整搜索中增加 adaptive `s`、critical-`s` controller、
formation/direct quota、regression credit、OP-specific rescue、critic 或 operator 调度。固定 anchor 的
生成接口识别可以另外配对比较 generic、`Refine` 和 `Explore` 简短意图，用来判断“明确
当次修改类型”是否能帮助 27B 模型利用历史。详细设计见
[RQ-003](../research/RQ-003-轨迹上下文与搜索评分.md)。

## 复现入口

- 审计脚本：`experiments/analysis/analyze_v9_v95_generation_interface.py`
- 聚合结果：`docs/analysis/traceaad_v9_v95_generation_interface/summary.json`
- 120 个随机样本的 token 分解：
  `docs/analysis/traceaad_v9_v95_generation_interface/sample_prompt_metrics.csv`
- 完整 prompt 可用脚本的 `--dump-prompts` 参数从本地原始工件重新导出；原始 prompt 不进入 Git。

脚本只读取正式工件，不修改运行状态。随机种子为 9509，每个 task × version 抽样 20 条。
