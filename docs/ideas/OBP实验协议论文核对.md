# Online Bin Packing 实验协议论文核对

## 核心结论

Online Bin Packing（OBP/BPO）在 LLM 驱动自动算法设计中的共同实验范式是：在一个预先固定的小型适应度数据集上反复评价候选 heuristic，搜索结束后将所得 heuristic 原样应用到独立测试集，并用不同 item 数和 capacity 检验泛化。这里的“训练”不是训练一个神经网络，而是候选程序搜索期间使用适应度数据集 \(D\)。

“训练只用 \(C=100\)，测试再外推到 \(C=500\)”是 EoH、FunSearch 以及 HSEvo 所沿用的早期协议，但并非整个领域当前统一遵守的协议。MCTS-AHD 明确指出，五个 \(5k,C=100\) 实例的旧协议经常产生在 \(C=500\) 上完全失效的 heuristic，因此把 \(1k/5k \times C=100/500\) 四个尺度都纳入搜索期数据集 \(D\)。CALM 使用与 MCTS-AHD 相同的训练和测试数据；PathWise 也使用四个训练实例，并将上述四个尺度标为 in-domain。

因此，需要先确定要回答的科学问题：

1. 如果研究问题是“只见过 \(C=100\) 的 heuristic 能否跨 capacity 泛化”，则不应把 \(C=500\) 加入训练；某个 run 在 \(C=500\) 上灾难性失败就是需要保留和解释的 OOD 结果。
2. 如果研究问题是“在 MCTS-AHD、CALM、PathWise 的近期标准协议下比较搜索方法”，则搜索期数据集应包含 \(C=500\)。这不会改变 OBP task，只是改变训练分布及 ID/OOD 的定义。
3. 两种协议都具有论文依据，但结果不能混为同一口径。尤其不能用 EoH 式单容量训练的结果，直接与 MCTS-AHD/PathWise 式多容量训练结果比较后，将差异完全归因于搜索方法。

## 各论文与原始实现

### EoH：固定五个 \(5k,C=100\) 实例，跨规模、跨 capacity 测试

EoH 在 heuristic evolution 期间使用五个 Weibull 实例，每个实例有 5,000 个 items，capacity 为 100；fitness 是五个实例上 \(lb/n\) 的平均值（`../papers/EoH/4-experiment.tex:5-7`）。因此 EoH 的搜索期数据集不是单个实例，而是五个固定实例，但只覆盖一个规模和一个 capacity。

搜索完成后，EoH 将 best heuristic 测试在六个设置：

- item 数：1k、5k、10k；
- capacity：100、500；
- 每个设置五个独立 Weibull 实例。

论文正文明确写出上述范围和每组五个实例（`../papers/EoH/4-experiment.tex:65-66`），Table 2 列出六组结果（`../papers/EoH/4-experiment.tex:71-86`）。附录进一步将 \((5k,C=100)\) 称为 evolution 中使用的 training distribution，并把其他 capacity 的结果解释为 generalization（`../papers/EoH/7-appendix.tex:271-278`）。

这直接支持“固定 \(C=100\) 训练，再测试 \(C=500\)”作为一个成立的科研协议。论文自身也展示了该协议可能暴露严重过拟合：EoC 和 EoH-e1 在 \(C=500\) 上出现很大 gap，而完整 EoH 保持较好泛化（`../papers/EoH/4-experiment.tex:214-228`）。

EoH 的主结果表展示的是所得 heuristic 在每组五个测试实例上的平均 gap；正文没有像 MCTS-AHD 那样清楚说明主表是否对三次独立搜索取均值。其若干消融和 LLM 对比实验明确重复三次（`../papers/EoH/5-discussion.tex:19-19,45-45`），因此不宜反向推断 EoH 主表的完整重复聚合口径。

### HSEvo：沿用 EoH 式单容量适应度集

HSEvo 的附录把 BPO 搜索期数据定义为五个随机生成的 Weibull \(5k,C=100\) 实例，并在五个实例上取平均目标（`../papers/HSEvo/appendix.tex:113-120`）。参数表再次写明 BPO problem size 为 5,000、capacity 为 100（`../papers/HSEvo/appendix.tex:164-175`）。论文实验使用三次 independent runs（`../papers/HSEvo/aaai25.tex:405-416`）。

作者原始代码与论文一致：

- Weibull shape=3、scale=45、item 最大值 100、bin capacity=100（`../reference_code/HSEvo/problems/bpp_online/gen_inst.py:5-10`）；
- 分别生成五个 train、五个 validation、五个 5k test、五个 10k test 和一个 100k test（同文件 `:54-70`）；
- evaluator 在 `train` 或 `val` 模式只读取 `weibull_5k_{mood}.pickle`（`../reference_code/HSEvo/problems/bpp_online/eval.py:91-109`）。

这里没有把 \(C=500\) 放入搜索期训练或验证。HSEvo 论文主要关注固定 BPO benchmark 上的搜索表现与多样性，并没有像 EoH、MCTS-AHD 那样建立六尺度测试主表。

### ReEvo 原始仓库中的 online BPP：同样是 \(5k,C=100\)

ReEvo 论文的主要应用是 GLS/ACO/GA 等任务，online BPP 不是其正文主结果的核心任务；不过作者原始仓库提供了 online BPP evaluator。该实现与早期 EoH/HSEvo 协议一致：

- capacity 固定为 100；
- train、validation、5k test、10k test 均为五个实例，另有一个 100k test；
- 搜索 evaluator 只读取 `weibull_5k_train.pickle` 或 `weibull_5k_val.pickle`。

证据见 `../reference_code/ReEvo/problems/bpp_online/gen_inst.py:5-10,54-70` 和 `../reference_code/ReEvo/problems/bpp_online/eval.py:91-109`。因此将 ReEvo 原始 online-BPP 代码视为 \(C=100\)-only 协议是有依据的，但不应把它表述成 ReEvo 论文提出的核心 OBP 实验结论。

### MCTS-AHD：明确否定单容量协议作为自身主协议

MCTS-AHD 对旧协议作了非常明确的修正。论文先说明旧 baselines 使用五个 \(5k,C=100\) Weibull 实例，随后指出这种设置经常导致 heuristic 在其他尺度“completely fail”，并以 \(5k,C=500\) 为例；因此 MCTS-AHD 使用 varying-scale evaluation dataset \(D\)（`../papers/MCTS-AHD/icml2025.tex:628-635`）。

MCTS-AHD 的搜索期 \(D\) 只有四个实例，但覆盖四个尺度：

- 一个 \(1k,C=100\)；
- 一个 \(1k,C=500\)；
- 一个 \(5k,C=100\)；
- 一个 \(5k,C=500\)。

论文 Table 9 给出这一组成（`../papers/MCTS-AHD/icml2025.tex:639-656`）。正文也概括为“四个具有不同尺度的 Weibull 实例”（同文件 `:358-360`）。

测试集仍然是六个设置 \(1k/5k/10k \times C=100/500\)，每个设置五个 Weibull 实例。表中前四个设置被下划线标记为包含在 \(D\) 中的 in-domain scales，10k 两组才是未在 \(D\) 中出现的尺度（同文件 `:361-377`）。

原始代码与论文完全对应：

- generator 为 \(1k/5k \times C=100/500\) 各创建一个 train instance（`../reference_code/MCTS-AHD-master/problems/bpp_online/gen_inst.py:55-75`）；
- 为同四个尺度各创建五个 validation instances，并为六个测试尺度各创建五个 test instances（同文件 `:76-85`）；
- evaluator 的 `train`/`val` 模式都遍历 `1k,5k` 和后缀 `1,2`，即四个数据文件（`../reference_code/MCTS-AHD-master/problems/bpp_online/eval.py:91-120`）。

因此，对 MCTS-AHD 而言，\(C=500\) 不是只在测试阶段出现的 OOD capacity，而是搜索期数据集 \(D\) 的组成部分。代码还提供同尺度 validation 文件，但论文主协议只把四个实例定义为 \(D\)，没有给出基于 validation 选择最终 heuristic 的独立正式步骤，不能把“代码存在 val 文件”扩写成论文已经采用验证集筛选。

MCTS-AHD 对每种方法进行三次独立搜索，并报告三次的平均 gap（同论文 `:320-325,364-364`）。OBP 的搜索预算为 2,000 次 heuristic evaluations，而其他任务通常为 1,000（同论文 `:630-632`）。

### CALM：使用 MCTS-AHD 的多容量训练/测试数据

CALM 明确声明：

- 与所有 LLM-based baselines 对齐训练数据、seed 和相近预算（`../papers/CALM/main.tex:289-289`）；
- OBP 表使用与 MCTS-AHD 相同的 training 和 test datasets；
- 结果对三次 runs 取平均；
- \(1k/5k \times C=100/500\) 四组都以下划线标成与训练 scale 匹配，10k 两组才是不同于训练的 scale（同文件 `:293-321`）。

因此 CALM 延续的是 MCTS-AHD 的多容量搜索协议，而不是 EoH 的 \(C=100\)-only 协议。

### PathWise：四个多容量训练实例，六尺度测试，每组十个实例

PathWise 的 Table 7 将 Online BPP 的训练数据写为 sizes \(\{1k,5k\}\)、共四个 instances；测试 sizes 为 \(\{1k,5k,10k\}\)、每个 test setting 十个 instances（`../papers/PathWise/example_paper.tex:1200-1226`）。论文另行说明 online BPP 的 capacity 根据 evaluation setting 取 100 或 500（同文件 `:1074-1075`）。

其结果表消除了“这四个训练实例是否包含 \(C=500\)”的歧义：\(1k/5k \times C=100/500\) 四个测试尺度全部被标为 in-domain，只有两个 10k 设置是 OOD size；每个测试设置包含十个 Weibull 实例（同文件 `:1350-1377`）。作者原始仓库中的数据文件也正好是四个 train pickles：`weibull_1k_train1/2` 和 `weibull_5k_train1/2`，分别对应 \(C=100/500\)。

PathWise 对每个 LLM-based AHD 方法报告三次独立 run 的均值，统一限制为每个 task 500 次 heuristic evaluations，并对每个 heuristic 在训练集上的执行设置 60 秒上限（同文件 `:797-805`）。

## 协议对照

| 论文/实现 | 搜索期实例 | 测试设置 | 每个测试设置实例数 | \(C=500\) 是否进入搜索期 | 搜索重复及主报告 |
|---|---|---|---:|---|---|
| EoH | 5 个 \(5k,C=100\) | \(1k/5k/10k \times C=100/500\) | 5 | 否，属于跨 capacity OOD | 主表重复聚合口径未明确；部分消融为 3 次 |
| HSEvo | 5 个 \(5k,C=100\) | 论文主 BPO 实验聚焦固定 benchmark | 未建立六尺度主表 | 否 | 3 次 independent runs |
| ReEvo 原始 online-BPP 代码 | 5 个 \(5k,C=100\) | 5k/10k/100k，代码中均为 \(C=100\) | 5/5/1 | 否 | online BPP 非 ReEvo 论文核心主表 |
| MCTS-AHD | 4 个：\(1k/5k \times C=100/500\) 各 1 | \(1k/5k/10k \times C=100/500\) | 5 | 是，四个前置尺度为 ID | 3 次均值 |
| CALM | 与 MCTS-AHD 相同 | 与 MCTS-AHD 相同 | 同 MCTS-AHD | 是 | 3 次均值 |
| PathWise | 4 个：\(1k/5k \times C=100/500\) 各 1 | \(1k/5k/10k \times C=100/500\) | 10 | 是，四个前置尺度为 ID | 3 次均值，500 evaluations |

## 对当前研究判断的含义

领域共识存在于实验结构，而不在单一 capacity 选择：

- 固定 task 定义、heuristic 接口、实例生成分布和搜索期数据，不为每个候选随机重采样；
- 搜索数据与测试数据分离；
- 用未参与搜索的更大规模测试集检验 size generalization；
- LLM 搜索具有明显随机性，近期工作通常做三次独立搜索并报告均值；
- 测试结果按一组独立 instances 聚合，而不是只看单个实例。

争议点是 capacity 是否属于需要外推的 OOD 维度。EoH 把 \(C=500\) 当作 OOD，用它证明或否证跨 capacity 泛化；MCTS-AHD 则根据已有失败证据把 \(C=500\) 纳入 \(D\)，只把 10k 当作 OOD size。两者对应不同、都可成立的科学问题。

因此，某个仅在 \(C=100\) 上搜索的 TraceAAD run 到 \(C=500\) 后“一件物品一个箱子”，不能归因于测试实例太少：三个 \(C=500\) 规模都达到结构性最坏行为，本身就是强机制证据。也不应立刻把它改成工程上的多容量验证问题。首先应给当前实验定性：

- 若声明采用 EoH 式单容量外推协议，该失败是合法且重要的泛化失败，正式三重复结果必须保留；
- 若声明复现 MCTS-AHD/CALM/PathWise 的近期 OBP 协议，则当前 \(C=100\)-only 搜索配置没有对齐论文，应在下一批正式实验中将四个尺度纳入 \(D\)，但不应回头修改或删除已经完成的 run；
- 若要同时回答搜索能力和外推能力，最清晰的做法是预注册两套实验：`single-capacity` 与 `multi-scale`。两套结果分别报告，不能把前者失败 run 从均值中事后排除。

## 当前采用的协议

2026-07-24 决定后，仓库的权威 `online_bin_packing` task 采用
MCTS-AHD/CALM/PathWise 式多容量协议：搜索集为固定的
\(1k/5k \times C=100/500\) 四个实例，测试集使用不同的固定 item
序列覆盖六个配置，每个配置五个实例，只有两个 10k 配置属于 OOD。
此前所有 \(5k,C=100\)-only 搜索结果作为旧协议历史证据保留；新协议
比较需要所有方法重新搜索，不复用旧 heuristic。
