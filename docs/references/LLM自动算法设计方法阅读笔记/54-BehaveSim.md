# BehaveSim

- 论文：*Rethinking Code Similarity for Automated Algorithm Design with LLMs*
- 本地论文：`../../../../papers/Rethinking_Code_Similarity_for_Automated_Algorithm_Design/paper.pdf`
- 官方实现：`../../../../reference_code/BehaveSim/`
- 会议：ICLR 2026

## 1. 测量对象

BehaveSim 测量两个算法在同一问题实例和同一起点上的求解行为。算法执行时产生的中间解序列称为 problem-solving trajectory（PSTraj）：

$$
T(A,I,s)=(x_0,x_1,\ldots,x_m).
$$

对于排列、类别和序数形式的中间解，单步距离使用编辑距离，并除以该问题下可能的最大编辑距离。论文的 AAD 开源实现将其操作化为除以两份当前部分解的较长长度。两条轨迹之间先用 DTW 对齐，再除以较短轨迹长度：

$$
d(x_i,y_j)=\frac{d_{\mathrm{edit}}(x_i,y_j)}{d_{\max}},
$$

$$
D_{\mathrm{traj}}(X,Y)=
\frac{\min_{\pi}\sum_{(i,j)\in\pi}d(x_i,y_j)}{\min(|X|,|Y|)}.
$$

最终相似度在实例分布与起点分布上取期望：

$$
\mathrm{BehaveSim}(A_1,A_2)=
\mathbb E_I\mathbb E_s[1-D_{\mathrm{traj}}(T(A_1,I,s),T(A_2,I,s))].
$$

轨迹可以截断或间隔采样。随机算法可以对每步状态分布取样，也可以固定随机种子；论文指出固定种子可复现，但可能使相似度依赖该随机流。

论文公式与官方源码有一个复现差异：源码的 `_cal_dtw_distance` 返回 DTW 累积距离，没有再除以较短轨迹长度；本仓库的 v3 离线画像按论文公式加入了这一归一化，并在协议中固定了采样点数。该选择应作为分析协议记录，不能把源码输出直接称为论文公式的完整实现。

## 2. 怎样进入 AAD 搜索

论文实现了两种集成。

1. **FunSearch+BehaveSim**：先画像 100 个初始化算法并按行为聚成多个岛；生成时以概率 $p_{s1}$ 从两个不同岛选示例，否则在同一岛内选择；新候选根据它与各岛候选的平均 BehaveSim 归入最相似的岛。
2. **EoH+BehaveSim**：把候选质量与相对当前最好算法的行为差异组成多目标选择信号，用 dominance-dissimilarity 同时决定父代选择和种群保留。

FunSearch 集成同时包含初始化聚类、跨岛选例和按行为重新归岛。EoH 集成改变了选择目标。它们展示了行为距离怎样参与搜索，没有给出脱离这些搜索结构的通用停滞阈值。

## 3. 论文证据

| 问题 | 证据 | 支持的认识 |
| --- | --- | --- |
| 静态代码相似度能否代表执行行为 | 四类构造配对，比较 token、AST、CodeBLEU、embedding 与 BehaveSim | 代码相近可以行为不同，代码不同也可以行为相同；执行轨迹提供了静态表征没有的信号 |
| 行为距离进入搜索是否有用 | ASP、TSP、CPP，GPT-5-Nano，FunSearch/EoH 各自与 BehaveSim 版本三重复比较 | 在论文的两种集成和三项任务上，Top-1 与 Top-10 均改善 |
| 跨行为区域选例是否有用 | ASP 上改变 $p_{s1}$，并做初始化聚类消融 | $p_{s1}=0.5$ 优于纯岛内选择；去掉初始化聚类后退化 |
| DTW 是否是唯一有效聚合 | 比较 mean-pairwise、DTW、ERP 与 cosine | mean-pairwise、DTW、ERP 高度相关；主要信息来自 PSTraj，DTW 不是唯一选择 |

## 4. 对 TraceAAD 的认识

改进轨迹和 PSTraj 描述两个维度。改进轨迹记录候选怎样形成，用于选择节点和组织下一次生成上下文；PSTraj 记录候选在 probe 上怎样求解，用于判断两个程序在这些 probe 上是否做出相近决策。同一行为不等于同一算法思想，不同行为也不保证两个思想能够融合。

BehaveSim 可以补充两个过程信号：

- 候选到历史行为的最近距离，用于识别反复生成相同执行行为；
- 两个候选或父子之间的行为距离，用于描述一次生成跨了多远。

停滞需要同时观察质量进展和行为变化。低行为新颖度可能是有效利用，高行为新颖度也可能只是无效发散。交叉可以用行为距离寻找不同父代，但“距离远”只说明固定 probe 上的执行过程不同；融合价值仍要由同一交叉协议下的近邻、远邻和随机配对实验识别。

本仓库的任务适配、稳定性和历史搜索分析见[BehaveSim 行为度量校正](../../experiments/机制实验/2026-08-26-BehaveSim行为度量校正/协议.md)。

本仓库的校正结果只支持固定 probe 上的执行行为描述和停滞诊断候选信号。它不验证语义算法思想簇，不把行为距离直接用于在线调度，也不把远距离父代自动视为适合交叉；对应的匹配实验见[校正结果](../../experiments/机制实验/2026-08-26-BehaveSim行为度量校正/结果.md)。
