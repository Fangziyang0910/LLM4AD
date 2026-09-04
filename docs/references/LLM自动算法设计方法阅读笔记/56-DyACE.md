# DyACE

- 论文：*DyACE: Dynamic Algorithm Coevolution*；本地来源：`../../../../papers/DyACE_Dynamic_Algorithm_Coevolution/paper.pdf`；设计对象：随求解阶段在线变化的组合优化启发式控制逻辑。

## 1. 核心问题与方法

DyACE 不把启发式当作一次生成后固定的程序，而把算法与解种群共同演化。look-ahead 模块从当前求解过程提取 Search Trajectory Features，Diagnosis Agent 形成 verbal gradients，Meta-Controller 在 receding horizon 中更新启发式逻辑。

## 2. 论文宣称的机制贡献（逐项）

- 以非平稳双层控制重述动态启发式设计。
- 用真实 search trajectory features 感知 landscape 和 operator 状态。
- grounded perception 是在线适应稳定工作的必要条件。

## 3. 实验究竟支持了什么

|机制主张|论文证据|证据等级|判断|
|---|---|---|---|
|完整方法在 JSSP/TSP/CVRP 有竞争力|Tables 1–3、Figure 2|间接支持|联合 look-ahead、特征、LLM controller 与动态代码的结果。|
|动态 control loop 有益|§4.4、Table 4，Full 对 Static|直接支持|规模越大差距越明显，ta71 为 11.13% 对 17.94%。|
|无感知的动态适应可能有害|Table 4，Blind 对 Static|直接支持|ta71 Blind 19.60% 劣于 Static 17.94%，是明确反例而非仅“无改善”。|
|trajectory features 与 controller 各自独立必要|Table 4|部分支持|Blind 去掉特征但保留 loop，w/o-both 同时去掉两者；未给只保留特征、关闭更新的完全析因设计。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

算法的有效操作随求解阶段变化，在线特征是 controller 的状态估计；没有状态估计，LLM 的频繁改动等于在 operator space 随机游走，会破坏已有动量。这里的轨迹是 solver 内部轨迹，不是算法设计 lineage。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：生成建议前必须让反馈对齐当前 search state。最小验证：动态有感知、动态盲控、静态三组，并同时记录切换次数和额外延迟。
