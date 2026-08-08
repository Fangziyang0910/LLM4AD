# RoCo

- 论文：*RoCo: Role-Based LLMs Collaboration for Automatic Heuristic Design*；本地来源：`../../../../papers/RoCo_Role_Based_LLMs_Collaboration_for_Automatic_Heuristic_Design/main.tex`；设计对象：ACO 启发信息及角色化 LLM 协作流程。

## 1. 核心问题与方法

RoCo 将生成、批评、反思等职责分给预定义角色，并把协作输出接入进化式启发式产生。系统定义包含 agent、role、环境、通信及协作输入/输出；论文还定义启发式表示与反思机制。目标不是训练模型参数，而是通过角色分工提升一次搜索中的候选质量。

## 2. 论文宣称的机制贡献（逐项）

- 角色专门化使设计、批评和修正的上下文分工。
- 协作/反思将多 agent 的文本产物转为可执行启发式。
- 适配 ACO 白盒与黑盒 prompt 设定。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|完整 RoCo 在五类 ACO 任务上有竞争力|Tables `tab:whitebox`、`tab:black_box_table`，Fig. `fig:whiteboxcurve`|间接支持|主结果均为完整协作配方，不能归因给角色或反思。|
|角色/反思组件的影响|Table `tab:ablation`，§`sec:ablation`|部分支持|表标题为 RoCo components on TSP、含白盒与黑盒；它只支持所列组件删除版本，仍需注意通信量/调用数是否随之变化。|
|多任务稳定性|Fig. `fig:blackbox`|部分支持|黑盒图为四次独立运行的均值和标准差；白盒主表为 64 实例、三次平均，不覆盖所有设置。|
|局部搜索变体解释收益|Table `tab:gls`|间接支持|只说明 LS variants 的表现，不能证明角色协作收益来自 LS。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

角色化可把“提出新想法”和“找漏洞”置于不同条件上下文，减少单次自我确认；反思把失败显式转成修改约束。代价是角色越多 token/轮次越多，因此与单 agent 的公平对照必须匹配模型调用、候选数和 evaluator 次数。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：先定义最小的批评契约（指出哪一段改动为何不合格）。前提：批评能回链代码与分数。风险：多角色只产生冗余文本。最小验证：固定 token/eval，比较单 agent 自反思与一名 critic。
- 可学习点：把协作消息当搜索证据而非算法事实。前提：消息有明确消费动作。风险：漂亮解释与性能脱钩。最小验证：统计被采纳批评后的真实改进率。

## 6. 证据边界

白盒测试集每项 64 实例、LLM-AHD 结果三次平均（`tab:whitebox` caption）；黑盒图为四次运行。任务限 TSP、OP、CVRP、MKP、BPP 的 ACO 框架；没有模型参数训练的证据，也不能把角色数/通信格式外推为普遍最优。

## 7. 论文内定位

`main.tex`：§Proposed RoCo（Fig. `fig:roco-arch`、角色与协作小节）；§Experiments，Tables `tab:whitebox`、`tab:black_box_table`、`tab:ablation`、`tab:gls`，Figs `fig:whiteboxcurve`、`fig:blackbox`；`appendix.tex`。
