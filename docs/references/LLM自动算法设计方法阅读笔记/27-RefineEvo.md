# RefineEvo

- 论文：RefineEvo: Planning-Guided Heuristic Evolution with Bidirectional Experience；本地来源：[main.tex](../../../../papers/RefineEvo_Planning-Guided_Heuristic_Evolution_with_Bidirectional_Experience/main.tex)，分文件 `03_method.tex`、`04_experiment.tex`、`06_appendix.tex`；设计对象为组合优化的启发式进化。

## 1. 核心问题与方法

RefineEvo 将启发式进化分为规划与细化：规划阶段选择下一类修改方向，细化阶段生成代码；经验分为正向（带来改进）与反向（失败/退化）两类，双向经验共同进入后续提示，试图避免只模仿成功样本而重复已知失败。

## 2. 论文宣称的机制贡献（逐项）

1. 规划指导降低盲目代码变异。
2. 正负双向经验共同引导改进。
3. 在 ACO 等任务上提升质量、收敛和存活率。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|整体效果|`04_experiment.tex` 主结果；`curves.pdf`、`bar_chart.pdf`、`aco_results_*`|间接支持|是规划、双向经验及其他流程的联合比较。|
|双向经验池|Appendix 的 `tab:ablation_bep`：移除 experiences 后 TSPLIB 表现退化|直接支持|目标经验池的移除对照支持该 TSP/TSPLIB 场景的局部贡献。|
|规划/操作精炼|Appendix `tab:full_k_ablation` 比较 Random/Planner Selection 与 Fixed-Interval Refinement 的 $k$|部分支持|参数/策略对照支持所测设置，不证明“规划”全部语义内容。|
|经验避免失败|`survival_rate_heatmap1.pdf`、`survival_rate_heatmap2.pdf`|间接支持|过程统计可描述现象，不能确认经验是唯一原因。|

## 4. 机制的底层逻辑

阅读分析：负经验的作用是建立“不要重复什么”的约束，正经验给出可延续的方向；规划使两者不只作为附注而参与操作选择。关键风险是失败常依赖父代、实例和 evaluator，去上下文的负规则可能错误排斥新组合。

## 5. 对 LLM4AD / TraceAAD 可学习之处

|可学习点|成立前提|主要风险|最小验证方式|
|---|---|---|---|
|成对保存成功边与失败边的条件|失败能定位到实际改动|把噪声/timeout写成机制失败|仅纳入重复评价一致的失败。|
|规划建议必须回链到实际操作|操作与提示均有日志|“规划”停留在叙述层|统计每类计划的执行率和一步收益。|

## 6. 证据边界

曲线、雷达图和 survival heatmap 是整体或过程证据；只有明示受控消融才能支持组件因果。需要区分搜索 evaluator 与 held-out 测试、每次运行的预算与 LLM 调用数，并避免把存活率直接等同最终泛化。

## 7. 论文内定位

入口：[main.tex](../../../../papers/RefineEvo_Planning-Guided_Heuristic_Evolution_with_Bidirectional_Experience/main.tex)。方法 `03_method.tex`；实验 `04_experiment.tex`；附录 `06_appendix.tex`（`tab:ablation_bep`、`tab:full_k_ablation`、`tab:std_tsp`、`tab:std_bpp`）；过程图 `curves.pdf`、`survival_rate_heatmap1.pdf`、`survival_rate_heatmap2.pdf`。
