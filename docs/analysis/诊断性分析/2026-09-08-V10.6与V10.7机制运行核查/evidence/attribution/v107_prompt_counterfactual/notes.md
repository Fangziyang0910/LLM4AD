# 固定 Prompt 反事实 pilot

24/24次生成、24/24次正式配置的隔离评价全部成功，finish_reason均stop，无重试、无补采样、无timeout。仅每条件2条，不能作统计显著性或长期搜索效果宣称。

固定4个锚点×A/B/C×seed 20260908/20260909；server3b同一profile，temperature=1/top_p=.95/top_k=20，thinking=False，max_tokens=16384，3个spawn worker随机交错作业。所有prompt输入<=16128；A逐字重建并与原call hash一致。plan.json含12个完整prompt、hash、锚点记录、父代和材料、donor替换及固定作业顺序。

- A：原Prompt。
- B：仅清空展示块Idea段，保留所有Code（含注释）、顺序、fitness和指令。
- C Pivot：同父代的ancestor_history builder与真实祖先，保留100words输出约束；它同时改变材料、算子措辞和代码注释展示，是bundle对照。
- C Fuse：保留辅助程序，将donor换为当时档案中最高分且容量允许的跨血缘不同代码程序，重新排序绑定角色。这是最强可用donor上限对照，不是旧top5均匀抽样复刻。VRPTW donor142→167；TSP donor99→366。

表内两个分数依次对应两个固定seed，higher is better。

| 锚点（父代fitness） | A原Prompt | B去Idea | C对照 |
|---|---|---|---|
| TSP Pivot r2 c97，parent83 (-6.82397) | -7.69884 / -7.62590 | -6.82397 / -6.78870 | -8.61902 / -7.99895 |
| OBP Pivot r3 c104，parent43 (-744.5) | -744.5 / -744.5 | -744.5 / -2359.5 | -3000 / -3000 |
| VRPTW Fuse r1 c197，parent127 (-20.69657) | -20.91309 / -20.80548 | -21.03378 / -20.78929 | -21.13827 / -20.69657 |
| TSP Fuse r3 c399，parent383 (-6.71086) | -6.71086 / -6.71086 | -6.84976 / -6.71086 | -6.99930 / -6.71086 |

观察：去Idea在TSP Pivot这个锚点的两条均优于A，但其它锚点效果混合；不能得出去Idea普遍改善。Pivot旧bundle的4条均差于A，但样本太少且变化打包，不能证明祖先历史机制总体较差。加强donor在两个Fuse锚点也未稳定改善。

实际代码反证：TSP Pivot B seed09仍有sub填inf后sum的主分数失效，只新加n_remaining==2精确比较；-6.78870的略提分不能理解成复杂决策被修正。TSP Fuse A seed08与C seed09也保留inf行均值/regret造成的主打分失效。OBP Pivot C seed08明确实现slack越大分越高的Worst-Fit，因此-3000来自设计方向本身，不是解析或执行失败。

本pilot不能唯一归因V106/V107长期差距，也不能比较独立校准的因果效应。B仍保留代码注释，未移除所有文字解释。所有输出均通过同一V106严格parse_response；没有因格式错误补救而改变样本组成。

成本：119975总tokens（88868输入、31107输出），生成耗时总和923.53秒，评价耗时总和24.59秒；并发运行的wall time更短。未触碰正式run、预算或检查点。

文件：summary.json为聚合；各job同名.json含原始response/usage/finish/评分/时间，.py为解析出的完整评价代码；plan.json和12个.prompt.txt为固定输入。执行脚本/tmp/v107_prompt_counterfactual.py禁止重复运行已有reserved/completed job，可用--summarize只读重算汇总。
