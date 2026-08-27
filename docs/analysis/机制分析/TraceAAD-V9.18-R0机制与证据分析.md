# TraceAAD V9.18-R0 机制与证据分析

## 核心判断

V9.18-R0 把 V9.16 的 landing 思想收缩为一个原子决策问题：轨迹产生的新
Explore 入口可以获得短暂的再选择机会，但每个 primary slot 仍只选择一个
锚点、生成一个候选并进行一次真实评价。机会项影响的是下一次锚点路由，不是
连续三次预算，也不是长期 continuation value。

生成与分配保持两个接口：Refine 使用父代的形成路径，Explore 使用固定的
探索提示；`q-atomic` 与 `q+O-atomic` 只改变锚点评分。Global-Facts-Lite
作为第二个单因素，只改变 Explore 的有界全局事实板。这样可以分别回答“机会
项是否改变预算流向”和“全局事实是否改变提议”，再决定是否做联合版本。

## 评分的可检验含义

在线机会项为

$$
S_t(a)=q_t(a)+lambda_O sigma_q O_t(a),
$$

其中 `O` 只对有效且非重复的 Explore 入口开启，并按锚点实际被选择的次数
衰减。`sigma_q` 从八个初始化根冻结；当根质量没有离散度时，机会项严格为
零。因而一个任务中记录了入口或 `opportunity>0`，只能证明入口状态存在；
必须检查 `selected_score-parent_q` 或选择快照，才能证明分数真正参与了路由。

机会项的预期作用范围很窄。它的最大增量是
`lambda_O * sigma_q`，只有质量接近时才可能改变 Boltzmann 竞争。分析因此
同时报告入口重访、入口覆盖、分数增量、选择熵和 top-k 份额；这些过程量不能
替代 search best 或 held-out。

## 证据分层

1. **实现事实**：checkpoint 版本、评分模式、`sigma_q`、候选快照、prompt
   hash、repair 和 evaluator calls。
2. **过程激活**：机会衰减是否符合 `tau_O`，分数是否相对 q-only 改变，
   Explore 提议的有效率和重复率是否稳定。
3. **搜索结果**：`100/250/500/750/1000` best-at-budget 与 search best，
   按任务和重复分别报告。
4. **泛化结果**：每个完成重复的独立 held-out；只有三重复完成后才形成任务
   结论，联合版本不能反推单一组件因果收益。

`sigma_q=0` 的任务是机会机制未激活的协议事实，应保留为边界样本，不能把
该任务的 q+O 与 q-only 差异解释为机会评分效果。重复节点仍会进入当前算法
锚点池，但不获得入口机会；重复锚点占用的选择份额单独审计，不与有效入口
覆盖混写。ESS 是概率形状的目标，不是覆盖保证；质量跨度很大时 beta 可能
数值上极端，报告实际熵和 top-k 更可靠。

## 自动执行闭环

`experiments/runners/traceaad/launch_v918.py` 是 A 阶段唯一调度入口。它按
以下规则运行：

- 已完成运行跳过，活动 tmux 会话保持不动；带 V9.18 checkpoint 的目录只做
  `--resume-from`，没有 checkpoint 的已有目录直接报错，避免覆盖工件。
- 通过 `_common.free_slots()` 读取 server3、server3b、server1 和 local 的
  实际空位，动态补位，不改变任务、重复和 bootstrap 配对关系。
- `--watch` 直到 30 路都达到 `status=finished` 与 `budget_slots=1000`；
  搜索未完成时不启动 held-out。
- 搜索完成后按任务×臂自动运行十组 held-out；结果目录保留每次运行的程序和
  配置，重复执行会检查 `run_records`，不会把 partial 结果当作完成。
- 最后调用 `analyze_v918_process.py`，输出 JSON 汇总和 Markdown 过程审计；
  incomplete 运行保留过程记录，但不进入完整运行聚合。

该闭环解决的是执行和证据边界问题，不会把实验结果预先写成机制成功。最终
判定仍遵循 V9.18-R0 协议：过程改变但质量不变记为 `ran_no_improvement`，
有效率、覆盖或质量出现系统性损失记为 `ran_harmful`，只有任务族的三重复
held-out 达到门槛才写 `conditional_work`。
