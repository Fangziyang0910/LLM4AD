# TraceAAD V9.8 机制识别实验

> 状态：Stage P 已完成。P1/P2 批次为 `20260815_221500_v98_p1_p2`，P3 批次为 `20260816_001100_v98_p3`；结果见[机制识别实验分析](../analysis/TraceAAD-V9.8机制识别实验分析.md)。完整 V9.8 的 1000-eval 批次 `20260815_225000` 已在 server3 启动但尚未完成，当前只构成运行事实，不构成性能证据。

## 1. 识别对象

V9.8 的机制识别分开处理 proposal 与 allocation。固定锚点实验只识别单步生成分布、历史作用和短续段价值；完整搜索消融才识别 hypothesis 聚合、边界宽限与历史发展收益的路由价值。任何固定锚点结果都不替代正式 1000-eval 搜索与 held-out 评价。

## 2. P1/P2：History × Intent

实验单位是从正统 V9.7 批次 `20260814_150927` 三重复事实层抽取的固定锚点。四任务分别按有向质量分成 low / middle / high 三层，每层抽取 6 个代码互异锚点，共 72 个锚点。每个锚点、每个条件重复三次。

采用配对的 $2\times2$ 设计：

| History | Intent |
| --- | --- |
| code-only | Refine |
| code-only | Explore |
| parent-path | Refine |
| parent-path | Explore |

同一个 `anchor × replicate` block 的四个条件共享 sampling seed；条件顺序按预注册旋转并在 task × quality stratum 内打散。总响应数为：

$$
72\times 3\times 4=864.
$$

每个 shard 对一条 trial 执行完整原子流水线：构造 prompt、调用模型一次、持久化原始 response、解析一份完整代码、立即调用对应任务 evaluator、追加 result，然后才进入下一条 trial。解析失败和 no-op 同样立即落盘；不存在“全部生成后再统一评价”的阶段边界。中断恢复优先读取已经持久化的 response，不重复调用模型。

P1 读取 Intent 主效应，检验 Refine 与 Explore 是否改变有效率、即时 $\Delta q$、代码修改规模和静态机制代理切换。P2 读取 History 主效应及 History × Intent 交互，检验 parent path 是否分别帮助两类 intent。独立重复单位仍是锚点；同一锚点内的四条件和三次采样是配对重复观测。

## 3. P3：Explore child 强制续段

P3 只使用 P1/P2 中 `parent_path × Explore` 条件下有效且非 no-op 的 child。P1/P2 的三次采样是嵌套在同一源锚点内的重复观测，因此 P3 不把三个 child 当作三个独立单位：每个源锚点按 replicate 顺序选择第一个有效且非 no-op 的 child，最多保留一个。每个入选 child 克隆为两个协议：

- `child_chain`：只从当前链尖继续；有效新 child 才推进链尖；
- `hypothesis_level`：保留 episode 内全部锚点，每一步按 V9.8 的局部 $q+s_0/\sqrt{n+1}$ 重新选择。

两个协议各运行五次原子 Refine，生成一条便于在 $H\in\{0,1,3,5\}$ 读取的嵌套前缀。每个响应仍立即评价并落盘。两个协议与多个 horizon 都是同一源锚点 / Explore child 内的配对重复测量，不是独立样本。P3 只检验短期 internal gain 与 parent recovery，不直接识别在线边界宽限 $C$ 的完整搜索效应。

## 4. 在线 allocation 消融

完整搜索固定 P1/P2 冻结的 prompt、operator seed schedule 和 evaluator 协议，依次比较：

1. Hypothesis-Uniform；
2. Route-$Q+U$；
3. Hypothesis-$Q+U$；
4. $Q+U+C$；
5. 完整 $Q+U+C+M$；
6. V9.7 与完整 V9.8 联合比较。

各臂使用相同的 1000 次真实 evaluator 预算。搜索过程、最终 held-out 与固定锚点结果分开报告；完整 V9.8 的联合结果不能自动归因给任一分数组件。

## 5. 当前工件与入口

- P1/P2 本地原始工件：`experiments/generation_probe/20260815_221500_v98_p1_p2/`；
- P3 本地原始工件：`experiments/generation_probe/20260816_001100_v98_p3/`；
- 流水线入口：`experiments.runners.traceaad.v98_mechanism_probe`；
- P3 入口：`experiments.runners.traceaad.v98_continuation_probe`；
- 分析入口：`experiments/analysis/analyze_v98_mechanism_probe.py`；
- 正式方法入口：`experiments.runners.traceaad.run --version v9_8`。
- 正式联合批次：`20260815_225000`，四任务 × 三重复，由 `experiments.runners.traceaad.launch_v98` 持续补位。

原始工件只保留本地。Stage P 的结论只从冻结后的完整工件形成；正式搜索完成前仍只报告运行状态，不从中间曲线形成性能结论。
