# TraceAAD V9.18-R0：原子预算、边界机会评分与全局事实算子

V9.18-R0 是 V9.16 的可证伪微调版本。它只回答两个窄问题：轨迹形成的
新入口是否值得获得一次短暂的再次选择机会，以及极短的全局事实是否能
改善 Explore 的结构性提议。每个 primary slot 只做一次锚点选择、一次
operator 选择、一次生成和一次真实评价。

R0 不把历史 realized gain 写成未来价值，不给退步节点额外奖励，不提供
完整参考程序，不引入独立 reflection call，也不把 Diagnosis 当作搜索
动作。所有更强的想法保留为后续单因素实验，不进入第一批正式长跑。

## 1. 版本边界

### 1.1 冻结的 V9.16 基座

- 八个有效根和单父 Algorithm 树；锚点是具体节点及其形成路径。
- 质量方向统一为越大越好；真实 evaluator 是唯一质量来源。
- 每个 primary slot 只评价一个初始候选；最多两次 bounded repair 单独记账。
- `Refine=0.7`、`Explore=0.3` 固定；每次动作完成后立即重选。
- Refine 使用 V9.16 的当前完整代码与父代改进来时路。
- 错误反馈、解析、超时、checkpoint 和 held-out 口径与正式协议一致。

R0 建立独立 `traceaad_v9_18` package 和版本化 checkpoint，不读取或迁移
V9.16 landing 状态。

### 1.2 两个可运行臂

| 臂 | 分配 | Explore 上下文 | 作用 |
| --- | --- | --- | --- |
| `q-atomic` | `q(a)` | V9.16 原 Explore | 原子质量基线 |
| `q+O-atomic` | `q(a)+lambda_O*sigma_q*O_t(a)` | V9.16 原 Explore | 边界机会评分单因素 |

算子单因素另行进行：固定 `q-atomic`，只把原 Explore 替换成
`Global-Facts-Lite Explore`。联合臂最后才运行，联合结果只支持整体系统
结论。

## 2. 研究认识的修订

### 2.1 轨迹展示价值与在线机会分开

固定锚点和 operator 的展示价值是反事实量：

$$
V_{show}(a,o)=E[q(x')|x_a,h_a,o]-E[q(x')|x_a,emptyset,o].
$$

普通搜索只观察一个条件，不能把这个量直接写入在线分数。History-on/off
固定锚点探针只估计局部条件生成差异，不证明完整搜索收益。

R0 在线使用的量不是 `V_show`，也不是 continuation value，而是一个短暂
的 `opportunity prior`：新 Explore 入口在形成后多获得几次被重新看到的
机会，机会随实际选择次数衰减。

### 2.2 历史增益暂不进入 R0 分数

`improve/plateau/regress` 是重要审计事实，但把它们加权为 `F` 再投入
分配，仍然是历史信用代理，且没有 operator 条件化的未来预测证据。R0
因此不使用 `F`。后续只有在固定锚点 transition 探针和离线 policy replay
都显示预测关系后，才单独测试 `F`。

### 2.3 landing 的保留部分

landing 的合理认识是“新结构有时需要再次观察”，不是“命中后承诺连续
三步”。R0 只保留一次选择后的短暂机会，不锁定后续 operator、child chain
或固定预算。低质量回撤节点不因回撤幅度变大而获得额外奖励。

## 3. 状态和尺度

每个有效 Algorithm 保存完整代码、fitness、父节点、形成 Idea、形成
operator、创建 slot、直接尝试次数 `n_after` 和是否为 Explore 入口。
无效候选不创建节点，但失败类型和 primary slot 仍写入事实日志。

初始化完成后，从恰好八个有效根的方向化质量计算冻结尺度：

$$
sigma_q=median_{r in R}|q(r)-median_{u in R}q(u)|.
$$

根质量、尺度、零尺度分支和初始化 slot 在 checkpoint 与 summary 中记录。
`sigma_q=0` 时关闭机会项，算法退化为 `q-atomic`。

## 4. R0 分配规则

### 4.1 质量主项

对每个有效锚点 `a`：

$$
S_t(a)=q_t(a)+lambda_O sigma_q O_t(a),
$$

固定 `lambda_O=0.10`。该系数在运行前写入配置，不看结果回调。

### 4.2 边界机会项

只有由 Explore 创建且代码不重复的有效节点在创建后进入 boundary 状态：

$$
O_t(a)=I_{entry}(a) exp(-n_after(a)/tau_O),
qquad tau_O=2.
$$

`n_after` 在选择动作确定后立即加一，无论本次生成有效、无效、重复或
修复成功。它表示已经获得的观察机会，不代表成功开发。节点不因相对父代
回撤、invalid、失败或重复而得到额外 `O`。这样 R0 不会把最大回撤误当作
最高潜力，也不会给所有普通 Refine 退步节点加奖励。

### 4.3 Boltzmann 与目标 ESS

对 `S_t(a)` 使用 V9.16 的稳定 Boltzmann 选择和目标 ESS：

$$
p_t(a)=\frac{exp(beta_t(S_t(a)-S_t^{max}))}{sum_b exp(beta_t(S_t(b)-S_t^{max}))}.
$$

$$
ESS_t=max(0.1|A_t|,2).
$$

ESS 只控制概率形状，不是算法簇覆盖保证。每个 slot 记录选择前的完整
分数快照、`q`、`O`、`n_after`、`sigma_q`、beta、ESS、实际选择锚点和
请求 seed。

## 5. Global-Facts-Lite Explore

### 5.1 只改变 Explore 上下文

算子单因素保持任务、当前代码、父代来时路、输出契约、temperature、
max output tokens 和 repair 完全固定，只增加一个有界事实板。事实板在
每个 slot 的选择前由当前已结算工件确定，不包含当前 pending 候选。

第一版最多三条事实：

1. 当前全局 best 的真实 fitness；
2. 距离最近一次全局 best 刷新的 primary slot 数；
3. 最近固定窗口的 valid、invalid、duplicate 比例。

事实板不含完整参考程序、代表锚点 Idea、人工机制簇标签、selection
concentration、coverage、未验证规则或 held-out 结果。失败比例按真实失败
类型保存，生成视图只使用固定摘要。

### 5.2 输出契约

R0 仍使用 V9.16 的 `Idea + Code` 契约。解析器额外容忍并记录模型偶尔
输出的 `Diagnosis`，但 prompt 不要求它，缺少 Diagnosis 不使代码失效，
也不能据此宣称诊断算子已经激活。Diagnosis、事实板快照 hash、prompt
hash、上下文字符数和省略标记进入审计日志。

### 5.3 暂缓参考程序

完整参考程序单独作为后续因素，不进入 R0。V9.13 已显示参考代码可能带来
迁移和复制污染；在没有固定选择规则、AST/行为相似度和有效率停测线之前，
不能把它当作“全局反思”证据。

## 6. 原子运行协议

````text
Create eight valid roots and freeze sigma_q.
While primary slots remain:
    score every valid anchor before the request
    sample one anchor from target-ESS Boltzmann probabilities
    draw Refine or Explore from the fixed 0.7/0.3 stream
    build the exact operator context
    generate one Idea + complete program
    evaluate once; bounded repair stays in the same primary slot
    update direct facts, n_after, and global board
    write the pre-decision and post-result audit records
Return the best valid program by the true objective.
````

没有 landing、maturation、sweep、block gain、active/reserve hypothesis、
独立锚点后验或动态 operator 比例。

## 7. 实验识别顺序

详细执行表见[TraceAAD V9.18-R0 实验协议](../experiments/TraceAAD-V9.18-R0实验协议.md)。

1. **实现与固定锚点探针**：验证 History-on/off 和 Global-Facts-Lite 的
   prompt 差异、有效率、单步 `Delta q`、修改幅度、prompt hash 和 token
   记录。共同 seed 只表示请求条件阻断，不表示相同输出。
2. **评分单因素**：`q-atomic` 与 `q+O-atomic` 共享精确八根 checkpoint，
   运行级配对三重复。称为 matched initialization，不称逐 slot 反事实。
   另做同一事实轨迹上的离线 policy replay，验证机会项是否真的改变选择。
3. **算子单因素**：固定 `q-atomic`，比较原 Explore 与 Global-Facts-Lite。
   先固定锚点快照，再做在线完整搜索；在线事实板各自更新，不宣称同快照。
4. **联合版本**：只有评分和算子各自通过过程激活与质量门槛后，才运行
   `q+O-atomic + Global-Facts-Lite`。

所有完整搜索使用 1000 primary slots、三次独立重复和完整同规模
held-out。搜索 best、100/250/500/750/1000 best-at-budget、held-out、
有效率、timeout、repair、duplicate、prompt/response 成本分开报告。

## 8. 判定标准

过程激活只说明机制运行。要说“work”，至少需要：

- 评分项实际改变选择几何，且没有系统性压缩入口覆盖或过投低质量入口；
- Global-Facts-Lite 改变可执行提议，且没有有效率、timeout 或近复制的
  系统性恶化；
- 完整搜索和 held-out 在预先指定的任务族上三重复同向改善；
- 其他任务族没有明显系统性退化。

只有一个任务改善时，结论写成条件性有效；只有过程改变而质量不改善时，
写成“机制运行但未改善”；有效率或泛化受损时，写成“运行后有害”。
“替代 V9.16”要求五任务整体不劣，不能由单一 CVRP 改善推出。

## 9. 后续候选，不属于 R0

- `F` 形成路径响应：先做单独 replay 和 operator-conditioned probe；
- 一份真实参考程序：先定义 deterministic reference 选择和近复制停测线；
- Diagnosis 强制契约：先记录缺失率和事实引用一致性，再决定是否加入；
- 多窗口、后代回传、Thompson posterior、在线聚类和动态 operator 比例。

这些机制不与 R0 同时扫描。

## 10. 主张边界

V9.18-R0 只验证：**质量主导的原子分配是否可以给新 Explore 入口一个
短暂、可衰减且不承诺连续预算的再选择机会，以及极短真实全局事实是否
能改善 Explore 的下一步提议。**

它不声称已经量化轨迹展示价值、估计长期 continuation value、识别算法簇，
或保证跨任务统一升级。
