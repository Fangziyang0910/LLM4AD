# TraceAAD V9.1：MCTS-AHD 语义对齐机制

V9.1 是独立于 V9-Core 的 TraceAAD 方法版本。它保留真实匹配历史、四个轨迹算子和一次
`Idea + Code` 生成，只重构持续搜索中的 MCTS 语义。

## 运行契约

| 配置 | V9.1 默认值 |
| --- | --- |
| 协议 | `traceaad-v9.1-mcts-aligned` |
| 初始根数量 | `n_init=4` |
| 每次扩展候选 | `offspring_per_iteration=1` |
| 渐进式拓展 | `int(visits ** alpha) > len(children)` |
| `alpha` | `0.5` |
| UCT 探索系数 | `lambda_0=0.1` |
| 价值归一化 | 全树 `q_min/q_max` min-max |

`v9_1` 是实验 runner 中的版本标识，实验目录为
`experiments/<task>/traceaad_v9_1/`。V9 的实现、协议和 checkpoint 不会被复用或覆盖。

## 持续搜索

虚拟根和普通程序节点都参与渐进式拓展。选择从虚拟根开始：当节点的拓展条件未满足时，
用 UCT 在已有子节点中选择一个并继续向下；条件满足时，当前节点获得一次新的子节点生成
机会。虚拟根没有程序代码，因此根拓展使用初始化的 `Idea + Code` 提示契约；普通节点拓展
使用该节点的真实匹配历史和选中的轨迹算子。

V9.1 不把 `new_child` 作为与已有子节点竞争的价值选项，也不使用 V9 的
`expansion_quality` 或 `expansion_batch_rewards`。新子节点是否出现只由渐进式拓展条件决定。

## 价值与回传

叶节点的 continuation value 等于自身有向 fitness。节点一旦拥有后代，其 value 等于所有
直接子树 continuation value 中最好者；父节点自身原来的高分不再与后代同时竞争。有效子节
点评价完成后，沿虚拟根到被扩展节点的路径执行一次回传：路径节点的有效访问次数增加一次，
而生成、解析或 evaluator 失败不会增加有效访问次数。

树同时维护所有已评价节点的 `q_min` 和 `q_max`，UCT 利用项按该范围归一化，探索项仍按剩余
预算衰减。checkpoint 保存树、访问状态、质量边界和版本协议身份，并在恢复时验证它们的一致性。

## 不变部分

- 当前程序的完整代码和真实形成历史进入上下文；
- `trace_ideate`、`trace_refine`、`trace_synthesize`、`trace_transfer` 四个入口不变；
- 双轨迹算子仍从其他根分支抽取参考程序及其真实历史；
- evaluator、fitness 方向和 global-best 规则不变。

机制证据与待验证实验见 [RQ-005](../research/RQ-005-V9.1-MCTS语义对齐.md)。
