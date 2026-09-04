# TraceAAD 历史 Prompt 素材库（V1–V10.2）

## 目的与结构

本文件系统梳理 TraceAAD 历代核心 Prompt 与上下文设计，将其统一投影到六个关键维度。本素材库聚焦于从第一性原理剖析不同上下文构造逻辑对 LLM 条件生成偏置、代码改写幅度及算法演化的深层影响：

1. **Current state**：当前代码、Idea、fitness、注释/契约信息。
2. **Trajectory**：提供给模型的历史步数、字段和改写上下文说明。
3. **Operator**：要求模型执行的算法改进行为及约束。
4. **Reference**：是否提供 donor/elite/外部算法及其机制信息。
5. **Meta instruction**：关于复杂度、可靠性、去重和探索偏好的全局指导。
6. **Output contract**：Idea/Code 格式、函数契约及解释限制。

本文件是设计素材库，不是历史 Prompt 效果排名。历史版本的最终 fitness、排名和完整搜索结果只作弱参考，用于确认设计曾出现并辅助选择 replay 范式；它们混合了搜索策略、预算分配、算子比例、模型调用和评价过程，不能归因某个 Prompt 成分，也不能恢复已经丢失的逐步 proposal 质量。

## 版本索引与六维表征

| 版本 | 源码快照（提交；路径） | Current state | Trajectory | Operator | Reference | Meta instruction | Output contract | 状态说明 |
|---|---|---|---|---|---|---|---|---|
| V1 | [`f2fade66`](https://github.com/Fangziyang0910/LLM4AD/blob/f2fade662cbbc74069e0338c54828ba223ca8cd8/llm4ad/method/traceaad/prompt.py)；`traceaad/prompt.py` | 初始模板函数；改进时给当前节点 Idea、fitness、完整代码 | bounded trajectory；最多 5 个 action 相关历史节点，含 action、父/子 Idea、fitness 变化、outcome | 先让模型生成恰好 `action_count` 个候选修改，再由第二次调用实现单个 action；每个 action 只改一个主要机制 | 无 donor | 依据 fitness 方向判断 improved/regressed/unchanged；历史是“attempted modifications and observed outcomes” | `Idea:` + `Code:` fenced Python；保持函数名、参数、返回类型、输出契约；不输出 rationale/tests | 原始 Prompt 完整可追溯；V1 标签依据早期实验版本映射 |
| V2 | [`f2fade66`](https://github.com/Fangziyang0910/LLM4AD/blob/f2fade662cbbc74069e0338c54828ba223ca8cd8/llm4ad/method/traceaad2/prompt.py)；`traceaad2/prompt.py` | 当前程序 Idea + 完整代码 | Prompt 本身只有当前程序；history/reference 由后续调用方接口传入 | 单个自然语言 action→Code；调用方可把结构化 action 序列化为请求文本 | Prompt 本身无 donor 区块 | “Implement the requested modification”；目标函数契约不变 | `Idea:` + 完整函数代码；仅输出结果 | 原始 Prompt 完整可追溯；V2 是 `traceaad2` 线 |
| V3 | [`e8e94f6f`](https://github.com/Fangziyang0910/LLM4AD/blob/e8e94f6f7bc6386fb172e47aeb860aaac77eeddf/llm4ad/method/traceaad/prompt.py)；`traceaad/prompt.py` | 当前 `ProgramNode` 的 node id、Idea、完整代码；带目标函数模板 | 无默认历史段（history 由调用方后续版本接入） | 调用方先给自然语言 `action`，Prompt 负责单 action→code；自然语言改动请求 | 无 donor 字段 | 明确 fitness 趋势；目标函数接口和可用信息边界 | `Idea:` + `Code:`；不输出 rationale、analysis、tests、额外文本 | 原始 Prompt 完整可追溯 |
| V4 | [`d79c4a5e`](https://github.com/Fangziyang0910/LLM4AD/blob/d79c4a5e6904199f84bc0d416ecf2e4942a78867/llm4ad/method/traceaad/prompt.py)；`traceaad/prompt.py` | 与 V3 相同，增加当前节点的实现上下文 | 新增 `[History Available During Implementation]` | `action` + `[Operator Constraint]`；允许算子约束由搜索器注入 | 无固定 donor | history 与 operator constraint 分区，默认约束为保持目标契约并实现请求 | 与 V3 相同；完整可执行实现 | 原始 Prompt 完整可追溯 |
| V5 | [`7da32dc7`](https://github.com/Fangziyang0910/LLM4AD/blob/7da32dc745f540c361c4b19163e7bf93ef0fc33d/llm4ad/method/traceaad_v5/prompt.py)；`traceaad_v5/prompt.py` | `[Primary Program: the only structural parent]`，Idea claim、代码；目标函数 | `Primary Trajectory Context`；可选 `Reference Trajectory Context` | StructuredAction 的 `relation/change/novel_difference`；operator constraint；明确“exactly the requested change” | `[Reference Program: knowledge only, never a parent]`，给 donor Idea、fitness、代码、history | 允许复杂实现但要求结构父代唯一；不把 edge/global experience 当实现对象 | `Idea:`（brief implementation claim）+ fenced complete implementation；不输出解释 | 原始 Prompt 完整可追溯；该快照代表 V5.0 初始契约 |
| V5.1–V5.4 | [`d2ed6698`](https://github.com/Fangziyang0910/LLM4AD/blob/d2ed66988aed22f71fa088851c46b78f9cded5d6/llm4ad/method/traceaad_v5/prompt.py)、[`3cc726d2`](https://github.com/Fangziyang0910/LLM4AD/blob/3cc726d21c09405a60212376125a2de71543a195/llm4ad/method/traceaad_v5/prompt.py)、[`bb40ff05`](https://github.com/Fangziyang0910/LLM4AD/blob/bb40ff05fc7e5d589cce4b7577295b2470c68a36/llm4ad/method/traceaad_v5/prompt.py)、[`3035b01e`](https://github.com/Fangziyang0910/LLM4AD/blob/3035b01e8024e1f34bfbbe77f8c09c7e2c2a615c/llm4ad/method/traceaad_v5/prompt.py) | 保持 primary program + Idea + code | primary history；reference history 保留 | 结构化 action 失败后回退自然语言 action；V5.2/V5.3/V5.4 继续修订上下文措辞 | donor 为知识来源，不是结构父代 | 删除全局反思门禁后，Prompt 仍强调当前改动和目标契约 | `Idea:` + `Code:`；完整程序 | Git 中可追溯多个修订快照；没有单独的 V5.x Prompt 文件 |
| V6 | [`fde9129c`](https://github.com/Fangziyang0910/LLM4AD/blob/fde9129cc04ebbd80380c0835fe5cc4d6d4db1e8/llm4ad/method/traceaad_v6/prompt.py)；`traceaad_v6/prompt.py` | `[Current Program History]`、当前完整代码；Idea 不再单独格式化为 claim | history 字符串；原文说明“histories describe what has been tried” | 自然语言 `action`；可选 reference；不再要求 StructuredAction 字段 | 可选 reference code + reference history | “simple, complete, valid”；允许模板 imports 和小 helper；fitness direction hint | Idea 一句话（最多 300 字符）+ 完整代码；函数契约不变 | 原始 Prompt 完整可追溯 |
| V7 | [`7e7f260c`](https://github.com/Fangziyang0910/LLM4AD/blob/7e7f260ce22c6aa173269565175577fb29f4afc1/llm4ad/method/traceaad_v7/prompt.py)；`traceaad_v7/prompt.py` | 与 V6 同构：current history + current code | history；可选 reference history | 自然语言 action；完整实现 | 可选 reference program/history | 保持接口；允许 imports/top-level helpers；Idea ≤300 字符 | Idea + fenced Python；解析器接受完整程序 | 原始 Prompt 完整可追溯 |
| V8/V8.2 | [`85e97ff2`](https://github.com/Fangziyang0910/LLM4AD/blob/85e97ff2c5ecdc3d260a121fc6bf9c1924293a55/llm4ad/method/traceaad_v8/prompt.py)；`traceaad_v8/prompt.py` | 明确源码注释“V8 retains the V5 program generation and parsing protocol” | 继承 V5 的 history 结构 | 继承 V5 action/parse；树分支由调用方决定 | 继承 V5 可选 donor | 无新增 Prompt 全局原则 | 继承 V5 | 复用证据：源码只是一层 re-export，不应重复解释成新模板 |
| V8.3 | [`04aaef90`](https://github.com/Fangziyang0910/LLM4AD/blob/04aaef90fd15f92c8fc2d4d318f85db101c4b2ac/llm4ad/method/traceaad_v8_3/prompt.py)；`traceaad_v8_3/prompt.py` | `[Current Algorithm]` 给 fitness、模型生成的 Description、完整 code | `[Local Exploration Context]`；由调用方提供局部历史摘要 | `[Operator]` 给名称和 instruction；要求一次 meaningful change，避免重复已试方向 | 初始化可给多个 `[Reference Algorithm i]`；搜索时可给一个 reference 的 description/code/fitness | 允许必要的参数、条件、数据流或局部结构改动；禁止 placeholder，Idea 必须与代码一致 | 严格 `Design Idea:` + `Code:` 完整可执行实现；另有独立 description call | 原始 Prompt 完整可追溯；V8.3 引入 description 与 reference-aware 文本 |
| V9 | [`444bc65d`](https://github.com/Fangziyang0910/LLM4AD/blob/444bc65d1541182b22fab23bad6fa665aefc90c8/llm4ad/method/traceaad_v9/prompt.py)；`traceaad_v9/prompt.py` | 继承 V8/V5 的完整当前程序和 fitness | 形成路径/局部历史由调用方传入；默认单步实现 | Refine/Explore 语义由调用方 action/constraint 表达 | 无固定 donor | 继续保持完整可执行程序和接口契约 | Idea + Code | 原始 Prompt 可追溯；V9-Core 的 Prompt 与 V8 同源 |
| V9.1 | [`6adf870e`](https://github.com/Fangziyang0910/LLM4AD/blob/6adf870ea035d8acd12b26ae79f01b1121630f86/llm4ad/method/traceaad_v9_1/prompt.py)；`traceaad_v9_1/prompt.py` | 当前锚点/代码/fitness | MCTS 语义对齐的 trajectory；含 action/outcome | 由 action prompt 先生成候选修改，再实现 | 无稳定 donor | 强调 fitness direction 和可执行改动 | Idea + Code；完整函数 | 原始 Prompt 可追溯 |
| V9.2 | [`a077b375`](https://github.com/Fangziyang0910/LLM4AD/blob/a077b375be9510dc8d914625520edbdf76713d14/llm4ad/method/traceaad_v9_2/prompt.py)；`traceaad_v9_2/prompt.py` | 锚点 current program + Idea + fitness | 固定局部 history window；含父链形成事件和附近下游尝试 | action 实现；要求结合锚点上下文 | donor 非核心 | 局部窗口和真实 fitness 作为主要证据 | Idea + Code | 原始 Prompt 可追溯 |
| V9.3 | [`bfbf1035`](https://github.com/Fangziyang0910/LLM4AD/blob/bfbf10358aa3c46bc150b0d070498dff241ef872/llm4ad/method/traceaad_v9_3/prompt.py)；`traceaad_v9_3/prompt.py` | 当前可执行 anchor；初始策略与 Idea 分开 | `window_text` 轨迹窗口；可附 `Initial Route Strategy` | 先生成恰好 N 条 `Strategy i:`，再决定一个 next Idea，最后实现已批准 Idea；要求机制级差异、禁止代码/搜索讨论 | 无 donor | 策略必须 task-grounded、互补、非 cosmetic；代码只保留 executable source、去注释 | 策略阶段为编号列表；决策阶段只输出一行 Idea；实现阶段 `Idea + Code` | 原始 Prompt 完整可追溯；三段调用契约是 V9.3 特征 |
| V9.4 | [`bfbf1035`](https://github.com/Fangziyang0910/LLM4AD/blob/bfbf10358aa3c46bc150b0d070498dff241ef872/llm4ad/method/traceaad_v9_4/prompt.py)；`traceaad_v9_4/prompt.py` | 当前可执行 anchor | `window_text` 局部历史；可附初始策略和 `failure_memory_text` | 生成恰好一个 next Idea + Code；失败记忆用于避免重复实现，要求不要把错误文本复制进代码 | 无 donor | `failure_memory` 是显式上下文；保持函数签名/契约，完整有效实现 | 严格一行 Idea + 一个 Code block；缺 Idea 触发 `ProgramResponseError` | 原始 Prompt 完整可追溯；与 V9.3 同提交但不是同构模板 |
| V9.5 | [`8545d737`](https://github.com/Fangziyang0910/LLM4AD/blob/8545d73778d4d17c405394b406c495144a3acd76/llm4ad/method/traceaad_v9_5/prompt.py)；`traceaad_v9_5/prompt.py` | 当前 executable anchor 完整 code + anchor fitness | `evidence_text` 由调用方提供父代/锚点历史和结果 | 单一“Improve the current algorithm”请求；没有 Refine/Explore 文本枚举 | 无 donor | 历史是 evidence 而非 strict prohibition；允许以不同实现重试失败方向 | Optional Idea（≤300）+ mandatory fenced Code；不输出 reasoning/operator/patch | 原始 Prompt 可追溯；存在固定锚点单步配对证据 |
| V9.6 | [`07ac41b9`](https://github.com/Fangziyang0910/LLM4AD/blob/07ac41b9ef9eb7e1cd6b2af328986147d7741b1d/llm4ad/method/traceaad_v9_6/prompt.py)；`traceaad_v9_6/prompt.py` | 当前 executable anchor 完整 code + fitness | `history_text`；源码明确 output/root parser 与 V9.5 identical，仅 anchor history 结构变化 | 单一 coherent modification 请求；Refine/Explore 仍由调用方外部决定 | 无 donor | 保留有用机制，参考已试修改及结果；失败方向可用不同实现重试 | 与 V9.5 相同：optional Idea + mandatory Code | 原始 Prompt 可追溯；V9.5–V9.6 有固定锚点 proposal-level 实验 |
| V9.7 | [`05ce3fb3`](https://github.com/Fangziyang0910/LLM4AD/blob/05ce3fb3381a2a925d5c7f9594468eb22cdaf02f/llm4ad/method/traceaad_v9_7/prompt.py)；`traceaad_v9_7/prompt.py` | 当前代码 + fitness；默认显示父代改进路径 | 父代 formation path，最多 8 条；Direct Attempts 从默认 Prompt 移除 | Refine：保持核心方向；Explore：提出不同主机制 | 无 donor | “histories describe what has been tried”；保持函数契约；Idea ≤300 | 一句话 Idea + 完整 executable code；不输出分析 | 原始 Prompt 可追溯；`05ce3fb3` 明确 Direct Attempts 移除 |
| V9.7-CO | [`d9af37b9`](https://github.com/Fangziyang0910/LLM4AD/blob/d9af37b9629c045155d433b2f56b6889af53e50e/llm4ad/method/traceaad_v9_7_co/prompt.py)；`traceaad_v9_7_co/prompt.py` | current code/fitness | parent path + 跨区域/重访相关 context 由实验调用方注入 | Explore/Refine 以及 region-revisit 约束 | 可能包含前沿区域摘要；不是稳定 donor code 契约 | 试图区分区域和机制族 | Idea + Code | 一次性 probe/ablation 分支；不等同 canonical V9.7 |
| V9.8 | [`0d688b29`](https://github.com/Fangziyang0910/LLM4AD/blob/0d688b29e5a9a70a512bd8a0bba7ff096f356fc4/llm4ad/method/traceaad_v9_8/prompt.py)；`traceaad_v9_8/prompt.py` | `[Current Algorithm]` code + fitness | `history_text`，用于记录 formation path；Explore/Refine 均读同一历史 | `INTENT_INSTRUCTIONS`: Refine 保持 central principle 并 focused change；Explore 改变 central decision principle | 无固定 donor | 避免 placeholder/trivial baseline；完整有效实现 | Optional short Idea（≤300）+ mandatory Code；不输出 reasoning/operator label/patch | 原始 Prompt 可追溯 |
| V9.9 | [`779bd91a`](https://github.com/Fangziyang0910/LLM4AD/blob/779bd91a8488553c5b3fa2431ac74a9eee02bbdf/llm4ad/method/traceaad_v9_9/prompt.py)；`traceaad_v9_9/prompt.py` | current code + fitness | geometry-rank 选择；Prompt 仍为 current + history | Refine/Explore intent；算子概率由调用方调节 | 无固定 donor | 减少额外控制信息，保持完整程序 | Idea + Code | 原始 Prompt 可追溯 |
| V9.10 | [`78726907`](https://github.com/Fangziyang0910/LLM4AD/blob/7872690725789a9b7d4289cf3dd37d58b9557da0/llm4ad/method/traceaad_v9_10/prompt.py)；`traceaad_v9_10/prompt.py` | current algorithm code + fitness | formation path/history；联合 arm（anchor×intent）在调度侧，非输出格式 | Refine/Explore intent | 无稳定 donor | Prompt 仍是完整实现，后验/折扣在选择器而非文本契约 | Idea + Code；完整函数 | 原始 Prompt 可追溯；V9.10 负结果不能归因于文本单一成分 |
| V9.11 | [`6ad6e89d`](https://github.com/Fangziyang0910/LLM4AD/blob/6ad6e89d13fb0961fe41d69ccfcf39a0c7cb0783/llm4ad/method/traceaad_v9_11/prompt.py)；`traceaad_v9_11/prompt.py` | current code + fitness | history；Explore 后一次 Landing 是调度行为，不是额外 Prompt 字段 | stagnation-triggered Explore / landing Refine | 无 donor | Explore 需 coherent alternative；Landing 保持可执行性 | Idea + Code | 原始 Prompt 可追溯；landing 质量证据来自过程记录，不单独证明 Prompt 效果 |
| V9.12 | [`f90acfb5`](https://github.com/Fangziyang0910/LLM4AD/blob/f90acfb574a60b520b97f2a7591404f63f36b005/llm4ad/method/traceaad_v9_12/prompt.py)；`traceaad_v9_12/prompt.py` | current code + fitness | 最近历史窗口；失败率影响 intent 选择而非历史文本结构 | progress-conditioned Refine/Explore | 无 donor | 按局部失败率改变探索节律；保持接口 | Idea + Code | 原始 Prompt 可追溯 |
| V9.13 | [`5a7738ce`](https://github.com/Fangziyang0910/LLM4AD/blob/5a7738ce78add8623cc8b0dd849b334f376a0286/llm4ad/method/traceaad_v9_13/prompt.py)；`traceaad_v9_13/prompt.py` | current code + fitness | history 之外，Explore 可注入已访问机制区域/前沿摘要 | Explore 的 central decision principle 约束；区域前沿条件化 | 前沿表不是 donor code；提供质量/机制标签摘要 | 试图阻止低质量区域重建；容量约束和跨区域标签同时进入上下文 | Idea + Code；不输出分析或 patch | Stage-P 有 proposal-level 证据，完整搜索被提前终止 |
| V9.14 | [`ee7cf712`](https://github.com/Fangziyang0910/LLM4AD/blob/ee7cf712469272432f00087c3ceb8d96b9ab8069/llm4ad/method/traceaad_v9_14/prompt.py)；`traceaad_v9_14/prompt.py` | current anchor + code + fitness | formation path / local history | 单树 unified search 的 Refine/Explore | 无稳定 donor | 继续单步、完整函数契约 | Idea + Code | 原始 Prompt 可追溯 |
| V9.15 | [`9507330b`](https://github.com/Fangziyang0910/LLM4AD/blob/9507330b20c96a0f289ca280e95748c5e7b28d2c/llm4ad/method/traceaad_v9_15/prompt.py)；`traceaad_v9_15/prompt.py` | current code + fitness；错误反馈在 repair prompt 中单独呈现 | trajectory-aware history | intent/control + bounded repair | 无稳定 donor | repair 保持 Idea 和函数签名；错误类型/traceback 是 repair 专用上下文 | Idea + Code；repair 仍要求完整程序 | 原始 Prompt 可追溯；EH 后续并入 V9.15 |
| V9.16 | [`a144f601`](https://github.com/Fangziyang0910/LLM4AD/blob/a144f6016774b50ebd5fc06e9bba3c1530b2e9f2/llm4ad/method/traceaad_v9_16/prompt.py)；`traceaad_v9_16/prompt.py` | current code + fitness | formation path；landing 不改变文本契约 | q baseline + landing 的 Refine/Explore | 无 donor | 控制器简化；完整代码、接口不变 | Idea + Code | 原始 Prompt 可追溯 |
| V9.17 | [`7a5d6ed5`](https://github.com/Fangziyang0910/LLM4AD/blob/7a5d6ed552b71986bc2187078d04d74191a78543/llm4ad/method/traceaad_v9_17/prompt.py)；`traceaad_v9_17/prompt.py` | current code + fitness | parent path/history | Adaptive/FixedCycle scheduler 只改变何时切换，Prompt 仍为 Refine/Explore | 无 donor | 保持 matched prompt、repair、模型和预算；调度器对照不改变文本 | Idea + Code | 原始 Prompt 可追溯 |
| V9.18 | [`7bd9edbc`](https://github.com/Fangziyang0910/LLM4AD/blob/7bd9edbc9e013a9cd8686ba32492c79c1b6877cf/llm4ad/method/traceaad_v9_18/prompt.py)；`traceaad_v9_18/prompt.py` | current code + fitness | history/trajectory；R0 机制主要在选择和结构层 | Refine/Explore | 无稳定 donor | 简化 search skeleton，保持完整程序 | Idea + Code | 原始 Prompt 可追溯 |
| V9.19 | [`517d6cc6`](https://github.com/Fangziyang0910/LLM4AD/blob/517d6cc66e2c58d842f150dc0048bd4e649b6d8b/llm4ad/method/traceaad_v9_19/prompt.py)；`traceaad_v9_19/prompt.py` | current algorithm + fitness + optional behavior description | formation history plus optional BehaveSim/landscape summary | `DEVELOP` preserves framework; `EXPLORE` changes main decision logic; `CROSSOVER` combines behavior-compatible reference | optional reference Idea/description/code/fitness/distance | evidence-backed rationale; avoid cosmetic variation and blind copy | complete Idea + Code | 原始 Prompt 可追溯 |
| V9.20 | [`517d6cc6`](https://github.com/Fangziyang0910/LLM4AD/blob/517d6cc66e2c58d842f150dc0048bd4e649b6d8b/llm4ad/method/traceaad_v9_20/prompt.py)；`traceaad_v9_20/prompt.py` | current algorithm + fitness + optional behavior description | `context_mode=explore` 可只显示 current 并追加 `[Direct Outcome Ledger]`；develop/crossover 可显示 formation/reference history | DEVELOP/EXPLORE/CROSSOVER；按 mode 改变历史暴露和 donor history | crossover 可注入 reference history；行为兼容 donor | 禁止盲目重复失败编辑；compact ledger 用于探索；保持 executable | Idea + Code | 原始 Prompt 可追溯；与 V9.19 同提交但文本明显不同 |
| V9.21 | [`bbf4b61d`](https://github.com/Fangziyang0910/LLM4AD/blob/bbf4b61dbe9c25adbea6e27bd96b80b6f46a7db5/llm4ad/method/traceaad_v9_21/prompt.py)；`traceaad_v9_21/prompt.py` | Idea 阶段给 stable scaffold、可选 working implementation；realization 阶段给 base implementation 和 Idea under test | formation history + implementation evidence/ledger；可选 public experiment card | `continue` 精炼同一 hypothesis；`branch` 提出 materially different hypothesis；Idea 与 realization 分两次调用 | public card 是可选机制来源，不是固定 donor parent | hypothesis 与 executable realization 解耦；独立 realization；bounded execution | Idea 阶段只输出一行 Idea；realization 阶段输出 Idea + Code | 原始 Prompt 可追溯；首跑有过程记录 |
| V9.22 | [`c3bc08e1`](https://github.com/Fangziyang0910/LLM4AD/blob/c3bc08e1dd79262b2d7826edcefbc15f0f84a5a3/llm4ad/method/traceaad_v9_22/prompt.py)；`traceaad_v9_22/prompt.py` | stable scaffold/working code 的当前视图由调用方决定 | frozen batch context；branch context 与 stable scaffold 分离 | branch/develop 约束；初始版本禁用 verified public-card transfer | no donor code；public cards 是 whole-code/idea exposure | interface/branch isolation 等机制在上下文构造层体现 | Idea + Code | 原始 Prompt 可追溯；过程证据不等于效果证据 |
| V10 | [`2b5fe679`](https://github.com/Fangziyang0910/LLM4AD/blob/2b5fe67954e661630b32d11024b179988962de9d/llm4ad/method/traceaad_v10/prompt.py)；`traceaad_v10/prompt.py` | `[Current Algorithm]` fitness + 完整代码；generator sees less than critic | `[Formation Path]`；Restart 只看 verified improvement cards，不给 code | Develop、Pivot、Transfer、Restart、SemanticRepair 五类；分别规定保留/替换核心机制 | Transfer 给 donor code、fitness、Idea、reference formation path；Restart 不给 donor code | reliability：bounded execution、每次返回契约值、不修改输入；Repair 给 failure/traceback | concise Idea + `Code:` fenced complete program；不写 docstring/comments/reasoning | 原始 Prompt 可追溯；V10 已删除，快照保留 |
| V10.1 | [`b7b038fc`](https://github.com/Fangziyang0910/LLM4AD/blob/b7b038fc2ba6c5387adf1a5fb984ea1177c70c83/llm4ad/method/traceaad_v10_1/prompts.py)；`traceaad_v10_1/prompts.py` | `# Current Algorithm`：Idea、fitness、代码；当前代码与目标函数契约分开 | `# Historical Design Trajectory`；显示最近至多 `max_gens`，每步 Idea + fitness + trend | Refine 保持核心原则；Pivot 换主机制；Fuse 选择 current 与 donor 的互补机制 | `# Reference Algorithm`：donor Idea、fitness、完整代码；只在 Fuse 注入 | `# Implementation Principle` 尚未加入 V10.2 的 anti-bloat 细节；输出限制较短 | `Idea:` 一句话 + fenced Python；不要求额外解释 | 原始 Prompt 完整可追溯 |
| V10.2 | [`87265a83`](https://github.com/Fangziyang0910/LLM4AD/blob/87265a834c105875c64333ed56d2fcae8de8adcc/llm4ad/method/traceaad_v10_2/prompts.py)；`traceaad_v10_2/prompts.py` | current Idea、fitness、代码；Prompt view 用 `strip_comments_for_prompt` 去除普通注释，存储/评价代码保持原样 | `# Historical Design Trajectory`；最多 `max_gens`，超长时从最老 generation 开始裁剪；每步 Idea、fitness、trend | Refine：保持核心原则；Pivot：改变主机制、避免 lineage 重复；Fuse：current 保留机制 + donor 兼容机制，替换重叠组件 | Fuse donor 的 Idea、fitness、comment-free code；donor 是参考，不是结构父代 | `# Implementation Principle`：优先质量，允许有用复杂计算；删除重叠机制而非机械叠加；代码少注释；不输出 reasoning | `Latest Design Idea:` + fenced complete function；不写解释/注释/替代方案；严格两部分 | 原始 Prompt 完整可追溯；V10.2 另有 2026-09-03 prompt probe 工件 |

## 代表性生成范式

历史版本按上下文组织方式归并为以下七类。类别用于覆盖明显不同的设计空间，不表示历史效果优劣；同一类别内的版本仍需保留其原始措辞和字段差异。

| 范式 ID | 代表版本 | 生成上下文骨架 | 主要差异 | replay 适配 |
|---|---|---|---|---|
| P1 `current-action` | V2–V4 | Task + Current + action/constraint | 历史最少，改动由调用方或单段 action 指定 | 单次 Idea+Code；不注入 trajectory/donor |
| P2 `current-history` | V6–V7、V9.5–V9.6 | Task + Current + history/evidence | 历史作为“已尝试证据”，允许在同一方向重新实现 | 单次 Idea+Code；使用固定 parent trajectory |
| P3 `structural-donor` | V5、V8 | Primary Current + primary/reference history + donor | 明确 structural parent 与 knowledge-only donor 的边界 | Fuse 提供 donor；Refine/Pivot 去掉 donor |
| P4 `trajectory-intent` | V9.7–V9.12、V9.14–V9.18 | Current + formation path + intent | Refine/Explore 以不同设计行为读取同一来时路 | 映射为 Refine/Pivot；Fuse 使用同一历史骨架加 donor |
| P5 `failure-frontier` | V9.4、V9.13 | Current + history + failure memory 或 frontier summary | 把失败方向、已访问区域或机制标签作为额外证据 | 额外字段固定为状态快照，不能随 trial 更新 |
| P6 `behavior-crossover` | V9.19–V9.20 | Current + formation/behavior summary + reference | donor 带行为距离、机制摘要或 direct-outcome ledger | Fuse 保留 reference 摘要；其他算子只保留 current 侧字段 |
| P7 `explicit-three-operator` | V10.1–V10.2 | 标题化 Current、Trajectory、Operator、可选 Donor、全局原则、Output | 算子指令、anti-bloat、注释过滤和输出契约均显式化 | 作为现代完整端点；只在固定状态上与 P1–P6 比较 |

V1、V9.1、V9.3、V9.21 等包含独立的策略/Idea 预生成调用。它们属于 `plan-then-realize` 变体，调用次数和生成协议与单次 Idea+Code 不同，不放入主 replay 的同一处理臂；如需研究，另设调用预算匹配的子实验。

## 按 Prompt 成分归类

### 1. Current state

- 所有可核对版本都至少提供当前可执行代码和目标函数契约。
- V3–V7 主要以 `Idea + code` 作为 current state；V8 以后逐步显式加入 fitness。
- V10.1/V10.2 将 current、trajectory、operator、reference 设为稳定的标题化区块。
- V10.2 只在 prompt view 去普通注释，未改变存储或评价代码；这是上下文呈现选择，不是代码语义修改。

### 2. Trajectory

- V1/V2 的 trajectory 是“尝试过的 action 及其结果”，可包含多个节点和边。
- V5–V8 允许 primary/reference history；V9.2–V9.6 逐步固定局部窗口和锚点历史。
- V9.7 以后形成路径成为默认历史，Direct Attempts 从 canonical Prompt 移除。
- V9.8–V9.22 主要改变历史如何被调度、截断或扩展；V10.2 明确旧 generation 优先裁剪。

### 3. Operator

- 早期（V1–V7）由模型先提出 action，再实现 action，或由调用方给自然语言 action。
- V9.5 起 Refine/Explore 成为稳定意图对；V10 引入 Develop/Pivot/Transfer/Restart/SemanticRepair；V10.1/V10.2 收敛为 Refine/Pivot/Fuse。
- V10.1/V10.2 的算子措辞已从“参数调整或表面重构”提升为“主机制层面的保持、替换和兼容组合”。

### 4. Reference

- V1–V4、V9.x canonical 大多没有固定 donor code。
- V2/V5–V7 有 donor/reference，但源码明确它是 knowledge-only，不是 structural parent。
- V9.13 的前沿/机制标签不是 donor 算法；V10 Transfer/Fuse 才把另一个算法的代码或机制正式放入生成上下文。

### 5. Meta instruction

- 早期重点是格式和函数契约；V5 引入“primary structural parent”边界。
- V6/V7 强调 simple/complete/valid、允许模板 imports 和小 helper。
- V9.7 起强调避免重复和保持核心方向；V10 加入 bounded execution、输入不变性和 repair 约束。
- V10.2 的 anti-bloat 表述是选择性的：允许有独立作用的昂贵计算，只删除重叠或被取代机制。

### 6. Output contract

- 共同骨架是 `Idea`（或 `Latest Design Idea`）加一个完整 Python fenced block。
- 从 V3 起反复要求保持函数名、参数、返回类型和 evaluator contract；V10 还限制 docstring/comments/reasoning。
- V5 的 StructuredAction 是输入 action 的契约，不是最终输出格式；V5.1 后回到自然语言 action。

## 历史测试与观察参考

已有固定锚点或 Stage-P 工件可以作为方法学参考，说明配对 replay 如何记录响应；它们不被回写成历史 Prompt 的统一优劣，也不替代新的代表性范式 replay。

### V9.5–V9.6 固定锚点单步识别

现有汇总 [`docs/analysis/机制分析/TraceAAD-V1-V9.6机制诊断.md`](./机制分析/TraceAAD-V1-V9.6机制诊断.md) 记录三轮、合计 1,512 次固定锚点生成；其中第三轮主实验为 648 次，比较 `code_only`、`+父代来时路`、`+父代来时路+直接子代尝试`。报告的可用事实是：父代来时路在四个任务上的配对区间均为正，72 个锚点中 58 个更好、14 个更差；追加 direct attempts 的区间均跨 0（35 好、34 差、3 平）；极端改写比例同步收缩。该结果表明“父代形成路径是有用上下文候选，局部失败尝试没有稳定额外收益”，不支持把 V9.5 或 V9.6 的全部搜索结果归因于单一 Prompt。

### V9.7 canonical

V9.7 有完整搜索工件和局部 trajectory 分析，但其 Prompt、分配、树/路径选择共同变化。现有结果可说明 canonical V9.7 是一个强联合协议；不能从最终 best 识别“最多 8 条父路径”或“Direct Attempts 移除”的单独因果效应。

### V9.13 Stage P

V9.13 的 Stage-P 工件比较了前沿表、地板表和标签措辞。部分条件下前沿信息使 Explore 单步均值提高并减少亚前沿重入；但 CVRP 上跨区域机制标签同时造成运行期有效率下降，实验按预设红线终止。这里可记录为“前沿上下文改变了 proposal 分布并伴随尾部风险”，不能概括为普适有效 Prompt。

### V10.2 Prompt probe

针对 V10.1 与 V10.2 的 Fuse 算子端点拆分测试（340 次固定状态生成）表明：文本指令与上下文细节会显著重塑模型的 proposal kernel。例如配对均值差显示：CVRP `+0.0396`、OBP `+11.6818`、OP `+0.0370`、TSP `-0.7909`。去注释约束与算子硬指令对生成稳定性的影响高度依赖具体任务的语法空间。

## 核心机理总结

1. **父代形成路径是最稳定的正向上下文**：“当前完整代码 + 匹配父代形成路径（Idea、fitness 变化、已尝试方向）”能有效帮助模型维持算法核心骨架并开展定向利用。
2. **失败尝试不宜直接进入默认提示**：直接展示子代失败尝试容易诱导模型围着缺陷代码局部纠缠，移除直接失败记录能促使模型进行更有价值的有效重写。
3. **算子指令应形成清晰分工**：Refine 聚焦局部一致性发展；Pivot 强调核心假设替换；Fuse 限定外部机制的选择性融合与兼容性去重。
4. **上下文精简与去冗余**：仅保留支撑当前决策必需的机制上下文，剔除重叠与无实质算法作用的解释性文本，减少注意力被无关事实分散的风险。
