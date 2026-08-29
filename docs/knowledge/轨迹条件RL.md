# 轨迹条件 RL

本文记录 TraceAAD 在 V9.19 之后要做的搜索—学习工作。V9.19 主实验仍按[完整机制](../methods/TraceAAD-V9.19完整机制设计.md)运行：一 slot 一个候选，搜索控制器固定。本文不改那套搜索规则。

当前主张：

> Algorithm design is a learnable sequential improvement process. Search provides both inference-time optimization and structured learning experience; trajectories expose how algorithms evolve, and RL internalizes these improvement patterns into the model.

分三阶段推进。

$$
\boxed{
\text{Trajectory-guided Search}
\rightarrow
\text{Trajectory-derived Learning Signal}
\rightarrow
\text{Search-and-Learn}
}
$$

## 1. 职责划分

V9.19 负责搜出高质量算法，同时产生结构化决策数据。RL 负责学习：在给定 trajectory state 和 action 时，怎样生成更好的下一步。

第一阶段只训练生成模型，搜索控制器固定：

$$
\boxed{
\text{V9.19 Search Controller 固定}
\quad+\quad
\text{LLM Policy 持续学习}
}
$$

搜索器由 \(P,U,T\) 选择 parent，再用同一个 \(T\) 选择 Develop 或 Explore，形成决策状态

$$
s_t
=
(\text{task},
\text{current code},
\text{behavior-grounded formation trajectory},
\text{action}).
$$

LLM 输出 \(a_t=(\text{Idea},\text{Code})\)，evaluator 给出 \(q(a_t)\)：

$$
\boxed{
s_t
\xrightarrow{\pi_\theta}
a_t
\xrightarrow{\text{evaluator}}
r_t
}.
$$

V9.19 的搜索机制因此同时定义未来 RL 的 state distribution。

第一阶段要学的政策是

$$
\boxed{
\pi_\theta(
\text{Idea, Code}
\mid
\text{Task, Code, Trajectory, Action}
)
}.
$$

Action policy \(\pi_\phi(o\mid s)\) 和 parent allocation policy \(\pi_\psi(a\mid\mathcal A_t)\) 留到生成政策成立之后，顺序为

$$
\boxed{
\text{Generation Policy}
\rightarrow
\text{Action Policy}
\rightarrow
\text{Allocation Policy}
}.
$$

## 2. Stage I：Search

就是当前 V9.19。目标是构成强搜索，并完整记录每次原子决策。主实验口径不变：五任务、三重复、1000 primary slots。

每个 transition 保存

$$
D_t=
(
\text{task},
\text{parent id},
\text{current code},
\text{formation path},
\text{action},
\text{LLM output},
q_p,
q_c,
\text{result},
\nu,
\text{behavior tag},
P,U,T
)
$$

以及当时模型实际看到和写出的内容：`exact_prompt`、`exact_response`、`model_id`、`sampling_temperature`、`seed`。V9.19 实现按这个接口落 `decisions.jsonl`，训练时才能复原 \(s_t\)。

## 3. Stage II：Learning Signal

把搜索痕迹转成

$$
\boxed{
\text{Decision State}
\rightarrow
\text{Candidate Action}
\rightarrow
\text{Outcome}
}.
$$

关键是规定哪些候选在什么状态下相互比较。GRPO 下，同一个

$$
(\text{parent},\text{trajectory},\text{action})
$$

构成一个 group context。

搜索器选定 parent \(a\) 和 action 后，prompt \(x\) 完全确定。GRPO 从同一 \(x\) 采样 \(K\) 个候选，例如 \(K=4\)：

$$
y_1,\ldots,y_K\sim\pi_\theta(\cdot\mid x).
$$

Group 必须 action-conditioned：同一 parent、同一 trajectory、同一 action 才进同一 group。Develop 与 Explore 分开放进不同 group。Explore 的即时 fitness 通常低于 Develop；混在一个 group 里会把政策推向安全的小修改。Explore group 只比较谁是更好的 Explore。

第一版 reward 只用同 group 内的真实 evaluator fitness：

$$
r_i=q(y_i).
$$

GRPO 的 group-relative advantage 在同 task、同 prompt 内比较，绝对尺度随任务变化不影响这一比较。Invalid / timeout / duplicate 的 reward 低于该 group 中全部有效候选。BehaveSim 继续作为搜索控制信号和分析信号，第一版不进入 RL reward。

在必须 Explore 的条件下，政策学习的是怎样 Explore 得更好：有质量的结构变化，而不是距离越大越好。

轨迹作为 prompt 的一部分进入 \(x_t\)。政策在大量不同 formation path 上更新后，学习的是算法改进模式：连续堆机制而无收益时收手，刚改善的机制可以继续 refine，连续 regress 时换方向。

## 4. Stage III：Search-and-Learn

形成在线循环：

$$
\boxed{
\text{Search}
\rightarrow
\text{collect rollouts}
\rightarrow
\text{GRPO update}
\rightarrow
\text{better policy}
\rightarrow
\text{Search}
}.
$$

V9.19 每个 state 只生成一个候选。RL 版本（例如 V10）再引入独立的 RL rollout group，不回改 V9.19 主实验的原子协议。

推荐的边搜边训节奏：每 \(B=50\) 个普通 search slots，从最近 replay buffer 采 \(N=8\) 个 decision states，每个 state 采 \(K=4\) 个 rollout，一次更新额外评价 32 个候选，然后带着更新后的模型继续搜索。

```text
Run 50 atomic search slots
        ↓
sample 8 stored trajectory states
        ↓
4 candidates/state
        ↓
evaluate 32 rollouts
        ↓
GRPO update
        ↓
resume search with updated model
```

Training rollouts 与 primary search budget 分开记账。1000-slot 搜索结果不把额外训练评价算进去。

## 5. 现在做什么

先把 V9.19 跑起来。日志按 \(D_t\) 和 exact prompt/response 保存，作为 Stage II 的 replay 接口。
