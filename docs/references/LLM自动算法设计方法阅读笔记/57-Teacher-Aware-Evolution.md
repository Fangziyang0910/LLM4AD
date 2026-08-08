# Teacher-Aware Evolution

- 论文：*Teacher-Aware Evolution of Heuristic Programs from Learned Optimization Policies*；本地来源：`../../../../papers/Teacher_Aware_Evolution_of_Heuristic_Programs/paper.pdf`；设计对象：部署时不依赖教师的静态启发式程序。

## 1. 核心问题与方法

候选程序先 rollout 产生 on-policy states，再查询一个独立训练的优化 policy，比较程序动作与 teacher preference；analyzer 把分歧转成 structural rewrite、parameter calibration、mechanism fusion 三类修改建议。种群以任务目标与 teacher alignment 做 Pareto 更新，最终仍按任务目标选程序。

## 2. 论文宣称的机制贡献（逐项）

- learned teacher 提供比终端分数更密集的动作级行为反馈。
- on-policy 查询覆盖候选实际访问而非教师自身轨迹。
- 教师只指导搜索，不要求最终程序忠实模仿或部署教师。

## 3. 实验究竟支持了什么

|机制主张|论文证据|证据等级|判断|
|---|---|---|---|
|完整方法跨 JSSP/TSP/CVRP/MaxCut 有竞争力|Tables 3–7|间接支持|联合 teacher、analyzer、operators 与 Pareto selection。|
|teacher feedback 有益|§4.4、Table 8 Performance-only|直接支持|移除全部教师反馈后 ID/OOD 均退化。|
|analyzer 与 teacher-guided operators 有益|Table 8|直接支持|各自移除均在 JSSP/TSP 退化。|
|三种 revision mode 各有贡献|Table 8|直接支持|逐项 w/o structural/parameter/fusion 均比完整方法差，但幅度依任务。|
|最大化 teacher alignment 可替代任务目标|Table 8 Max-align|反向或混合证据|按 alignment 选最终程序更差，说明教师应是辅助信号而非目标。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

teacher 把稀疏终点奖励分解为局部行动偏好，而任务 evaluator 防止学生复制教师缺陷。on-policy 查询尤其重要，因为候选程序会访问教师 rollout 看不到的状态。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：用强 solver/learned policy 诊断候选，而不把它当最终答案。最小验证：teacher agreement 是否预测后续真实增益，并保留 objective-only final selection。

## 6. 证据边界

每个任务需独立训练且接口可对齐的教师，也增加 analyzer 调用；因此不是免费训练信号。教师参数不由 AAD 过程更新，不能归为“训练 LLM 设计器”。

## 7. 论文内定位

§3.2–3.4；Table 1；Figure 1；Tables 2–7；§4.4 Table 8；Appendix Algorithm 1、Figure 2。
