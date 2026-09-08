# TraceAAD 提示词索引

各文件记录对应版本的提示词与上下文组织，供比较机制和复现实验。它们面向实验中的算法生成模型，原文里的指令不约束科研协作者。需要精确核对时读取对应版本的 `prompts.py`，实际请求见运行日志。

## 近期实现入口

| 版本 | 提示词来源 | 设计记录 |
| --- | --- | --- |
| V10.7 / V10.7R | [实现](../../llm4ad/method/traceaad_v10_7/prompts.py) | [原 V10.7](../methods/TraceAAD-V10.7-机制设计.md)保留统一采样实验身份；[V10.7R](../methods/TraceAAD-V10.7R-机制设计.md)要求自足 Idea 并按设计任务组织证据，收益待验证 |
| V10.6 | [实现](../../llm4ad/method/traceaad_v10_6/prompts.py) | [机制设计](../methods/TraceAAD-V10.6/机制设计.md)：Idea → Code → 独立实现 Idea；此前单次 Code → Summary 批次以实际日志为准 |
| V10.5 | [实现](../../llm4ad/method/traceaad_v10_5/prompts.py) | [机制设计](../methods/TraceAAD-V10.5-机制设计.md) |

## 历史快照

按版本自然顺序排列。快照用于历史比较，不替代对应运行日志中的实际请求。

- [V1](TraceAAD-V1-提示词.md)
- [V2](TraceAAD-V2-提示词.md)
- [V3](TraceAAD-V3-提示词.md)
- [V4](TraceAAD-V4-提示词.md)
- [V5](TraceAAD-V5-提示词.md)
- [V6](TraceAAD-V6-提示词.md)
- [V7](TraceAAD-V7-提示词.md)
- [V8](TraceAAD-V8-提示词.md)
- [V8.3](TraceAAD-V8.3-提示词.md)
- [V9](TraceAAD-V9-提示词.md)
- [V9.1](TraceAAD-V9.1-提示词.md)
- [V9.2](TraceAAD-V9.2-提示词.md)
- [V9.3](TraceAAD-V9.3-提示词.md)
- [V9.4](TraceAAD-V9.4-提示词.md)
- [V9.5](TraceAAD-V9.5-提示词.md)
- [V9.6](TraceAAD-V9.6-提示词.md)
- [V9.7](TraceAAD-V9.7-提示词.md)
- [V9.8](TraceAAD-V9.8-提示词.md)
- [V9.9](TraceAAD-V9.9-提示词.md)
- [V9.10](TraceAAD-V9.10-提示词.md)
- [V9.11](TraceAAD-V9.11-提示词.md)
- [V9.12](TraceAAD-V9.12-提示词.md)
- [V9.13](TraceAAD-V9.13-提示词.md)
- [V9.14](TraceAAD-V9.14-提示词.md)
- [V9.15](TraceAAD-V9.15-提示词.md)
- [V9.16](TraceAAD-V9.16-提示词.md)
- [V9.17](TraceAAD-V9.17-提示词.md)
- [V9.18](TraceAAD-V9.18-提示词.md)
- [V9.19](TraceAAD-V9.19-提示词.md)
- [V9.20](TraceAAD-V9.20-提示词.md)
- [V9.21](TraceAAD-V9.21-提示词.md)
- [V9.22](TraceAAD-V9.22-提示词.md)
- [V10](TraceAAD-V10-提示词.md)
- [V10.1](TraceAAD-V10.1-提示词.md)
- [V10.2](TraceAAD-V10.2-提示词.md)
- [V10.3](TraceAAD-V10.3-提示词.md)
- [V10.4](TraceAAD-V10.4-提示词.md)
