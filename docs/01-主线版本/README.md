# 01-主线版本：TraceAAD 算法各代设计与演进

本模块纯粹聚焦于 **TraceAAD 算法本身的各代机制设计与演化脉络**。

所有跑榜数据、排行榜、基线对比、模型对比与实验配置规范已独立至 [02-实验结果](../02-实验结果/README.md)。

---

## 模块结构

所有 TraceAAD 版本机制设计平铺直列于本目录，统一以 `TraceAAD-V{版本}-机制设计.md` 命名，无冗余嵌套：

```text
docs/01-主线版本/
├── README.md                      # 算法主线全景导览
├── 机制设计演进.md                 # 跨版本核心机制设计演变脉络
├── TraceAAD-V10.10-机制设计.md     # 最新主线版本
├── TraceAAD-V10.9-机制设计.md
├── TraceAAD-V10.8-机制设计.md
├── TraceAAD-V10.7-机制设计.md
├── TraceAAD-V10.7R-机制设计.md
├── TraceAAD-V10.6-机制设计.md
├── TraceAAD-V10.5-机制设计.md ~ TraceAAD-V10.0-机制设计.md
├── TraceAAD-V9.22-机制设计.md ~ TraceAAD-V9.0-机制设计.md
└── TraceAAD-历史机制探索.md        # V1–V8 早期历史机制探索
```

---

## 核心版本演进

- **[TraceAAD-V10.10-机制设计](TraceAAD-V10.10-机制设计.md)**（当前主线）：基于 V10.9 移除新结构试用与直接子代试错上下文，引入一次有界修复；解决 Fuse 超限后整边裁剪 capfix；sequential informed initialization。
- **[TraceAAD-V10.9-机制设计](TraceAAD-V10.9-机制设计.md)**：思想形成借鉴与精炼，评估失败分类排查。
- **[TraceAAD-V10.8-机制设计](TraceAAD-V10.8-机制设计.md)**：显式近期形成轨迹，父代机会分配。
- **[TraceAAD-V10.7-机制设计](TraceAAD-V10.7-机制设计.md) / [V10.7R](TraceAAD-V10.7R-机制设计.md)**：分层采样与试错经验沉淀。
- **[TraceAAD-V10.6-机制设计](TraceAAD-V10.6-机制设计.md)**：代码优先生成与摘要对齐，父代先行分配。
- **TraceAAD-V10.1 ~ V10.5**：从单纯提示演进到记忆池与轨迹引导基线。
- **TraceAAD-V9 系列**：探索算法行为反馈与初步轨迹引导。
- **[TraceAAD-历史机制探索](TraceAAD-历史机制探索.md)**：V1 ~ V8 早期概念验证原型。

