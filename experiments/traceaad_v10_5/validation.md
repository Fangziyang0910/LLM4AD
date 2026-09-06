# V10.5 首发审查与验证

日期：2026-09-05。范围：V10.5 方法、运行入口、15 路空闲槽位调度器，以及共享 OpenAI API 的元数据读取入口。审查基线为任务开始时 HEAD `92c157693a7454e06452f60d38bc2d631c0ce23d`；其他预先存在的未提交变更不属于本次修改。

## Standards

两个独立审查中的标准轴发现：旧程序转换器可能使评价代码与存档代码不同；恢复指纹未覆盖共享依赖。两项已修复并复查通过。完整模块中的装饰器、尾部全局变量和中间类定义均有实际执行测试；指纹包括关键共享源文件、具体评价/LLM 类和任务契约。无剩余发布阻塞项。

非阻塞维护建议：后续可将 V10.3 与 V10.5 共同的初始化和存储部分提为共享组件，减少继承暴露的旧机制入口。本次不加入无关重构。

## Spec

设计轴确认首发概率、父节点混合分布、形成事件、单次生成和 donor 规则一致。发现并修复四项：完整模块语义、unknown reservation 独立记账、tokenizer 辅助调用日志、原始 parent/frontier/both delta。复查无剩余阻塞项。

## 自动化检查

共 **60 项通过**，由以下两组组成：

```bash
.venv/bin/python -m pytest -q tests/method/test_traceaad_v105.py tests/experiments/test_traceaad_v105_launch.py tests/tools/test_openai_api.py tests/method/test_traceaad_v103_schema.py tests/method/test_traceaad_v104.py tests/experiments/test_traceaad_v104_launch.py
# 55 passed
.venv/bin/python -m pytest -q tests/experiments/test_infra_runner_and_launcher.py
# 5 passed
```

另外通过 V10.5 目录 compileall 和本次共享文件变更的 diff whitespace 检查。全工作区 diff 检查报告的三个尾部空行属于预先存在的 V10.2/V10.3 文件，未在本任务中改动。

覆盖内容包括：算子频率与 Fuse 回退、次数修正和 Pivot 混合分布、完整历史裁剪、donor 可容纳性、完整模块执行、有限 fitness、真实评价预算、持久化响应重用、传输重试选父不变、评价收据和树提交之间多个断点、未知评价阻断，以及槽位分配与停止状态。

## 真实服务验证

首轮 smoke 批次 `smoke_20260905_170624` 五个任务均完成：TSP/CVRP/OP/VRPTW 各 3 次真实评价，OBP 10 次，共 22 次。OBP 使用正式的 8 根设置并完成 Fuse 和 Refine 扩展；其他任务用 2 根缩短验证时间。

三路远端模型调用的精确 prompt token 计数均与服务返回 usage 一致。server1、server3、server3b、本地四个 tokenizer 端点全部验证可用。真实 smoke 中格式错误不消耗评价，算法非法输出消耗评价，行为符合预算定义。

另以 `smoke_pivot_20260905_170926` 在 VRPTW 完成 5 次真实评价，验证 Pivot 的真实调用和评价。合计 27 次 smoke 评价，所有 smoke 单列，不纳入正式实验的 15 × 1000 预算或效果比较。

正式运行状态以 `results/batch_*.json` 为准；本文件记录验证事实，不代替实验结果。
