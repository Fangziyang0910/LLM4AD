# 结果记录

结果按 task 组织，不按 method 或 model 拆目录：

```text
docs/results/<task>/
  结果汇总.md
  搜索曲线.png
```

`结果汇总.md` 内记录参与比较的 method、model、实验配置、重复 run、测试分数和简短分析；只有重复实验和测试评估全部完成后，才更新结果页。新增 task 时建立新的 task 子目录，新增 method 或 model 时补充到对应 task 的结果页中。
