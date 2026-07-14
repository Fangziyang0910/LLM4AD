# MCTS-AHD Orienteering Construct 三次重复

OP 初始实验因 `max_length_ratio=0.35` 导致 budget 过大、任务无区分度而作废；随后按 ReEvo/DeepACO 标准修正 budget/prize，并用 budget=3 重启 MCTS-AHD 三次重复。

修正后的三个 run 均完成测试评估，结果见 [MCTS-AHD OP 结果](../results/mcts-ahd-qwen36-27b-orienteering-construct.md)。
