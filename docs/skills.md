# 技能安装分类

更新日期：2026-09-06。按 TraceAAD 科研需求筛选，保留原安装位置。现有 100 份技能主文件：项目 82 个、全局用户工具 1 个、系统 6 个、插件 11 个。

## 筛选结果

- Orchestra Research：保留 61 个，覆盖科研构思、算法演化、写作绘图、LLM 微调与强化学习、推理优化和实验工具。删除与当前研究距离较远的语音、视觉、机器人、向量数据库、安全审核、专用模型架构及云平台技能。
- Orchestra 未安装的 ara-compiler、ara-research-manager、ara-rigor-reviewer 已检查：三者围绕 ARA 档案、固定证据字段和分级审查，暂不符合本项目偏好的轻量科研协作，不补装。
- Matt Pocock：保留 17 个原有技能，补装手动调用的 retro，共 18 个。清理依赖未启用工单流程或重复交互的 8 个技能。to-spec 改为生成本地实现说明，移除已删除 setup 的依赖。
- retro 安装来源为 mattpocock/skills 的 skills/in-progress/retro，提交 3cca18b368ae95cdbdebbff572ccafa662551015；SKILL.md 和 agents/openai.yaml 均与上游文件哈希一致。该技能是上游 in-progress 条目，仅手动调用。
- 保留 Academic Humanizer 0.3.3；Ponytail 仅保留插件版。此前删除的 K-Dense、Draw.io 和模板缓存不恢复。
- 保留的技能文件不代表相关外部软件、模型服务或 API 凭据已经配置。系统 review-agent 文件可读取，但未列入本次会话技能目录。

来源：[Matt Pocock](https://github.com/mattpocock/skills)、[Orchestra Research](https://github.com/Orchestra-Research/AI-Research-SKILLs)、[Academic Humanizer](https://github.com/AIScientists-Dev/academic-humanizer)、[Ponytail](https://github.com/DietrichGebert/ponytail)。

## Orchestra Research（61）

- [autoresearch](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/0-autoresearch-skill/SKILL.md)
- [evolving-ai-agents](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/a-evolve/SKILL.md)
- [academic-plotting](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/academic-plotting/SKILL.md)
- [huggingface-accelerate](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/accelerate/SKILL.md)
- [awq-quantization](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/awq/SKILL.md)
- [axolotl](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/axolotl/SKILL.md)
- [evaluating-code-models](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/bigcode-evaluation-harness/SKILL.md)
- [quantizing-models-bitsandbytes](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/bitsandbytes/SKILL.md)
- [brainstorming-research-ideas](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/brainstorming-research-ideas/SKILL.md)
- [creative-thinking-for-research](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/creative-thinking-for-research/SKILL.md)
- [deepspeed](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/deepspeed/SKILL.md)
- [dspy](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/dspy/SKILL.md)
- [optimizing-attention-flash](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/flash-attention/SKILL.md)
- [gguf-quantization](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/gguf/SKILL.md)
- [gptq](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/gptq/SKILL.md)
- [grpo-rl-training](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/grpo-rl-training/SKILL.md)
- [guidance](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/guidance/SKILL.md)
- [hqq-quantization](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/hqq/SKILL.md)
- [huggingface-tokenizers](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/huggingface-tokenizers/SKILL.md)
- [instructor](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/instructor/SKILL.md)
- [knowledge-distillation](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/knowledge-distillation/SKILL.md)
- [langsmith-observability](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/langsmith/SKILL.md)
- [implementing-llms-litgpt](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/litgpt/SKILL.md)
- [llama-cpp](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/llama-cpp/SKILL.md)
- [llama-factory](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/llama-factory/SKILL.md)
- [evaluating-llms-harness](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/lm-evaluation-harness/SKILL.md)
- [long-context](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/long-context/SKILL.md)
- [miles-rl-training](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/miles/SKILL.md)
- [ml-paper-writing](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/ml-paper-writing/SKILL.md)
- [ml-training-recipes](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/ml-training-recipes/SKILL.md)
- [mlflow](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/mlflow/SKILL.md)
- [model-merging](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/model-merging/SKILL.md)
- [model-pruning](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/model-pruning/SKILL.md)
- [nemo-evaluator-sdk](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/nemo-evaluator/SKILL.md)
- [nnsight-remote-interpretability](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/nnsight/SKILL.md)
- [openrlhf-training](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/openrlhf/SKILL.md)
- [outlines](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/outlines/SKILL.md)
- [peft-fine-tuning](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/peft/SKILL.md)
- [phoenix-observability](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/phoenix/SKILL.md)
- [presenting-conference-talks](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/presenting-conference-talks/SKILL.md)
- [pytorch-fsdp2](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/pytorch-fsdp2/SKILL.md)
- [pytorch-lightning](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/pytorch-lightning/SKILL.md)
- [pyvene-interventions](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/pyvene/SKILL.md)
- [ray-data](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/ray-data/SKILL.md)
- [ray-train](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/ray-train/SKILL.md)
- [sparse-autoencoder-training](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/saelens/SKILL.md)
- [sentencepiece](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/sentencepiece/SKILL.md)
- [sglang](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/sglang/SKILL.md)
- [simpo-training](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/simpo/SKILL.md)
- [slime-rl-training](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/slime/SKILL.md)
- [speculative-decoding](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/speculative-decoding/SKILL.md)
- [experiment-tracking-swanlab](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/swanlab/SKILL.md)
- [systems-paper-writing](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/systems-paper-writing/SKILL.md)
- [tensorboard](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/tensorboard/SKILL.md)
- [torchforge-rl-training](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/torchforge/SKILL.md)
- [transformer-lens-interpretability](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/transformer-lens/SKILL.md)
- [fine-tuning-with-trl](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/trl-fine-tuning/SKILL.md)
- [unsloth](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/unsloth/SKILL.md)
- [verl-rl-training](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/verl/SKILL.md)
- [serving-llms-vllm](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/vllm/SKILL.md)
- [weights-and-biases](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/weights-and-biases/SKILL.md)

## Academic Humanizer（1）

- [academic-humanizer](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/academic-humanizer/SKILL.md)

## Matt Pocock（18）

- [code-review](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/code-review/SKILL.md)
- [codebase-design](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/codebase-design/SKILL.md)
- [diagnosing-bugs](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/diagnosing-bugs/SKILL.md)
- [domain-modeling](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/domain-modeling/SKILL.md)
- [grilling](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/grilling/SKILL.md)
- [handoff](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/handoff/SKILL.md)（手动调用）
- [implement](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/implement/SKILL.md)（手动调用）
- [improve-codebase-architecture](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/improve-codebase-architecture/SKILL.md)（手动调用）
- [prototype](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/prototype/SKILL.md)
- [research](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/research/SKILL.md)
- [resolving-merge-conflicts](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/resolving-merge-conflicts/SKILL.md)
- [retro](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/retro/SKILL.md)（手动调用）
- [tdd](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/tdd/SKILL.md)
- [teach](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/teach/SKILL.md)（手动调用）
- [to-spec](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/to-spec/SKILL.md)（手动调用）
- [wait-what](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/wait-what/SKILL.md)（手动调用）
- [wizard](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/wizard/SKILL.md)
- [writing-for-agents](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/writing-for-agents/SKILL.md)

## 全局用户工具（1）

- [find-skills](/home/fang/.agents/skills/find-skills/SKILL.md)

## Codex 系统（6）

- [imagegen](/home/fang/.codex/skills/.system/imagegen/SKILL.md)
- [openai-docs](/home/fang/.codex/skills/.system/openai-docs/SKILL.md)
- [plugin-creator](/home/fang/.codex/skills/.system/plugin-creator/SKILL.md)
- [review-agent](/home/fang/.codex/skills/.system/review-agent/SKILL.md)
- [skill-creator](/home/fang/.codex/skills/.system/skill-creator/SKILL.md)
- [skill-installer](/home/fang/.codex/skills/.system/skill-installer/SKILL.md)

## 插件（11）

- [sites-building](/home/fang/.codex/plugins/cache/openai-bundled/sites/0.1.57/skills/sites-building/SKILL.md)
- [sites-hosting](/home/fang/.codex/plugins/cache/openai-bundled/sites/0.1.57/skills/sites-hosting/SKILL.md)
- [visualize](/home/fang/.codex/plugins/cache/openai-bundled/visualize/1.0.29/skills/visualize/SKILL.md)
- [deep-research](/home/fang/.codex/plugins/cache/openai-curated-remote/deep-research-work/0.1.14/skills/deep-research/SKILL.md)
- [plugin-management](/home/fang/.codex/plugins/cache/openai-curated-remote/plugin-management/0.1.0/skills/plugin-management/SKILL.md)
- [ponytail](/home/fang/.codex/plugins/cache/ponytail/ponytail/4.9.0/skills/ponytail/SKILL.md)
- [ponytail-audit](/home/fang/.codex/plugins/cache/ponytail/ponytail/4.9.0/skills/ponytail-audit/SKILL.md)
- [ponytail-debt](/home/fang/.codex/plugins/cache/ponytail/ponytail/4.9.0/skills/ponytail-debt/SKILL.md)
- [ponytail-gain](/home/fang/.codex/plugins/cache/ponytail/ponytail/4.9.0/skills/ponytail-gain/SKILL.md)
- [ponytail-help](/home/fang/.codex/plugins/cache/ponytail/ponytail/4.9.0/skills/ponytail-help/SKILL.md)
- [ponytail-review](/home/fang/.codex/plugins/cache/ponytail/ponytail/4.9.0/skills/ponytail-review/SKILL.md)

## 本轮移除目录（42）

这些目录按当前项目用途移除，不表示技能本身没有价值。需要时可从对应上游重新安装。

- `audiocraft`
- `blip-2`
- `clip`
- `cosmos-policy`
- `llava`
- `openpi`
- `openvla-oft`
- `segment-anything`
- `stable-diffusion`
- `whisper`
- `chroma`
- `faiss`
- `pinecone`
- `qdrant`
- `sentence-transformers`
- `constitutional-ai`
- `llamaguard`
- `nemo-guardrails`
- `prompt-guard`
- `autogpt`
- `crewai`
- `langchain`
- `llamaindex`
- `mamba`
- `rwkv`
- `nanogpt`
- `megatron-core`
- `torchtitan`
- `moe-training`
- `nemo-curator`
- `lambda-labs`
- `modal`
- `skypilot`
- `tensorrt-llm`
- `ask-matt`
- `grill-me`
- `grill-with-docs`
- `setup-matt-pocock-skills`
- `to-questionnaire`
- `to-tickets`
- `triage`
- `wayfinder`

## Claude Scholar（2）

来源：[Galaxy-Dawn/claude-scholar](https://github.com/Galaxy-Dawn/claude-scholar)，安装提交 `6ed46dac03191c7a734f49ed48b41195012098ff`。两个技能的 17 份原始文件均已与上游哈希核对。

- [review-response](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/review-response/SKILL.md)：审稿回复与 rebuttal。
- [results-analysis](/home/fang/code/LLM4AD/LLM4AD/.agents/skills/results-analysis/SKILL.md)：正式统计与结果分析，设为手动调用；补入共享 research-contract.md 并改为本地引用。

手动调用配置及共享说明引用为本地适配，其他上游内容保留。本次未运行统计分析或安装额外 Python 依赖。
