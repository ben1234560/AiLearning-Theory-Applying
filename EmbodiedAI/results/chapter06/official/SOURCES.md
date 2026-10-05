# SmolVLA 官方配置快照

核查日期：2026-10-04。这里只保存小型配置与元数据，没有保存模型权重，也没有执行预训练模型推理。

- 模型：`lerobot/smolvla_base`。
- 固定 revision：`d9f33c94a60fb382c90dea2164c96845bd955e28`。
- [固定版本模型卡与文件列表](https://huggingface.co/lerobot/smolvla_base/tree/d9f33c94a60fb382c90dea2164c96845bd955e28)。
- 三个 JSON 保持下载内容，未修改配置值。SHA-256 记录在上一级的 [smolvla_schema.json](../smolvla_schema.json) 中。

| 本地文件 | 官方来源 |
| --- | --- |
| [config.json](config.json) | [模型配置](https://huggingface.co/lerobot/smolvla_base/blob/d9f33c94a60fb382c90dea2164c96845bd955e28/config.json) |
| [policy_preprocessor.json](policy_preprocessor.json) | [预处理配置](https://huggingface.co/lerobot/smolvla_base/blob/d9f33c94a60fb382c90dea2164c96845bd955e28/policy_preprocessor.json) |
| [policy_postprocessor.json](policy_postprocessor.json) | [后处理配置](https://huggingface.co/lerobot/smolvla_base/blob/d9f33c94a60fb382c90dea2164c96845bd955e28/policy_postprocessor.json) |
| [model_metadata.json](model_metadata.json) | [Hugging Face 官方模型 API](https://huggingface.co/api/models/lerobot/smolvla_base?blobs=true) 的选定字段，以及固定版本文件树中的文件大小；这是整理后的元数据摘要 |

模型卡标注 Apache-2.0。附带 [LICENSE-Apache-2.0.txt](LICENSE-Apache-2.0.txt)，为 [LeRobot 固定提交的 LICENSE 原文](https://github.com/huggingface/lerobot/blob/8c920c4270460851cedd2737657584586d3dc66f/LICENSE)，包括其版权与第三方声明；没有修改许可文本。版权归原作者，教程中的说明与图表不属于官方模型发布内容。

配置可用于核对接口声明，但不能确定每个动作分量的机器人语义，也不能证明模型与某个环境或版本已兼容。脚本明确记录相机键名差异；完整接入仍需核对训练数据、预后处理器和机器人定义。
