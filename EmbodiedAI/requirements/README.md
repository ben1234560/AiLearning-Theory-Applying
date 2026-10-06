# 依赖清单

从教程根目录执行安装命令，按自己的设备选择清单。

| 文件 | 用途 |
| --- | --- |
| [requirements-chapter07.txt](requirements-chapter07.txt) | 第七章本机 MPS 验证组合：Torch 2.12.1／Torchvision 0.27.1，以及核心依赖 |
| [requirements-chapter07-core.txt](requirements-chapter07-core.txt) | 两种设备共用的仿真与分析依赖；不包含 Torch／Torchvision |
| [requirements-chapter07-gpu.txt](requirements-chapter07-gpu.txt) | 第七章 CUDA 验证的逻辑版本组合：Torch 2.7.1／Torchvision 0.22.1，以及核心依赖 |

本机安装：

```bash
python -m pip install -r requirements/requirements-chapter07.txt
```

已准备好匹配 Torch／Torchvision 的环境，只安装核心依赖：

```bash
python -m pip install -r requirements/requirements-chapter07-core.txt
```

CUDA wheel 的构建版本还取决于安装索引，完整准备步骤见[第七章](../第七章——跑通一个完整仿真任务.md#51-安装一个与前几章分开的环境)。清单中的 `-r requirements-chapter07-core.txt` 相对于所在文件解析，所以迁移后仍引用本目录中的核心清单。
