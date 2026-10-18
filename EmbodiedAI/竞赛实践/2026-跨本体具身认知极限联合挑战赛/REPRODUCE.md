# 复现指南

## 环境与数据

历史运行：Linux、4×L40S 48GB、Python3.11、RoboLab0.2.1、IsaacSim5.1.0.0、IsaacLab2.3.2.post1。四卡正式运行约6.8小时（含技术重试），不是其他机器的耗时承诺。数据盘至少保留20GiB余量，题包、资产和输出还需额外空间。

本次共享整理只完成CPU检查；没有重跑仿真、调用模型或验证Docker。

从官方取得`robolab-benchmark30-dual-runner-rgb3-20260818-v2.zip`，SHA256：

```text
9166b1ea123a0eae25d71bb174c8f70c4b8e46ba403b70e9d324f6a5422bcbc4
```

解压一份用于CPU接口测试；`--archive`仍传原始ZIP，运行入口在空目录重新解压。更新题包会被哈希检查拒绝，不可静默替换后沿用17.10分声明。

若只做CPU检查/API预检，不需要安装整个Isaac栈：在仓库根目录执行`python3.11 -m venv .venv`，然后`source .venv/bin/activate`、`pip install -r requirements.txt`，再运行下方检查命令。安装固定Runtime后，正式仿真应切换到`/data/RoboLab/.venv`。

历史题包是否仍可从官方入口取得未在本轮重新下载确认；如果只能拿到不同SHA的新题包，应向组委会索取历史包或另建版本验证，不能绕过校验后沿用本版成绩说明。

## 安装固定Runtime

在已有NVIDIA驱动的Linux机器上，先按官方题包说明安装系统依赖。Ubuntu24.04示例：

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git git-lfs ffmpeg python3-pip python3-venv \
  libglib2.0-0 libgl1 libegl1 libvulkan1 libsm6 libxext6 libxrender1 libxi6 \
  libxrandr2 libxtst6 libnss3 libasound2t64
```

驱动、EGL/Vulkan渲染与网络下载权限也是必要条件。正式入口只支持Linux/NVIDIA；macOS可运行CPU检查。独立安装Runtime：

```bash
python3 -m venv /data/tools-env
/data/tools-env/bin/pip install uv==0.11.31
export GIT_LFS_SKIP_SMUDGE=1
git clone https://github.com/NVLabs/RoboLab.git /data/RoboLab
git -C /data/RoboLab checkout 0aef241fb088ca21bb4ebd24448940ed56620d17
cd /data/RoboLab
/data/tools-env/bin/uv venv --python 3.11
/data/tools-env/bin/uv sync --extra isaac51
source .venv/bin/activate
cd /path/to/ccic-rgb3-baseline-v4
/data/tools-env/bin/uv pip install --python /data/RoboLab/.venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
  'torch==2.7.0+cu128' 'torchvision==0.22.0+cu128' 'torchaudio==2.7.0+cu128' \
  'opencv-python<4.12' -r requirements.txt
/data/tools-env/bin/uv pip check --python /data/RoboLab/.venv/bin/python
```

最后一步固定历史Torch/CUDA与NumPy，匹配的vision/audio版本参考[PyTorch官方安装表](https://pytorch.org/get-started/previous-versions/#v270)；OpenCV上限避免与NumPy1.26冲突。这是安装配方，不是历史全依赖锁文件；若安装检查失败先停止，本次未在全新Linux上验收它。其余间接依赖、驱动与API仍可能变化，不承诺精确同分。

场景资产由官方题包提供，不需要v1/v2/v3、作者缓存、训练数据或私有权重。已核对历史Runtime归档的2599个文件与上面公开commit逐字节相同。入口校验Git commit、关键源码是否改动、实际导入路径及六个历史包版本。保留Git目录；不要改动Runtime代码。

首次实际运行前自行阅读并接受NVIDIA/Omniverse许可，接受后设置`export OMNI_KIT_ACCEPT_EULA=Y`。

运行入口调整单进程渲染及CUDA/Vulkan设备选择，需要对独立虚拟环境有写权限；不修改任务、物理参数、控制适配器或关节限制。不要在无关项目共享的Runtime中安装。

## 配置多模态服务

同名模型有公开接入路线：进入[阿里云百炼/Model Studio](https://www.alibabacloud.com/help/en/model-studio/get-api-key)，按账户要求开通服务，在提供`qwen3.8-omni-flash`的地域/工作空间创建自己的API Key，并复制该空间Chat Completions的API Host。不同地域的Key与Endpoint不能混用。模型接口及`reasoning_effort=low`见[官方Qwen-Omni文档](https://www.alibabacloud.com/help/en/model-studio/qwen-omni)。公开文档核验日期：2026-10-16；未替读者创建账号或验证额度/权限。

参考`configs/env.example`，在Bash设置到当前shell：

```bash
export VLM_BASE_URL='https://your-provider.example/v1'
export VLM_MODEL='qwen3.8-omni-flash'
read -r -s -p 'API key: ' VLM_API_KEY; echo
export VLM_API_KEY
```

BASE_URL填`/v1`层级，代码追加`/chat/completions`。服务需支持三路图像、JSON输出、`reasoning_effort=low`及1400输出token上限；temperature=0，超时70秒，最多3次请求尝试。使用自己有权调用的服务；历史服务地址与密钥不公开。

本仓库没有本地VLM权重。相同模型名不保证后端版本相同；需确认能获取同一模型。API、图像编码和仿真误差使复跑不保证逐字节相同或同分。其他兼容服务可复用方法，但不能据此声称复现17.10。

先验证接入，再租GPU。下面第一条只检查配置格式；第二条**明确调用真实API，可能计费，最多尝试3次**，使用得分策略本身发送三张合成空白图及相同请求参数，验证非流式JSON响应能转为Action26，不启动仿真：

```bash
python tools/check_api.py --runner /data/robolab-benchmark30-dual-runner-rgb3-20260818-v2
python tools/check_api.py --runner /data/robolab-benchmark30-dual-runner-rgb3-20260818-v2 --call
```

本次没有执行真实API预检；公开文档存在不等于此参数组合已在新账号实测。预检通过也仅证明接入，不证明历史模型快照、机器人任务成功率或分数相同。

## CPU与单格检查

```bash
python tools/check.py --runner /data/robolab-benchmark30-dual-runner-rgb3-20260818-v2
python code/reproduce.py \
  --archive /data/robolab-benchmark30-dual-runner-rgb3-20260818-v2.zip \
  --runtime-source /data/RoboLab --work /data/v4-smoke \
  --task RET-011 --embodiment r1_lite --max-steps 600
```

第一条不调用API/GPU；第二条会调用API并启动GPU仿真，只导出单格demo，不是正式60格成绩。

## 完整运行、续跑和输出

执行README全量命令：默认run-id=`baseline-v4`、H=X=1、每格7200步、spaceSaving=true。默认单worker；历史并发为四卡八worker。

技术失败后保持代码、模型、配置相同，停止原运行进程再续跑：

```bash
python code/resume_run.py --work /data/v4-full
```

续跑仅针对`baseline-v4`、7200步、H=X=1。入口以`WORK/run-inputs.json`校验代码/配置哈希、Runtime和模型/Endpoint指纹，拒绝混合版本；不记录密钥或明文Endpoint，换Key但不换服务允许续跑。旧版本缺该记录的目录不能用此入口续跑。修改策略后须用全新空目录与run-id。

- 轨迹：`WORK/robolab-benchmark30-dual-runner-rgb3-20260818-v2/output/baseline-v4/`
- ZIP：同一题包目录下`commit/baseline-v4-OFFICIAL_30X2.zip`，以`WORK/package.log`为准。
- API缓存：`WORK/decision-cache/`，含画面和服务响应，保存在私有目录。
- 日志：`WORK/execute.log`及`WORK/policy-*.log`。

`WORK`中的缓存、服务响应和报错日志可能包含账号相关信息，不要整目录公开上传。正常从原始题包运行不需要`--cache`或`--replay-only`；这两个选项仅用于本人持有的同版本私有缓存诊断。

60格必须使用同一run-id、固定策略与模型，经官方收集验证；不能混合多版本轨迹。本仓库不提供历史轨迹、缓存或“无GPU重建历史分数”的路径。
