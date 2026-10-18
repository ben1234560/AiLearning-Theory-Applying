# 2026 具身仿真合成挑战赛：可复现 Baseline

本项目提供两条清晰区分的路径：

1. **路径 A：快速精确复现。** 使用冻结中间包，在无 GPU 环境下重建历史 `67.6000` 候选的同一字节，并做本地校验。
2. **路径 B：从官方输入重建。** 以 `question.zip` 与 `submission_example.zip` 为竞赛数据输入，在阿里云 GPU 上执行可审计的资产生成流程。

源码位于 [public_asset_baseline](public_asset_baseline/README.md)。官方原始数据、视频、模型缓存和任何密钥均不随本项目分发。

## 下载与解压

发布后，从 GitHub Release 下载 `public_asset_baseline_release.zip`。下载链接固定为：

```text
https://github.com/ben1234560/AiLearning-Theory-Applying/releases/download/embodied-simulation-baseline/public_asset_baseline_release.zip
```

解压并进入 Baseline：

```bash
unzip public_asset_baseline_release.zip
cd public_asset_baseline
```

本地待上传的同名文件位于 `release_assets/`；该目录中的 ZIP 被 Git 忽略，待官方核验通过后由仓库维护者在 GitHub Release 页面上传。上传前后均应对照 [CHECKSUMS.md](CHECKSUMS.md)。

## 校验

Release ZIP 的 SHA-256 和历史候选的 MD5/SHA-256 见 [CHECKSUMS.md](CHECKSUMS.md)。解压后，先校验预生成的提交包：

```bash
md5 quick_output/submission.zip
shasum -a 256 quick_output/submission.zip
```

预期值：

```text
MD5:    a220303381c7bb8886684778e34689df
SHA256: 2e979d08d10a785e0c47a4a1ba923131a52c8e53970bb0578f815893585e8d4b
```

再执行结构、纹理引用和轻量物理校验：

```bash
python scripts/validate_submission.py \
  --package quick_output/submission.zip \
  --report quick_output/submission_recheck.json
```

合格结果应为 `valid: true`、`valid_tasks: 34`、`physics_probe_steps: 120`。

## 路径 A：无需 GPU 的快速精确复现

Release 包已附带 `quick_output/submission.zip`，可直接比对或提交。若要从冻结中间包重建它：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
python -m pip install -r requirements-dev.txt

python scripts/quick_reproduce.py --output-root reproduced_quick_output
python scripts/validate_submission.py \
  --package reproduced_quick_output/submission.zip \
  --report reproduced_quick_output/validation_recheck.json
```

`reproduced_quick_output/submission.zip` 的 MD5 和 SHA-256 应与上方预期值一致。路径 A 不访问网络、不使用 GPU；它保证候选文件相同，而未来线上分数仍由平台评测环境决定。

## 路径 B：从官方 ZIP 的 GPU 重建

将官方获取的两份 ZIP 放在私有目录，且不要提交到 Git：

```text
/data/competition_input/question.zip
/data/competition_input/submission_example.zip
```

完整资源选择、许可前提、GPU 宿主机预检与运行方式见 [GPU_BUILD.md](public_asset_baseline/GPU_BUILD.md)。完成许可确认后，在 `public_asset_baseline/` 目录执行：

```bash
export DATA_DIR=/data/competition_input
export OUT_DIR=/data/asset_baseline_output
export ACCEPT_HUNYUAN_LICENSE=yes
bash scripts/run_aliyun.sh
```

路径 B 的输出是独立重建结果，可能因为精度等原因与路径 A 线上分数不一致。
