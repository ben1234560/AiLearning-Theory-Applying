"""读取固定版本的官方配置，推导接口形状；不加载网络、权重或运行推理。"""

import argparse
import hashlib
import json
import os
import ssl
import sys
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "lerobot/smolvla_base"
REVISION = "d9f33c94a60fb382c90dea2164c96845bd955e28"
FILES = ("config.json", "policy_preprocessor.json", "policy_postprocessor.json")


def inspect(config, preprocessor, postprocessor):
    """形状来自配置，不是实际张量或一次模型 forward 的输出。"""
    inputs = config["input_features"]
    image_keys = [key for key, feature in inputs.items() if feature["type"] == "VISUAL"]
    normalizer = next(step for step in preprocessor["steps"]
                      if step["registry_name"] == "normalizer_processor")
    tokenizer = next(step["config"] for step in preprocessor["steps"]
                     if step["registry_name"] == "tokenizer_processor")
    processor_image_keys = [key for key, feature in normalizer["config"]["features"].items()
                            if feature["type"] == "VISUAL"]
    action_dim = config["output_features"]["action"]["shape"][0]
    return {
        "verification_scope": "static_config_inspection_only; no model inference",
        "model_id": MODEL_ID, "revision": REVISION,
        "n_obs_steps": config["n_obs_steps"],
        "configured_input_shapes_for_batch_1": {
            key: [1] + value["shape"] for key, value in inputs.items()
        },
        "language_task_key": tokenizer["task_key"],
        "configured_token_ids_shape_for_batch_1": [1, tokenizer["max_length"]],
        "configured_action_chunk_shape_for_batch_1": [1, config["chunk_size"], action_dim],
        "configured_single_action_shape_for_batch_1": [1, action_dim],
        "internal_max_action_dim": config["max_action_dim"],
        "chunk_size": config["chunk_size"], "n_action_steps": config["n_action_steps"],
        "internal_generation_steps": config["num_steps"],
        "normalization_mapping": config["normalization_mapping"],
        "resize_imgs_with_padding": config["resize_imgs_with_padding"],
        "config_image_keys": image_keys, "processor_image_keys": processor_image_keys,
        "camera_key_sets_match": set(image_keys) == set(processor_image_keys),
        "postprocessor_steps": [step["registry_name"] for step in postprocessor["steps"]],
        "action_semantics": "not determined by shape; consult target robot/dataset metadata",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true",
                        help="重新下载固定 revision 的三个小 JSON；不会下载权重")
    parser.add_argument("--snapshot", type=Path, default=ROOT / "results/chapter06/official")
    parser.add_argument("--output", type=Path, default=ROOT / "results/chapter06/smolvla_schema.json")
    args = parser.parse_args()
    args.snapshot.mkdir(parents=True, exist_ok=True)
    if args.refresh:
        # 部分 macOS Python 安装缺少自己的证书包，使用系统 CA；始终验证 HTTPS。
        cafile = os.environ.get("SSL_CERT_FILE")
        if not cafile and sys.platform == "darwin" and Path("/etc/ssl/cert.pem").exists():
            cafile = "/etc/ssl/cert.pem"
        context = ssl.create_default_context(cafile=cafile)
        # 三份都成功读取后再保存，避免半套新快照。
        downloaded = {}
        for filename in FILES:
            url = f"https://huggingface.co/{MODEL_ID}/raw/{REVISION}/{filename}"
            with urlopen(Request(url, headers={"User-Agent": "EmbodiedAI-Chapter06"}),
                         timeout=30, context=context) as response:
                raw = response.read()
            json.loads(raw)
            downloaded[filename] = raw
        for filename, raw in downloaded.items():
            (args.snapshot / filename).write_bytes(raw)
    documents, hashes = [], {}
    for filename in FILES:
        path = args.snapshot / filename
        if not path.exists():
            raise SystemExit(f"缺少 {path}；请运行一次 --refresh 获取固定版本配置。")
        raw = path.read_bytes()
        documents.append(json.loads(raw))
        hashes[filename] = hashlib.sha256(raw).hexdigest()
    summary = inspect(*documents)
    summary["snapshot_sha256"] = hashes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("仅静态配置解析，未加载或执行 SmolVLA")
    print("model:", MODEL_ID, "revision:", REVISION)
    print("configured inputs:", summary["configured_input_shapes_for_batch_1"])
    print("configured language tokens:", summary["configured_token_ids_shape_for_batch_1"])
    print("configured action chunk:", summary["configured_action_chunk_shape_for_batch_1"])
    print("configured single action:", summary["configured_single_action_shape_for_batch_1"])
    print("H/E/internal steps:", summary["chunk_size"], summary["n_action_steps"],
          summary["internal_generation_steps"])
    print("camera key sets match:", summary["camera_key_sets_match"])
    print("summary:", args.output)


if __name__ == "__main__":
    main()
