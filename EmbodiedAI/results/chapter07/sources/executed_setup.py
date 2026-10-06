"""Download pinned public upstream code and weights outside the tutorial repository."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import requests
from huggingface_hub import hf_hub_download

LEROBOT_COMMIT = "3c0a209f9fac4d2a57617e686a7f2a2309144ba2"
MODEL_REPO = "lerobot/diffusion_pusht"
MODEL_REVISION = "84a7c23178445c6bbf7e1a884ff497017910f653"
WEIGHTS_SHA256 = "995d14d35db57d95c35ad9704c3d79c8612b7bc45f3877e5c46c2cdc516856a8"
DEFAULT_CACHE = Path.home() / ".cache" / "embodiedai-chapter07"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(cache: Path) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    source = cache / f"lerobot-{LEROBOT_COMMIT}"
    if not source.exists():
        archive = cache / "lerobot.tar.gz"
        response = requests.get(
            f"https://codeload.github.com/huggingface/lerobot/tar.gz/{LEROBOT_COMMIT}",
            timeout=180,
        )
        response.raise_for_status()
        archive.write_bytes(response.content)
        with tarfile.open(archive) as stream:
            stream.extractall(cache, filter="data")
    for filename in ("config.json", "README.md", "model.safetensors"):
        hf_hub_download(
            MODEL_REPO, filename=filename, revision=MODEL_REVISION,
            local_dir=cache / "model",
        )
    actual_hash = sha256(cache / "model/model.safetensors")
    if actual_hash != WEIGHTS_SHA256:
        raise ValueError("Downloaded weights do not match the pinned checkpoint SHA256.")
    manifest = {
        "lerobot_commit": LEROBOT_COMMIT,
        "lerobot_license": "Apache-2.0",
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_license": "Apache-2.0",
        "source_path": str(source),
        "model_path": str(cache / "model"),
        "weights_bytes": (cache / "model/model.safetensors").stat().st_size,
        "weights_sha256": actual_hash,
    }
    (cache / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    print(json.dumps(prepare(args.cache.expanduser()), indent=2))
