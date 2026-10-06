"""Real PushT physics rollout using the pinned official pretrained DiffusionPolicy.

This script performs inference only. It does not train a policy or choose test seeds
based on observed successes. It writes raw trajectories before aggregate analysis.
"""
from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timedelta, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import torch

from chapter07_setup import DEFAULT_CACHE, LEROBOT_COMMIT, MODEL_REPO, MODEL_REVISION, WEIGHTS_SHA256, sha256

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_policy(cache: Path, device: str):
    source = cache / f"lerobot-{LEROBOT_COMMIT}"
    if not source.exists() or not (cache / "model/model.safetensors").exists():
        raise SystemExit("Run python demo/chapter07_setup.py first (downloads ~1.05 GB).")
    sys.path.insert(0, str(source))
    import draccus
    from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from safetensors.torch import load_model

    raw = json.loads((cache / "model/config.json").read_text())
    fields = {field.name for field in dataclasses.fields(DiffusionConfig)}
    # Current Hub config adds runtime fields absent from the training commit's dataclass.
    # They do not change architecture, preprocessing, or sampling weights.
    extra = {key: raw.pop(key) for key in list(raw) if key not in fields}
    if set(extra) - {"type", "device", "use_amp"}:
        raise ValueError(f"Unexpected config fields: {extra}")
    config = draccus.decode(DiffusionConfig, raw)
    policy = DiffusionPolicy(config)
    missing, unexpected = load_model(policy, str(cache / "model/model.safetensors"), strict=True)
    assert not missing and not unexpected
    policy.eval().to(device)
    return policy, extra


def to_batch(observation: dict, device: str) -> dict:
    # HWC uint8 -> BCHW float32 in [0,1]. Official policy then normalizes image/state.
    return {
        "observation.image": torch.from_numpy(observation["pixels"].copy())
        .permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0,
        "observation.state": torch.from_numpy(observation["agent_pos"].copy())
        .unsqueeze(0).float().to(device),
    }


def sync(device: str):
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def state_step(env, info: dict, t: int, action, reward: float, terminated: bool,
               truncated: bool, replan: bool = False, inference_seconds: float = 0.0) -> dict:
    agent = np.asarray(info["pos_agent"]).tolist()
    pose = np.asarray(info["block_pose"]).tolist()
    return {
        "t": t, "simulation_seconds": t / env.unwrapped.control_hz,
        # Match the environment's privileged state-observation convention for angle.
        # Raw angle is retained in block_pose for exact geometric replay.
        "observation": agent + pose[:2] + [float(pose[2] % (2 * np.pi))],
        "agent_pos": agent, "block_pose": pose,
        "action": None if action is None else np.asarray(action).tolist(),
        "coverage": float(info.get("coverage", env.unwrapped._get_coverage())),
        "reward": float(reward), "terminated": bool(terminated), "truncated": bool(truncated),
        "replan": replan, "inference_seconds": inference_seconds,
    }


def rollout(policy, seed: int, execution_horizon: int, max_steps: int, device: str,
            output: Path, record_video: bool) -> dict:
    import gymnasium as gym
    import gym_pusht  # noqa: F401: registration import
    import imageio.v2 as imageio

    condition = f"E{execution_horizon}"
    name = f"{condition}_seed{seed}"
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array",
                   visualization_width=384, visualization_height=384, max_episode_steps=max_steps)
    observation, info = env.reset(seed=seed)
    policy.config.n_action_steps = execution_horizon
    policy.reset()
    assert policy._queues["action"].maxlen == execution_horizon
    metadata = {
        "condition": condition, "seed": seed, "execution_horizon": execution_horizon,
        "goal_pose": np.asarray(info["goal_pose"]).tolist(),
        "block_local_polygons": [
            [[float(v.x), float(v.y)] for v in shape.get_vertices()]
            for shape in env.unwrapped._block_shapes
        ],
        "agent_radius": float(env.unwrapped.agent.shapes.__iter__().__next__().radius),
        "coordinates": "x right, y down in pygame image; standard x/y rotation formula",
        "success_threshold": float(env.unwrapped.success_threshold), "success_comparison": ">",
        "physics_dt": float(env.unwrapped.dt), "control_hz": env.unwrapped.control_hz,
        "policy_inputs": ["observation.image", "observation.state (agent position only)"],
        "privileged_analysis_state_is_not_policy_input": True,
        "policy_seed_rule": "100000 + environment_seed * 1000 + replan_environment_step",
    }
    frames = [env.render()]
    steps = [state_step(env, info, 0, None, 0, False, False)]
    inference_seconds = 0.0
    replans = 0
    started = time.perf_counter()
    for t in range(max_steps):
        replan = not policy._queues["action"]
        if replan:
            # Identical noise seed whenever both paired conditions replan at the same t.
            torch.manual_seed(100000 + seed * 1000 + t)
            replans += 1
        sync(device)
        inference_started = time.perf_counter()
        with torch.inference_mode():
            action = policy.select_action(to_batch(observation, device)).squeeze(0).cpu().numpy()
        sync(device)
        duration = time.perf_counter() - inference_started
        inference_seconds += duration
        # Action is absolute target position, not displacement. No extra action clipping is applied.
        observation, reward, terminated, truncated, info = env.step(action)
        steps.append(state_step(env, info, t + 1, action, reward, terminated, truncated, replan, duration))
        frames.append(env.render())
        if replan:
            print(f"{name} t={t:03d} block_infer={duration:.3f}s coverage={info['coverage']:.3f}", flush=True)
        if terminated or truncated:
            break
    elapsed = time.perf_counter() - started
    frame_dir = output / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    coverages = [step["coverage"] for step in steps]
    peak = int(np.argmax(coverages))
    for label, index in (("initial", 0), ("peak", peak), ("final", len(frames) - 1)):
        imageio.imwrite(frame_dir / f"{name}_{label}.png", frames[index])
    video_path = None
    if record_video:
        video = output / "videos" / f"{name}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video, frames, fps=env.unwrapped.control_hz, codec="libx264", quality=7)
        video_path = video.relative_to(ROOT).as_posix() if video.is_relative_to(ROOT) else str(video)
    trajectory = output / "trajectories" / f"{name}.json"
    write_json(trajectory, {"metadata": metadata, "steps": steps})
    def relative(path):
        return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
    episode = {
        "condition": condition, "seed": seed, "success": bool(steps[-1]["terminated"]),
        "max_coverage": max(coverages), "final_coverage": coverages[-1],
        "n_steps": len(steps) - 1, "elapsed_seconds": elapsed,
        "inference_seconds": inference_seconds, "replans": replans,
        "trajectory_path": relative(trajectory), "video_path": video_path,
        "initial_frame_path": relative(frame_dir / f"{name}_initial.png"),
        "peak_frame_path": relative(frame_dir / f"{name}_peak.png"),
        "final_frame_path": relative(frame_dir / f"{name}_final.png"),
    }
    env.close()
    print(json.dumps(episode), flush=True)
    return episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/chapter07.json")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=ROOT / "results/chapter07")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--conditions", type=int, nargs="+", help="execution horizons, e.g. 8 4")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    args.cache, args.output = args.cache.expanduser().resolve(), args.output.resolve()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    config = json.loads(args.config.read_text())
    seeds = args.seeds or config["seeds"]
    horizons = args.conditions or config["execution_horizons"]
    max_steps = args.max_steps or config["max_steps"]
    if any(h < 1 or h > config["prediction_window"] - config["observation_history"] + 1 for h in horizons):
        raise ValueError("Execution horizon must lie in [1,15] for this checkpoint.")
    policy, extra = load_policy(args.cache, args.device)
    versions = {}
    for name in ("torch", "torchvision", "gym-pusht", "gymnasium", "pymunk", "diffusers", "draccus", "numpy", "huggingface-hub", "safetensors"):
        versions[name] = importlib.metadata.version(name)
    weights_hash = sha256(args.cache / "model/model.safetensors")
    if weights_hash != WEIGHTS_SHA256:
        raise ValueError("Checkpoint SHA256 differs from the pinned published artifact.")
    metadata = {
        "date": datetime.now(timezone(timedelta(hours=8))).date().isoformat(),
        "training_performed_here": False,
        "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
        "lerobot_commit": LEROBOT_COMMIT, "model_license": "Apache-2.0",
        "weights_sha256": weights_hash,
        "weights_bytes": (args.cache / "model/model.safetensors").stat().st_size,
        "parameters": sum(p.numel() for p in policy.parameters()),
        "device": args.device, "platform": platform.platform(), "python": platform.python_version(),
        "hardware": "Apple M1, 16 GB unified memory" if args.device == "mps" else platform.processor(),
        "gpu_name": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "cuda_version": torch.version.cuda if args.device == "cuda" else None,
        "inference_dtype": "float32", "amp_enabled": False,
        "cudnn_deterministic": True, "tf32_enabled": False if args.device == "cuda" else None,
        "versions": versions, "filtered_runtime_config_fields": extra,
        "seeds": seeds, "execution_horizons": horizons, "max_steps": max_steps,
        "denoising_steps": policy.diffusion.num_inference_steps,
        "scheduler": policy.config.noise_scheduler_type,
        "prediction_window": policy.config.horizon, "observation_history": policy.config.n_obs_steps,
        "success_threshold": .95, "success_comparison": ">",
        "reward_formula": "clip(coverage / 0.95, 0, 1)",
        "coverage_formula": "area(block intersection goal) / area(goal)",
        "action_semantics": "Absolute 2-D pusher target position in [0,512] scene units, 0.1 simulation seconds per action",
        "policy_seed_rule": config["policy_seed_rule"],
        "command": " ".join(sys.argv),
    }
    result_path = args.output / "evaluation.json"
    if result_path.exists():
        if not args.resume:
            raise SystemExit(f"{result_path} exists. Use another --output or --resume.")
        result = json.loads(result_path.read_text())
        for key in ("weights_sha256", "device", "max_steps", "denoising_steps", "seeds", "execution_horizons"):
            if result["metadata"][key] != metadata[key]:
                raise ValueError(f"Resume settings differ at {key}")
    else:
        result = {"metadata": metadata, "episodes": []}
    sources = args.output / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.cache / f"lerobot-{LEROBOT_COMMIT}/LICENSE", sources / "LEROBOT_LICENSE")
    shutil.copyfile(args.cache / "model/README.md", sources / "MODEL_CARD.md")
    shutil.copyfile(args.cache / "model/config.json", sources / "model_config.json")
    pusht_distribution = importlib.metadata.distribution("gym-pusht")
    for file in pusht_distribution.files or []:
        if "LICENSE" in str(file):
            shutil.copyfile(pusht_distribution.locate_file(file), sources / "GYM_PUSHT_LICENSE")
            break
    done = {(ep["condition"], ep["seed"]) for ep in result["episodes"]}
    for horizon in horizons:
        for seed in seeds:
            if (f"E{horizon}", seed) not in done:
                result["episodes"].append(rollout(policy, seed, horizon, max_steps, args.device, args.output, not args.no_video))
                write_json(result_path, result)
    result["metadata"]["completed_episodes"] = len(result["episodes"])
    write_json(result_path, result)


if __name__ == "__main__":
    main()
