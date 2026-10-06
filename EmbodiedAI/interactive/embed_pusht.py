"""把已运行的仿真与第五章去噪记录内嵌到离线交互页，不产生实验结果。

只读 results/，只写 interactive/index.html。重新评测之后运行本脚本即可更新。
使用 Python 标准库；不联网、不开启物理仿真、不读取模型权重。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_json(html: str, script_id: str, payload: dict) -> str:
    # JSON script 内容也必须转义 </，防止源字段中的 </script> 截断元素。
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    pattern = rf'(<script id="{re.escape(script_id)}" type="application/json">).*?(</script>)'
    updated, count = re.subn(pattern, lambda match: match[1] + serialized + match[2], html, flags=re.S)
    if count != 1:
        raise ValueError(f"Expected exactly one JSON block: {script_id}, got {count}")
    return updated


def main() -> None:
    evaluation_file = ROOT / "results/chapter07/evaluation.json"
    evaluation_bytes = evaluation_file.read_bytes()
    evaluation = json.loads(evaluation_bytes)
    metadata = evaluation["metadata"]
    episodes = []
    geometry = None
    seen = set()
    planned = {(f"E{h}", seed) for h in metadata["execution_horizons"] for seed in metadata["seeds"]}
    for summary in evaluation["episodes"]:
        identity = (summary["condition"], summary["seed"])
        if identity in seen or identity not in planned:
            raise ValueError(f"Duplicated or unplanned episode: {identity}")
        seen.add(identity)
        trajectory_file = ROOT / summary["trajectory_path"]
        trajectory = json.loads(trajectory_file.read_text(encoding="utf-8"))
        scene = trajectory["metadata"]
        if (scene["condition"], scene["seed"]) != identity:
            raise ValueError("Trajectory identity disagrees with the episode summary")
        if not scene["privileged_analysis_state_is_not_policy_input"]:
            raise ValueError("This replay assumes privileged block poses are not policy inputs")
        current_geometry = {
            "width": 512, "height": 512,
            "goal_pose": scene["goal_pose"],
            "block_parts": scene["block_local_polygons"],
            "agent_radius": scene["agent_radius"],
            "coordinates": scene["coordinates"],
        }
        if geometry is not None and geometry != current_geometry:
            raise ValueError("Different geometry across episodes; separate datasets are required")
        geometry = current_geometry
        steps = [{key: frame[key] for key in
                  ("t", "agent_pos", "block_pose", "action", "coverage", "reward", "replan", "simulation_seconds")}
                 for frame in trajectory["steps"]]
        if not steps or steps[0]["t"] != 0 or steps[0]["action"] is not None:
            raise ValueError(f"Missing explicit initial state: {trajectory_file}")
        if any(frame["t"] != index for index, frame in enumerate(steps)):
            raise ValueError("Evaluation budget assumes a complete per-action trajectory")
        if len(steps) - 1 != summary["n_steps"]:
            raise ValueError("Episode summary disagrees with the trajectory length")
        maximum = max(frame["coverage"] for frame in steps)
        if abs(maximum - summary["max_coverage"]) > 1e-9:
            raise ValueError("Recorded maximum coverage disagrees with the trajectory")
        if (maximum > metadata["success_threshold"]) != summary["success"]:
            raise ValueError("Strict success threshold disagrees with the episode summary")
        episode = {key: summary[key] for key in
                   ("condition", "seed", "success", "max_coverage", "final_coverage", "elapsed_seconds",
                    "trajectory_path", "video_path")}
        episode.update(action_steps=summary["n_steps"], steps=steps,
                       coverage_series=[frame["coverage"] for frame in steps])
        episodes.append(episode)
    if not episodes:
        raise ValueError("No actually completed evaluation episodes to embed")
    if metadata["success_comparison"] != ">":
        raise ValueError("This page uses the official strict > threshold")
    expected = len(metadata["seeds"]) * len(metadata["execution_horizons"])
    provenance = (
        f"来源：results/chapter07/evaluation.json；{metadata['date']} 实际运行；"
        f"{metadata['model_repo']}@{metadata['model_revision'][:12]}；"
        f"设备 {metadata['device']}；已完成 {len(episodes)}/{expected} 回合。"
        "本页按真实保存的刚体姿态重绘，不在浏览器求解物理。"
    )
    data = {
        "schema_version": 1, "episodes": episodes, "world": geometry,
        "condition_labels": {f"E{h}": f"每块执行 E = {h} 步" for h in metadata["execution_horizons"]},
        "protocol": {"threshold": metadata["success_threshold"], "budget": metadata["max_steps"],
                     "comparison": ">", "expected_episodes": expected,
                     "description": f"当前嵌入 {len(episodes)}/{expected} 个已完成回合。策略与采样种子规则固定，只改变 E。"},
        "provenance": {"description": provenance, "evaluation_sha256": hashlib.sha256(evaluation_bytes).hexdigest(),
                       "model_repo": metadata["model_repo"], "model_revision": metadata["model_revision"],
                       "lerobot_commit": metadata["lerobot_commit"],
                       "model_url": f"https://huggingface.co/{metadata['model_repo']}/tree/{metadata['model_revision']}",
                       "code_url": f"https://github.com/huggingface/lerobot/tree/{metadata['lerobot_commit']}",
                       "model_license": metadata["model_license"], "policy_inputs": scene["policy_inputs"],
                       "privileged_analysis_state_is_not_policy_input": True},
    }
    diffusion = json.loads((ROOT / "results/chapter05/metrics.json").read_text(encoding="utf-8"))["example"]
    html_file = ROOT / "interactive/index.html"
    html = html_file.read_text(encoding="utf-8")
    html = replace_json(html, "pusht-data", data)
    html = replace_json(html, "diffusion-data", diffusion)
    html_file.write_text(html, encoding="utf-8")
    print(f"Embedded {len(episodes)}/{expected} real episodes and {len(diffusion['denoising_history'])} denoising records")
    print(f"HTML: {html_file} ({html_file.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
