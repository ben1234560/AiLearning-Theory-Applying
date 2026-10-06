"""Generate a small GitHub-compatible GIF from an actually recorded simulation MP4."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def make_gif(result_dir: Path, condition: str, seed: int, output_name: str, fps=3, width=320):
    result = json.loads((result_dir / "evaluation.json").read_text())
    episode = next(e for e in result["episodes"] if e["condition"] == condition and e["seed"] == seed)
    source = ROOT / episode["video_path"]
    reader = imageio.get_reader(source)
    source_fps = reader.get_meta_data()["fps"]
    selected = []
    selected_indices = []
    next_index = 0
    source_count = 0
    for index, array in enumerate(reader):
        source_count += 1
        if index >= round(next_index * source_fps / fps):
            image = Image.fromarray(array)
            image = image.resize((width, round(image.height * width / image.width)), Image.Resampling.LANCZOS)
            selected.append(image)
            selected_indices.append(index)
            next_index += 1
    # Always show the true final state, even when temporal subsampling missed it.
    if selected_indices[-1] != source_count - 1:
        image = Image.fromarray(array)
        image = image.resize((width, round(image.height * width / image.width)), Image.Resampling.LANCZOS)
        selected.append(image)
        selected_indices.append(source_count - 1)
    reader.close()
    output = result_dir / "videos" / output_name
    durations = [round(1000 / fps)] * len(selected)
    durations[-1] = 1000
    selected[0].save(output, save_all=True, append_images=selected[1:], loop=0,
                     duration=durations, optimize=True)
    metadata = {
        "condition": condition, "seed": seed, "success": episode["success"],
        "source_video": episode["video_path"], "source_video_fps": source_fps,
        "display_fps": fps, "display_width": width, "gif_frames": len(selected),
        "source_frame_range_inclusive": [0, source_count - 1],
        "selected_source_indices": selected_indices, "final_frame_hold_ms": 1000,
        "environment_step_range_inclusive": [0, episode["n_steps"]],
        "note": "Full recorded episode, temporal subsampling for file size; not a new policy rollout.",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({**metadata, "output": str(output), "bytes": output.stat().st_size}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/chapter07")
    parser.add_argument("--condition", default="E8")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--name", default="representative_success.gif")
    args = parser.parse_args()
    make_gif(args.results.resolve(), args.condition, args.seed, args.name)
