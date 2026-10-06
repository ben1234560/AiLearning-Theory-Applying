#!/usr/bin/env python3
"""Analyze real Chapter 7 episodes; never generates or substitutes robot results.

Run from the tutorial directory:
    python3 demo/chapter08_evaluate.py
    python3 demo/chapter08_evaluate.py --input results/chapter07_repeat/evaluation.json --output results/chapter08_repeat

NumPy and Matplotlib are needed only for plots. Source observations are for
post-run diagnosis, and are never passed back into the policy by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def wilson_interval(successes: int, count: int, z: float = 1.96) -> tuple[float, float]:
    """Approximate 95% binomial interval, conditional on iid Bernoulli trials.

    Seeds in this tutorial are a small prespecified evaluation suite. This
    interval indicates sample uncertainty under the iid sampling assumption;
    it is not a generalization guarantee or a confidence interval on a paired
    treatment effect.
    """
    if count <= 0 or not 0 <= successes <= count:
        raise ValueError("Need N > 0 and 0 <= S <= N")
    rate = successes / count
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def validate(episodes: list[dict]) -> dict[str, list[dict]]:
    if not episodes:
        raise ValueError("No recorded episodes; run Chapter 7 first")
    groups: dict[str, list[dict]] = defaultdict(list)
    seen = set()
    for episode in episodes:
        for key in ("condition", "seed", "success", "max_coverage", "final_coverage", "n_steps"):
            if key not in episode:
                raise ValueError(f"Missing episode field: {key}")
        pair = (str(episode["condition"]), int(episode["seed"]))
        if pair in seen:
            raise ValueError(f"Duplicate condition/seed: {pair}")
        seen.add(pair)
        if type(episode["success"]) is not bool:
            raise ValueError(f"success must be a JSON boolean: {pair}")
        maximum, final = float(episode["max_coverage"]), float(episode["final_coverage"])
        if not all(math.isfinite(x) and -1e-8 <= x <= 1 + 1e-8 for x in (maximum, final)):
            raise ValueError(f"Raw coverage outside [0, 1]: {pair}")
        if final > maximum + 1e-8:
            raise ValueError(f"final_coverage exceeds max_coverage: {pair}")
        if int(episode["n_steps"]) < 1:
            raise ValueError(f"Episode has no executed actions: {pair}")
        groups[pair[0]].append(episode)
    return {condition: sorted(rows, key=lambda row: int(row["seed"])) for condition, rows in groups.items()}


def summarize(groups: dict[str, list[dict]]) -> list[dict]:
    result = []
    for condition, rows in groups.items():
        count, successes = len(rows), sum(row["success"] for row in rows)
        low, high = wilson_interval(successes, count)
        result.append({
            "condition": condition, "episodes": count, "successes": successes,
            "success_rate": successes / count, "wilson95": [low, high],
            "mean_max_coverage": float(np.mean([row["max_coverage"] for row in rows])),
            "mean_final_coverage": float(np.mean([row["final_coverage"] for row in rows])),
            "mean_steps": float(np.mean([row["n_steps"] for row in rows])),
            "mean_elapsed_seconds": float(np.mean([row["elapsed_seconds"] for row in rows])) if all("elapsed_seconds" in r for r in rows) else None,
            "seeds": [int(row["seed"]) for row in rows],
        })
    return result


def pair_conditions(groups: dict[str, list[dict]], first: str, second: str) -> dict:
    if first not in groups or second not in groups:
        raise ValueError(f"Paired conditions {first!r}, {second!r} missing; available: {list(groups)}")
    a = {int(row["seed"]): row for row in groups[first]}
    b = {int(row["seed"]): row for row in groups[second]}
    if set(a) != set(b):
        raise ValueError("Paired comparison needs identical seed sets, not unmatched averages")
    rows = []
    for seed in sorted(a):
        rows.append({
            "seed": seed, "first_success": a[seed]["success"], "second_success": b[seed]["success"],
            "first_max_coverage": a[seed]["max_coverage"], "second_max_coverage": b[seed]["max_coverage"],
            "delta_max_coverage": b[seed]["max_coverage"] - a[seed]["max_coverage"],
            "delta_final_coverage": b[seed]["final_coverage"] - a[seed]["final_coverage"],
        })
    return {
        "first_condition": first, "second_condition": second, "direction": "second minus first", "pairs": rows,
        "mean_delta_max_coverage": float(np.mean([row["delta_max_coverage"] for row in rows])),
        "mean_delta_final_coverage": float(np.mean([row["delta_final_coverage"] for row in rows])),
        "first_only_success": sum(row["first_success"] and not row["second_success"] for row in rows),
        "second_only_success": sum(row["second_success"] and not row["first_success"] for row in rows),
        "both_success": sum(row["first_success"] and row["second_success"] for row in rows),
        "both_failure": sum(not row["first_success"] and not row["second_success"] for row in rows),
        "interpretation_limit": "One fixed checkpoint, backend and seed suite; no population causal claim or training generalization proof.",
    }


def write_table(summary: list[dict], paired: dict, output: Path) -> None:
    lines = ["# 第七章实际评测结果", "", "由 `demo/chapter08_evaluate.py` 读取逐回合记录生成，未运行新策略。", "",
             "| 条件 | 成功 / 回合 | 成功率 | 95% Wilson 区间（选修） | 平均最大 coverage | 平均最终 coverage | 平均动作步数 |",
             "| --- | ---: | ---: | --- | ---: | ---: | ---: |"]
    for row in summary:
        lo, hi = row["wilson95"]
        lines.append(f"| {row['condition']} | {row['successes']} / {row['episodes']} | {row['success_rate']:.1%} | [{lo:.1%}, {hi:.1%}] | {row['mean_max_coverage']:.4f} | {row['mean_final_coverage']:.4f} | {row['mean_steps']:.1f} |")
    lines.extend(["", f"配对方向：{paired['second_condition']} 减 {paired['first_condition']}；共 {len(paired['pairs'])} 对同 seed 初始条件。",
                  f"平均最大 coverage 差：{paired['mean_delta_max_coverage']:+.4f}；平均最终 coverage 差：{paired['mean_delta_final_coverage']:+.4f}。",
                  "", "Wilson 区间依赖独立同分布回合假设；教程的少量固定 seed 主要用于复现，不足以代表所有环境或实体机器人。",
                  "相同 seed 控制初始环境和采样种子；执行时域改变之后，策略读取观察及消耗随机数的时机也会改变，轨迹不再相同。",
                  "", "| seed | 第一组成功 | 第二组成功 | 第二组减第一组：最大 coverage |",
                  "| --- | --- | --- | ---: |"])
    for row in paired["pairs"]:
        lines.append(f"| {row['seed']} | {row['first_success']} | {row['second_success']} | {row['delta_max_coverage']:+.4f} |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(summary: list[dict], paired: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2), constrained_layout=True)
    labels = [row["condition"] for row in summary]
    rates = [row["success_rate"] for row in summary]
    errors = np.array([[row["success_rate"] - row["wilson95"][0], row["wilson95"][1] - row["success_rate"]] for row in summary]).T
    axes[0].bar(labels, rates, color=["#3979b9", "#e68a42"][:len(labels)])
    axes[0].errorbar(labels, rates, yerr=errors, fmt="none", ecolor="#333", capsize=5)
    axes[0].set(ylabel="Success rate", ylim=(0, 1.08), title="Episode success (95% Wilson)")
    for i, row in enumerate(summary):
        # Keep the count to the right of the vertical interval, so the line
        # does not cross the text when both are centered on the bar.
        axes[0].text(i + .09, rates[i] + 0.015, f"{row['successes']}/{row['episodes']}", ha="left")
    indices = np.arange(len(summary)); width = .34
    axes[1].bar(indices - width / 2, [r["mean_max_coverage"] for r in summary], width, label="Episode maximum", color="#499ac6")
    axes[1].bar(indices + width / 2, [r["mean_final_coverage"] for r in summary], width, label="Final", color="#f4b367")
    axes[1].set(xticks=indices, xticklabels=labels, ylim=(0, 1.05), ylabel="Raw geometric coverage", title="Maximum versus final overlap")
    axes[1].legend(fontsize=8)
    pairs = paired["pairs"]
    axes[2].bar([str(row["seed"]) for row in pairs], [row["delta_max_coverage"] for row in pairs], color=["#3979b9" if row["delta_max_coverage"] >= 0 else "#e68a42" for row in pairs])
    axes[2].axhline(0, color="#333", lw=.8)
    axes[2].set(ylabel="Difference in maximum coverage", title=f"Matched seeds: {paired['second_condition']} - {paired['first_condition']}")
    axes[2].tick_params(axis="x", rotation=45)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=.15)
    fig.savefig(output, dpi=160); plt.close(fig)


def resolve_path(value: str, input_path: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return input_path.parent / candidate


def verify_trajectories(groups: dict[str, list[dict]], input_path: Path, threshold: float) -> int:
    """Check the saved summary against actual per-step logs and the frozen env contract."""
    checked = 0
    for rows in groups.values():
        for episode in rows:
            path = resolve_path(episode["trajectory_path"], input_path)
            trajectory = json.loads(path.read_text(encoding="utf-8"))
            steps = trajectory["steps"]
            expected_length = int(episode["n_steps"]) + 1
            if len(steps) != expected_length or [s["t"] for s in steps] != list(range(expected_length)):
                raise ValueError(f"Initial state + action-step counts do not align: {path}")
            if steps[0]["action"] is not None:
                raise ValueError(f"Initial reset observation must not contain an executed action: {path}")
            coverage = [float(s["coverage"]) for s in steps]
            if not math.isclose(max(coverage), episode["max_coverage"], abs_tol=1e-10):
                raise ValueError(f"Summary maximum disagrees with step log: {path}")
            if not math.isclose(coverage[-1], episode["final_coverage"], abs_tol=1e-10):
                raise ValueError(f"Summary final coverage disagrees with step log: {path}")
            if bool(steps[-1]["terminated"]) != episode["success"]:
                raise ValueError(f"Summary success disagrees with environment termination: {path}")
            # t=0 comes from reset, which has no reward. Its stored reward=0 is
            # a sentinel, so only actual env.step records implement this formula.
            for step in steps[1:]:
                c = float(step["coverage"])
                expected_reward = min(1.0, max(0.0, c / threshold))
                if not math.isclose(float(step["reward"]), expected_reward, abs_tol=1e-9):
                    raise ValueError(f"Recorded step reward contradicts frozen environment: {path}, t={step['t']}")
                if bool(step["terminated"]) != (c > threshold):
                    raise ValueError(f"Recorded success contradicts strict coverage threshold: {path}, t={step['t']}")
                action = np.asarray(step["action"], dtype=float)
                if action.shape != (2,) or not np.all(np.isfinite(action)):
                    raise ValueError(f"Executed action is not a finite 2-vector: {path}, t={step['t']}")
            checked += 1
    return checked


def paired_prefix_check(groups: dict[str, list[dict]], input_path: Path, first: str, second: str,
                        shared_actions: int = 4, tolerance: float = 1e-5) -> dict:
    """E8/E4 share the first generated chunk; neither replans before four actions.

    This checks the intended one-factor implementation, rather than assuming
    that giving both conditions the same numeric seed controlled everything.
    It does not require trajectories to match after the first E4 replan.
    """
    a = {int(e["seed"]): e for e in groups[first]}
    b = {int(e["seed"]): e for e in groups[second]}
    checks = []
    for seed in sorted(a):
        x = json.loads(resolve_path(a[seed]["trajectory_path"], input_path).read_text())["steps"]
        y = json.loads(resolve_path(b[seed]["trajectory_path"], input_path).read_text())["steps"]
        length = min(shared_actions, len(x) - 1, len(y) - 1)
        initial_difference = float(np.max(np.abs(np.asarray(x[0]["observation"]) - np.asarray(y[0]["observation"]))))
        action_x = np.array([step["action"] for step in x[1:length + 1]], dtype=float)
        action_y = np.array([step["action"] for step in y[1:length + 1]], dtype=float)
        state_x = np.array([step["observation"] for step in x[1:length + 1]], dtype=float)
        state_y = np.array([step["observation"] for step in y[1:length + 1]], dtype=float)
        action_difference = float(np.max(np.abs(action_x - action_y))) if length else 0.0
        state_difference = float(np.max(np.abs(state_x - state_y))) if length else 0.0
        checks.append({
            "seed": seed, "executed_prefix_steps": length,
            "initial_state_max_abs_difference": initial_difference,
            "prefix_action_max_abs_difference": action_difference,
            "prefix_state_max_abs_difference": state_difference,
            "within_tolerance": max(initial_difference, action_difference, state_difference) <= tolerance,
        })
    return {
        "expected_shared_actions": shared_actions, "absolute_tolerance": tolerance,
        "checks": checks, "all_within_tolerance": all(row["within_tolerance"] for row in checks),
        "limit": "Control check only before the first E4 replan; later differences are intended and cannot be uniquely attributed to one internal mechanism.",
    }


def transformed_polygon(vertices: list, pose: list) -> np.ndarray:
    x, y, angle = map(float, pose)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return np.asarray(vertices, dtype=float) @ rotation.T + [x, y]


def failure_frames(episode: dict, steps: list[dict], best_index: int, input_path: Path, output: Path) -> bool:
    """Combine the environment's actual render frames, rather than redraw an imagined outcome."""
    keys = ("initial_frame_path", "peak_frame_path", "final_frame_path")
    if not all(episode.get(key) for key in keys):
        return False
    paths = [resolve_path(episode[key], input_path) for key in keys]
    if not all(path.exists() for path in paths):
        return False
    recorded = [steps[0], steps[best_index], steps[-1]]
    labels = ["Initial", "Best reached", "Final"]
    fig, axes = plt.subplots(1, 3, figsize=(10.3, 3.65), constrained_layout=True)
    for ax, path, step, label in zip(axes, paths, recorded, labels):
        ax.imshow(plt.imread(path)); ax.axis("off")
        ax.set_title(f"{label}: step {step['t']}\ncoverage = {step['coverage']:.4f}", fontsize=10)
    fig.suptitle(f"Actual environment frames: {episode['condition']}, seed {episode['seed']}", fontsize=12)
    fig.savefig(output, dpi=160); plt.close(fig)
    return True


def failure_plot(episode: dict, input_path: Path, output: Path, threshold: float) -> dict:
    trajectory_path = resolve_path(episode["trajectory_path"], input_path)
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    steps = trajectory["steps"]
    coverages = np.array([float(step["coverage"]) for step in steps])
    best_index = int(np.argmax(coverages))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), constrained_layout=True)
    axes[0].plot([step["t"] for step in steps], coverages, color="#3979b9", label="Raw coverage")
    axes[0].axhline(threshold, color="#bd4a42", ls="--", label=f"Success requires > {threshold}")
    axes[0].scatter([steps[best_index]["t"]], [coverages[best_index]], color="#e68a42", zorder=5, label="Best reached")
    axes[0].set(xlabel="Executed action step", ylabel="Raw geometric coverage", ylim=(0, 1.03), title=f"Recorded failure: {episode['condition']}, seed {episode['seed']}")
    axes[0].legend(fontsize=8, framealpha=1)
    agent = np.array([step["agent_pos"] for step in steps], dtype=float)
    block = np.array([step["block_pose"][:2] for step in steps], dtype=float)
    axes[1].plot(agent[:, 0], agent[:, 1], color="#888", alpha=.7, label="Agent path")
    axes[1].plot(block[:, 0], block[:, 1], color="#3979b9", label="Block reference point path")
    axes[1].scatter(*agent[-1], color="#333", s=24)
    metadata = trajectory.get("metadata", {})
    polygons = metadata.get("block_local_polygons")
    goal_pose = metadata.get("goal_pose")
    if polygons and goal_pose:
        from matplotlib.patches import Polygon
        for vertices in polygons:
            axes[1].add_patch(Polygon(transformed_polygon(vertices, goal_pose), facecolor="#85b881", alpha=.3, edgecolor="#397d35"))
            axes[1].add_patch(Polygon(transformed_polygon(vertices, steps[-1]["block_pose"]), facecolor="#4e8cba", alpha=.5, edgecolor="#2b587c"))
        axes[1].scatter([], [], color="#85b881", marker="s", label="Goal silhouette")
        axes[1].scatter([], [], color="#4e8cba", marker="s", label="Final block silhouette")
    elif goal_pose:
        axes[1].scatter(*goal_pose[:2], marker="*", s=100, color="#397d35", label="Goal reference point")
    axes[1].set(xlabel="Environment x", ylabel="Environment y", xlim=(0, 512), ylim=(512, 0), title="Logged state diagnosis (not policy input)")
    axes[1].set_aspect("equal"); axes[1].legend(fontsize=7, loc="best")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output, dpi=160); plt.close(fig)
    frames_created = failure_frames(episode, steps, best_index, input_path, output.parent / "failure_frames.png")
    final_pose = np.asarray(steps[-1]["block_pose"], dtype=float)
    diagnostic = {"final_minus_max_coverage": float(coverages[-1] - coverages[best_index])}
    if goal_pose is not None:
        goal = np.asarray(goal_pose, dtype=float)
        angle_error = (final_pose[2] - goal[2] + math.pi) % (2 * math.pi) - math.pi
        diagnostic.update({
            "final_block_reference_point_distance_to_goal": float(np.linalg.norm(final_pose[:2] - goal[:2])),
            "final_wrapped_orientation_error_radians": float(angle_error),
            "final_wrapped_orientation_error_degrees": float(math.degrees(angle_error)),
        })
    if len(steps) >= 21:
        prior_pose = np.asarray(steps[-21]["block_pose"], dtype=float)
        diagnostic["last_20_actions_block_reference_point_displacement"] = float(np.linalg.norm(final_pose[:2] - prior_pose[:2]))
    return {
        "condition": episode["condition"], "seed": episode["seed"], "success": episode["success"],
        "max_coverage": episode["max_coverage"], "final_coverage": episode["final_coverage"],
        "n_steps": episode["n_steps"], "best_step": steps[best_index]["t"],
        "trajectory_path": str(trajectory_path), "video_path": episode.get("video_path"),
        "frames_plot": str(output.parent / "failure_frames.png") if frames_created else None,
        "observed_diagnostics": diagnostic,
        "selection_rule": "First failed episode in declared condition order, then increasing seed; not selected by severity.",
        "diagnosis_limit": "Coverage and geometry identify observed failure; they do not identify a unique root cause inside the model.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/chapter07/evaluation.json"))
    parser.add_argument("--output", type=Path, default=Path("results/chapter08"))
    parser.add_argument("--first", default="E8")
    parser.add_argument("--second", default="E4")
    args = parser.parse_args()
    if not args.input.exists():
        raise SystemExit(f"No recorded experiment at {args.input}; run Chapter 7 before analysis.")
    source_bytes = args.input.read_bytes()
    source = json.loads(source_bytes)
    groups = validate(source["episodes"])
    planned_seeds = source.get("metadata", {}).get("seeds")
    if planned_seeds is not None:
        expected = set(map(int, planned_seeds))
        for condition in (args.first, args.second):
            recorded = {int(row["seed"]) for row in groups.get(condition, [])}
            if recorded != expected:
                missing = sorted(expected - recorded)
                extra = sorted(recorded - expected)
                raise SystemExit(f"Experiment is incomplete or protocol differs for {condition}: missing={missing}, extra={extra}")
    threshold = float(source.get("metadata", {}).get("success_threshold", .95))
    trajectories_checked = verify_trajectories(groups, args.input, threshold)
    summary = summarize(groups)
    paired = pair_conditions(groups, args.first, args.second)
    prefix = paired_prefix_check(groups, args.input, args.first, args.second)
    if not prefix["all_within_tolerance"]:
        print("Control-check warning: paired prefixes differ; inspect paired_prefix_check in metrics.json before interpreting the comparison.")
    args.output.mkdir(parents=True, exist_ok=True)
    write_table(summary, paired, args.output / "RESULTS.md")
    plot_summary(summary, paired, args.output / "evaluation_summary.png")
    failure = None
    for condition in (args.first, args.second):
        for episode in groups[condition]:
            if not episode["success"]:
                failure = failure_plot(episode, args.input, args.output / "failure_case.png", threshold)
                break
        if failure is not None:
            break
    report = {
        "analysis_type": "Read-only aggregation of actually recorded simulation episodes",
        "input_path": str(args.input), "input_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "trajectories_checked": trajectories_checked,
        "source_metadata": source.get("metadata", {}), "summary": summary, "paired_comparison": paired,
        "paired_prefix_check": prefix,
        "failure_case": failure,
        "limitations": ["No training performed here", "No retraining-seed comparison", "No hardware safety verification", "Fixed small evaluation suite is not proof of broad generalization"],
    }
    (args.output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print((args.output / "RESULTS.md").read_text(encoding="utf-8"))
    print(f"Saved analysis: {args.output}; failure_case={failure is not None}")


if __name__ == "__main__":
    main()
