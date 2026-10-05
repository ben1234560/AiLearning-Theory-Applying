"""第六章原理 Demo：用人工规则连接图像、固定指令和动作块。

没有训练或调用 VLA 模型；没有自然语言理解、物理仿真或机器人控制。
只依赖 NumPy 和 Matplotlib。图像包含目标，当前位置由单独的状态输入提供。
"""

import argparse
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HORIZON, EXECUTE_STEPS = 4, 2
GAIN, LIMIT = 0.5, 0.2
COMMANDS = {"move to red": [1.0, 0.0], "move to blue": [0.0, 1.0]}


def make_image():
    """101×101 RGB；像素 u 向右、v 向下。目标质心落在整数像素上。"""
    image = np.full((101, 101, 3), 245, dtype=np.uint8)
    v, u = np.indices(image.shape[:2])
    image[(u - 75) ** 2 + (v - 50) ** 2 <= 4 ** 2] = [230, 30, 30]
    image[(u - 25) ** 2 + (v - 25) ** 2 <= 4 ** 2] = [30, 70, 230]
    return image


def visual_features(image):
    """颜色阈值→像素质心→归一化坐标；返回两行 [红, 蓝]，每行 [x,y]。"""
    masks = [
        (image[..., 0] > 200) & (image[..., 1] < 100) & (image[..., 2] < 100),
        (image[..., 2] > 200) & (image[..., 0] < 100) & (image[..., 1] < 100),
    ]
    height, width = image.shape[:2]
    targets, pixels = [], []
    for color, mask in zip(["red", "blue"], masks):
        v, u = np.nonzero(mask)
        if not len(u):
            raise ValueError(f"Missing {color} target; no action produced")
        center_u, center_v = float(u.mean()), float(v.mean())
        pixels.append([center_u, center_v])
        targets.append([center_u / (width - 1), 1 - center_v / (height - 1)])
    return np.array(targets), np.array(pixels)  # 两者都是 (2,2)


def encode_command(instruction):
    """严格匹配两条固定英文指令，不支持同义句或其他自然语言。"""
    if instruction not in COMMANDS:
        raise ValueError("Unknown instruction; no action produced")
    return np.array(COMMANDS[instruction])  # (2,)


def predict_chunk(position, goal):
    """内部预测 H 步；逐轴裁剪位移至 [-0.2,0.2]，不会推进实际执行状态。"""
    predicted_position = position.copy()
    actions = []
    for _ in range(HORIZON):
        action = np.clip(GAIN * (goal - predicted_position), -LIMIT, LIMIT)
        actions.append(action)
        predicted_position += action
    return np.array(actions)  # (H,2)


def policy(image, instruction, position):
    targets, pixels = visual_features(image)
    language = encode_command(instruction)
    goal = language @ targets  # (2,) @ (2,2) → (2,)
    return predict_chunk(position, goal), goal, targets, pixels, language


def run_case(image, instruction, initial_position):
    actions, goal, targets, pixels, language = policy(image, instruction, initial_position)
    # 无物理的二维运动学：p_next = p + a；只执行块的前 E 步。
    executed_positions = [initial_position.copy()]
    for action in actions[:EXECUTE_STEPS]:
        executed_positions.append(executed_positions[-1] + action)
    # 目标静止，图像不变；执行后的新位置作为新的状态观察输入。
    observed_position = executed_positions[-1]
    replanned, *_ = policy(image, instruction, observed_position)
    return {
        "instruction": instruction, "language_one_hot": language.tolist(),
        "pixel_centers_red_blue": pixels.tolist(), "targets_red_blue": targets.tolist(),
        "initial_position": initial_position.tolist(), "selected_goal": goal.tolist(),
        "action_chunk": actions.tolist(), "executed_actions": actions[:EXECUTE_STEPS].tolist(),
        "executed_positions": np.array(executed_positions).tolist(),
        "position_after_two_steps": observed_position.tolist(),
        "replanned_chunk": replanned.tolist(),
        "distance_before": float(np.linalg.norm(goal - initial_position)),
        "distance_after_two_steps": float(np.linalg.norm(goal - observed_position)),
    }


def make_plot(image, runs, output):
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image, origin="upper")
    axes[0].set(title="Input image: RGB (101,101,3)", xlabel="u: right (pixels)", ylabel="v: down (pixels)")
    for axis, run, color in zip(axes[1:], runs, ["tab:red", "tab:blue"]):
        targets = np.array(run["targets_red_blue"])
        initial = np.array(run["initial_position"])
        predicted = np.vstack([initial, initial + np.cumsum(run["action_chunk"], axis=0)])
        executed = np.array(run["executed_positions"])
        axis.scatter(targets[:, 0], targets[:, 1], c=["tab:red", "tab:blue"], marker="*", s=200)
        axis.plot(predicted[:, 0], predicted[:, 1], "o--", color="gray", label="Predicted H=4")
        axis.plot(executed[:, 0], executed[:, 1], "o-", color=color, linewidth=3, label="Executed E=2")
        axis.scatter(*initial, marker="s", color="black", s=60, zorder=5, label="Initial position")
        axis.annotate("New state; replan here", xy=executed[-1], xytext=(0.07, 0.12),
                      arrowprops={"arrowstyle": "->", "color": color}, fontsize=9)
        axis.set(title=f'"{run["instruction"]}"', xlabel="x: right (workspace units)", ylabel="y: up (workspace units)",
                 xlim=(0, 1), ylim=(0, 1), aspect="equal")
        axis.grid(alpha=0.2)
        axis.legend(loc="lower right", fontsize=8)
    figure.suptitle("Hand-coded visual + fixed-command interface; no VLA model or training", fontsize=12)
    figure.tight_layout()
    figure.savefig(output / "vla_pipeline.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results/chapter06")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    image, position = make_image(), np.array([0.25, 0.5])
    # 控制变量：图像、起点、增益、限幅、H/E 完全相同，只改变指令。
    runs = [run_case(image, command, position) for command in COMMANDS]
    missing = image.copy()
    missing[(missing[..., 0] > 200) & (missing[..., 1] < 100)] = [245, 245, 245]
    try:
        policy(missing, "move to red", position)
    except ValueError as error:
        failure = {"change": "remove_red_target_only", "instruction": "move to red",
                   "rejected": True, "error": str(error), "actions": None}
    else:
        raise AssertionError("Missing target must stop action production")
    compute_seconds = time.perf_counter() - started

    assert np.allclose(runs[0]["action_chunk"], [[0.2, 0], [0.15, 0], [0.075, 0], [0.0375, 0]])
    assert np.allclose(runs[1]["action_chunk"], [[0, 0.125], [0, 0.0625], [0, 0.03125], [0, 0.015625]])
    plt.imsave(args.output / "toy_rgb.png", image)
    plt.imsave(args.output / "toy_missing_red.png", missing)
    make_plot(image, runs, args.output)
    metrics = {
        "type": "hand_coded_interface_demo_no_vla_training_or_inference",
        "randomness": "none; all inputs and rules deterministic", "device": "CPU",
        "python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        "platform": platform.platform(), "machine": platform.machine(),
        "H": HORIZON, "E": EXECUTE_STEPS, "gain": GAIN, "limit_per_axis": LIMIT,
        "position_semantics": "x-right/y-up normalized workspace [0,1]^2; dimensionless",
        "action_semantics": "2D displacement per abstract step in workspace units; each axis in [-0.2,0.2]",
        "step_duration": "one abstract time unit; not calibrated to robot seconds",
        "shapes": {"image_HWC": list(image.shape), "position": [2], "visual_features": [2, 2],
                   "language_one_hot": [2], "selected_goal": [2], "action_chunk": [HORIZON, 2],
                   "executed_actions": [EXECUTE_STEPS, 2]},
        "controlled_factor": "instruction only", "number_of_cases": 2, "repetitions_per_case": 1,
        "runs": runs, "controlled_failure": failure,
        "compute_seconds_excluding_plots_io": compute_seconds,
        "total_seconds_before_json_write": time.perf_counter() - started,
        "limitations": ["Rules and fixed command lookup; no learned semantics or generalization evidence",
                        "Idealized static targets and exact position observations; no contact physics",
                        "Euclidean distances are illustrative; no robot task success rate measured"],
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"image={image.shape}, actions=({HORIZON}, 2), H={HORIZON}, E={EXECUTE_STEPS}")
    for run in runs:
        print(f'{run["instruction"]}: goal={run["selected_goal"]}, first_action={run["action_chunk"][0]}, '
              f'position_after_2={run["position_after_two_steps"]}, distance={run["distance_after_two_steps"]:.6f}')
    print(f'failure rejected={failure["rejected"]}: {failure["error"]}')
    print(f"results: {args.output}")


if __name__ == "__main__":
    main()
