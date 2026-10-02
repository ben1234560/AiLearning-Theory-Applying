"""第四章原理演示：动作序列张量、边界掩码与滚动执行。没有训练 ACT。"""

import argparse
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def expert_action(x, goal):
    return float(np.clip(0.5 * (goal - x), -1, 1))


def predict_chunk(x, goal, horizon):
    """从当前观察预测 H 个位移动作；内部预测不推进真实环境。"""
    predicted_x = x
    actions = []
    for _ in range(horizon):
        action = expert_action(predicted_x, goal)
        actions.append([action])
        predicted_x += action
    return np.array(actions)  # (H, 1)


def make_windows(episode_actions, starts, horizon):
    """每个窗口只从当前回合取动作；不足 H 步时补零并标记为无效。"""
    actions = np.asarray(episode_actions)
    targets = np.zeros((len(starts), horizon, actions.shape[1]))
    valid = np.zeros((len(starts), horizon), dtype=bool)
    for sample, start in enumerate(starts):
        length = min(horizon, len(actions) - start)
        targets[sample, :length] = actions[start:start + length]
        valid[sample, :length] = True
    return targets, valid


def masked_mse(prediction, target, valid):
    squared = (prediction - target) ** 2
    return float((squared * valid[..., None]).sum() / (valid.sum() * target.shape[-1]))


def rollout(horizon, execute_steps, steps, disturbance):
    x, goal = 2.0, 5.0
    positions, records, replans = [x], [], []
    pending = []
    for t in range(steps):
        if not pending:
            pending = predict_chunk(x, goal, horizon)[:execute_steps, 0].tolist()
            replans.append(t)
        action = pending.pop(0)
        new_x = x + action
        if t == 1:
            new_x += disturbance  # 同一个时间步、同一个外部扰动
        records.append({"t": t, "x": x, "action": action, "next_x": new_x})
        x = new_x
        positions.append(x)
    final_error = abs(goal - x)
    return {
        "H": horizon, "E": execute_steps, "disturbance": disturbance,
        "steps": steps, "replans": replans, "policy_calls": len(replans),
        "first_replan_after_disturbance": next((t for t in replans if t >= 2), None),
        "action_at_t2": records[2]["action"],
        "final_error": final_error, "success": final_error < 0.1,
        "positions": positions, "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--disturbance", type=float, default=-0.8)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results/chapter04")
    args = parser.parse_args()
    if args.chunk_size < 1 or args.steps < 3:
        parser.error("chunk-size 至少为 1，steps 至少为 3")
    horizons = args.chunk_size
    execute_options = sorted({1, min(2, horizons), horizons})

    # 第一章 6 步专家轨迹；尾部窗口不能连接到另一个回合。
    actions = predict_chunk(2.0, 5.0, 6)
    targets, valid = make_windows(actions, [0, 5], horizons)
    predictions = targets + 0.1
    predictions[~valid] = 99  # 故意把补齐位置设为大值，检查其确实不计入损失。
    loss = masked_mse(predictions, targets, valid)
    assert np.isclose(loss, 0.01)
    assert targets.shape == (2, horizons, 1)

    conditions = [("no disturbance", 0.0), ("same disturbance", args.disturbance)]
    runs = [rollout(horizons, e, args.steps, d) for _, d in conditions for e in execute_options]
    if horizons == 4 and args.disturbance == -0.8:
        disturbed = {r["E"]: r for r in runs if r["disturbance"] == -0.8}
        assert np.isclose(disturbed[1]["action_at_t2"], 0.9)
        assert np.isclose(disturbed[4]["action_at_t2"], 0.5)

    args.output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "type": "principle_demo_no_act_training", "python": platform.python_version(),
        "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        "H": horizons, "steps": args.steps, "disturbance": args.disturbance,
        "initial_x": 2.0, "goal": 5.0, "success_tolerance": 0.1,
        "action_semantics": "1D displacement per step, range [-1,1]",
        "window_shape": list(targets.shape), "valid_mask": valid.astype(int).tolist(),
        "masked_mse_example": loss, "runs": runs,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for group_index, (axis, (label, disturbance)) in enumerate(zip(axes, conditions)):
        group = runs[group_index * len(execute_options):(group_index + 1) * len(execute_options)]
        for run in group:
            axis.plot(run["positions"], marker=".", label=f"E={run['E']}")
        axis.axhline(5.0, color="black", linestyle="--", label="goal")
        if disturbance:
            axis.axvline(2, color="gray", linestyle=":", label="disturbance observed")
        axis.set(xlabel="Executed steps", ylabel="Position", title=f"H={horizons}, {label}")
        axis.legend()
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(args.output / "rollouts.png", dpi=150)
    plt.close(figure)

    print(f"action_windows.shape = {targets.shape}")
    print(f"valid_mask = {valid.astype(int).tolist()}")
    print(f"masked_mse = {loss:.6f}")
    for run in runs:
        print(f"disturbance={run['disturbance']:+.1f}, E={run['E']}, "
              f"calls={run['policy_calls']}, action_t2={run['action_at_t2']:.4f}, "
              f"final_error={run['final_error']:.6f}, success={run['success']}")
    print(f"results: {args.output}")


if __name__ == "__main__":
    main()
