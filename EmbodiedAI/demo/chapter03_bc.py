"""行为克隆原理实验：从第一章人工策略的示范中学习一维位移动作。"""

import argparse
import copy
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn


def expert(x, goal):
    return max(-1.0, min(1.0, 0.5 * (goal - x)))


def demonstrations(initials):
    """每个初始条件生成完整轨迹；先划分回合，再展开样本。"""
    observations, actions = [], []
    for x, goal in initials:
        for _ in range(12):
            action = expert(x, goal)
            observations.append([x / 2, goal / 2])
            actions.append([action])
            x += action
    return torch.tensor(observations), torch.tensor(actions)


@torch.no_grad()
def rollout(model, x, goal):
    positions = [x]
    for _ in range(20):
        inputs = torch.tensor([[x / 2, goal / 2]], dtype=torch.float32)
        action = float(model(inputs)[0, 0].clamp(-1, 1))
        x += action
        positions.append(x)
    return {"initial_x": positions[0], "goal": goal,
            "positions": positions, "final_error": abs(goal - x),
            "success": abs(goal - x) < 0.1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results/chapter03")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs 必须为正整数")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    initials = [(rng.uniform(-2, 2), rng.uniform(-2, 2)) for _ in range(60)]
    train_x, train_a = demonstrations(initials[:32])
    valid_x, valid_a = demonstrations(initials[32:40])
    test_initials = initials[40:]
    test_x, test_a = demonstrations(test_initials)
    model = nn.Sequential(nn.Linear(2, 32), nn.Tanh(),
                          nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    history = []
    best_loss = float("inf")
    for epoch in range(args.epochs + 1):
        if epoch:
            model.train()
            optimizer.zero_grad()
            prediction = model(train_x)
            loss = loss_fn(prediction, train_a)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss = float(loss_fn(model(train_x), train_a))
            valid_loss = float(loss_fn(model(valid_x), valid_a))
        history.append({"epoch": epoch, "train_mse": train_loss, "valid_mse": valid_loss})
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()

    in_range = [rollout(model, x, goal) for x, goal in test_initials]
    # 唯一变化：测试初始条件整体放大 3 倍，超出训练范围。
    out_range = [rollout(model, 3*x, 3*goal) for x, goal in test_initials]
    with torch.no_grad():
        test_mse = float(loss_fn(model(test_x), test_a))
    examples = {"in_range": in_range[0],
                "outside_range": max(out_range, key=lambda case: case["final_error"])}
    metrics = {
        "type": "principle_demo_not_pusht",
        "python_device": "cpu", "torch_version": torch.__version__,
        "seed": args.seed, "epochs": args.epochs, "best_epoch": best_epoch,
        "split_episodes": {"train": 32, "validation": 8, "test": 20},
        "train_samples": len(train_x), "train_input_shape": list(train_x.shape),
        "train_action_shape": list(train_a.shape),
        "initial_train_mse": history[0]["train_mse"],
        "best_train_mse": history[best_epoch]["train_mse"],
        "best_validation_mse": best_loss, "test_mse": test_mse,
        "evaluation_steps": 20, "success_tolerance": 0.1,
        "in_range_successes": sum(r["success"] for r in in_range),
        "outside_range_successes": sum(r["success"] for r in out_range),
        "eval_episodes_per_group": 20, "examples": examples,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output / "loss_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.output / "rollouts.json").write_text(
        json.dumps({"in_range": in_range, "outside_range": out_range}, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), args.output / "policy.pt")
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].semilogy([r["epoch"] for r in history], [r["train_mse"] for r in history], label="train")
    axes[0].semilogy([r["epoch"] for r in history], [r["valid_mse"] for r in history], label="validation")
    axes[0].axvline(best_epoch, color="gray", linestyle="--", label="selected epoch")
    axes[0].set(xlabel="Epoch", ylabel="Action MSE", title="Supervised training")
    axes[0].legend()
    for axis, (name, case) in zip(axes[1:], examples.items()):
        axis.plot(case["positions"], marker=".", label="learned policy")
        axis.axhline(case["goal"], color="black", linestyle="--", label="goal")
        axis.set(xlabel="Step", ylabel="Position", title=name.replace("_", " "))
        axis.legend()
    figure.tight_layout()
    figure.savefig(args.output / "training_and_rollouts.png", dpi=150)
    plt.close(figure)
    for key in ("train_input_shape", "train_action_shape", "initial_train_mse",
                "best_train_mse", "best_validation_mse", "test_mse",
                "in_range_successes", "outside_range_successes"):
        print(f"{key}: {metrics[key]}")
    print(f"results: {args.output}")


if __name__ == "__main__":
    main()
