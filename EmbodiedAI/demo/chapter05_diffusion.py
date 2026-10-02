"""第五章原理实验：训练条件噪声预测器，再从随机噪声生成动作块。

这是 CPU 上的合成一维任务，不是完整 Diffusion Policy 或 PushT 实验。
"""

import argparse
import copy
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


HORIZON = 4


def demonstrations(count, seed):
    """先抽独立初始条件，再生成整块示范；不跨回合拼接动作。"""
    generator = torch.Generator().manual_seed(seed)
    conditions = 4 * torch.rand(count, 2, generator=generator) - 2
    x, goal = conditions[:, 0].clone(), conditions[:, 1]
    actions = []
    for _ in range(HORIZON):
        action = (0.5 * (goal - x)).clamp(-1, 1)
        actions.append(action)
        x = x + action
    # 条件与第三章一致，除以 2；位移动作已在 [-1,1]，不用再缩放。
    # 标准动作块 (N,H,1) 省略最后的单例维，输入 MLP 时为 (N,H)。
    return conditions / 2, torch.stack(actions, dim=1)  # (N,2), (N,4)


def schedule(noise_steps):
    """余弦 beta 日程（s=0.008、beta 最大 .999），与官方任务配置区分。"""
    fraction = torch.arange(noise_steps + 1) / noise_steps
    desired = torch.cos((fraction + 0.008) / 1.008 * torch.pi / 2) ** 2
    desired = desired / desired[0]
    beta = (1 - desired[1:] / desired[:-1]).clamp(max=0.999)
    # 索引 0 表示无噪声 A^0；索引 1...K 表示相应扩散时刻。
    return torch.cat([torch.ones(1), torch.cumprod(1 - beta, dim=0)])


def add_noise(clean, noise, k, alpha_bar):
    weight = alpha_bar[k, None]  # (B,1)，广播至动作块 (B,4)
    return weight.sqrt() * clean + (1 - weight).sqrt() * noise


class NoisePredictor(nn.Module):
    def __init__(self, noise_steps):
        super().__init__()
        self.noise_steps = noise_steps
        self.register_buffer("frequencies", torch.arange(1, 9) * torch.pi)
        self.network = nn.Sequential(
            nn.Linear(HORIZON + 2 + 16, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, HORIZON),
        )

    def forward(self, noisy_actions, conditions, k):
        angle = (k[:, None] / self.noise_steps) * self.frequencies[None, :]
        time_embedding = torch.cat([angle.sin(), angle.cos()], dim=1)
        inputs = torch.cat([noisy_actions, conditions, time_embedding], dim=1)
        return self.network(inputs)  # 预测噪声，形状 (B,4)


@torch.no_grad()
def sample(model, conditions, initial_noise, alpha_bar, keep_history=False):
    """DDIM eta=0：推理只用观察条件和初始噪声，不接收专家动作。"""
    noisy = initial_noise.clone()
    history = [noisy[0].tolist()] if keep_history else []
    model.eval()
    for step in range(len(alpha_bar) - 1, 0, -1):
        k = torch.full((len(conditions),), step, dtype=torch.long)
        predicted_noise = model(noisy, conditions, k)
        current, previous = alpha_bar[step], alpha_bar[step - 1]
        predicted_clean = (
            (noisy - (1 - current).sqrt() * predicted_noise) / current.sqrt()
        ).clamp(-1, 1)
        # 最后一步 previous=alpha_bar[0]=1，噪声项变为 0。
        noisy = previous.sqrt() * predicted_clean + (1 - previous).sqrt() * predicted_noise
        if keep_history:
            history.append(noisy[0].tolist())
    return noisy, history


def mse(prediction, target):
    return float(torch.mean((prediction - target) ** 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results/chapter05")
    args = parser.parse_args()
    if args.train_steps < 1:
        parser.error("train-steps 至少为 1")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    noise_steps, batch_size = 100, 256
    alpha_bar = schedule(noise_steps)
    train_c, train_a = demonstrations(4096, args.seed)
    valid_c, valid_a = demonstrations(1024, args.seed + 1)
    # seed+2 已用于开发探索；最终测试保留新种子，不能用于选择模型。
    test_seed = args.seed + 1002
    test_c, test_a = demonstrations(256, test_seed)
    model = NoisePredictor(noise_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_generator = torch.Generator().manual_seed(args.seed + 100)
    validation_generator = torch.Generator().manual_seed(args.seed + 101)
    valid_k = torch.randint(1, noise_steps + 1, (len(valid_c),), generator=validation_generator)
    valid_noise = torch.randn(valid_a.shape, generator=validation_generator)
    valid_noisy = add_noise(valid_a, valid_noise, valid_k, alpha_bar)
    history = []
    best_loss = float("inf")
    started = time.perf_counter()
    for step in range(args.train_steps + 1):
        if step:
            model.train()
            indices = torch.randint(len(train_c), (batch_size,), generator=train_generator)
            clean, conditions = train_a[indices], train_c[indices]
            k = torch.randint(1, noise_steps + 1, (batch_size,), generator=train_generator)
            noise = torch.randn(clean.shape, generator=train_generator)
            noisy = add_noise(clean, noise, k, alpha_bar)
            prediction = model(noisy, conditions, k)
            loss = torch.mean((prediction - noise) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if step % 100 == 0 or step == args.train_steps:
            model.eval()
            with torch.no_grad():
                valid_loss = mse(model(valid_noisy, valid_c, valid_k), valid_noise)
            history.append({"step": step, "batch_noise_mse": float(loss.detach()) if step else None,
                            "validation_noise_mse": valid_loss})
            if valid_loss < best_loss:
                best_loss, best_step = valid_loss, step
                best_state = copy.deepcopy(model.state_dict())
    train_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.eval()

    # 单因素对照：只打乱观察条件；模型、目标标签、初始噪声完全相同。
    inference_generator = torch.Generator().manual_seed(args.seed + 200)
    initial_noise = torch.randn(test_a.shape, generator=inference_generator)
    permutation = torch.randperm(len(test_c), generator=inference_generator)
    predicted_a, correct_history = sample(model, test_c, initial_noise, alpha_bar, True)
    shuffled_a, shuffled_history = sample(model, test_c[permutation], initial_noise, alpha_bar, True)
    assert predicted_a.shape == test_a.shape == (256, HORIZON)
    assert torch.isfinite(predicted_a).all()
    assert predicted_a.abs().max() <= 1
    action_mse, shuffled_mse = mse(predicted_a, test_a), mse(shuffled_a, test_a)
    per_example_mse = torch.mean((predicted_a - test_a) ** 2, dim=1)
    worst_index = int(per_example_mse.argmax())

    # 独立于训练的一个可手算前向加噪例子：alpha_bar=.64。
    example_clean = torch.tensor([[1.0, 0.5]])
    example_noise = torch.tensor([[0.5, -1.0]])
    example_noisy = 0.8 * example_clean + 0.6 * example_noise
    assert torch.allclose(example_noisy, torch.tensor([[1.1, -0.2]]))
    metrics = {
        "type": "trained_conditional_diffusion_principle_demo_not_pusht",
        "device": "cpu", "python": platform.python_version(), "numpy": np.__version__,
        "torch": torch.__version__, "matplotlib": matplotlib.__version__,
        "seed": args.seed, "train_steps": args.train_steps, "best_step": best_step,
        "train_seconds": train_seconds, "parameter_count": sum(p.numel() for p in model.parameters()),
        "split_initial_conditions": {"train": 4096, "validation": 1024, "test": 256},
        "split_seeds": {"train": args.seed, "validation": args.seed + 1, "test": test_seed},
        "training_noise_seed": args.seed + 100, "validation_noise_seed": args.seed + 101,
        "inference_seed": args.seed + 200, "batch_size": batch_size,
        "H": HORIZON, "action_dimension": 1, "noise_steps": noise_steps,
        "noise_schedule": "cosine, s=0.008, beta_max=0.999", "alpha_bar_K": float(alpha_bar[-1]),
        "sampler": "DDIM eta=0, 100 steps; predicted clean actions clipped to [-1,1]",
        "action_semantics": "1D displacement per environment step in [-1,1]",
        "condition_semantics": "[x/2, goal/2], x and goal sampled independently in [-2,2]",
        "train_condition_shape": list(train_c.shape), "train_action_flat_shape": list(train_a.shape),
        "train_action_block_shape": [len(train_a), HORIZON, 1],
        "initial_validation_noise_mse": history[0]["validation_noise_mse"],
        "best_validation_noise_mse": best_loss, "test_action_block_mse": action_mse,
        "shuffled_condition_action_block_mse": shuffled_mse,
        "evaluation_examples_per_group": len(test_c),
        "same_initial_noise_in_both_groups": True,
        "worst_example": {"test_index": worst_index,
                          "x": float(test_c[worst_index, 0] * 2),
                          "goal": float(test_c[worst_index, 1] * 2),
                          "target": test_a[worst_index].tolist(),
                          "predicted": predicted_a[worst_index].tolist(),
                          "action_block_mse": float(per_example_mse[worst_index])},
        "example": {"x": float(test_c[0, 0] * 2), "goal": float(test_c[0, 1] * 2),
                    "target": test_a[0].tolist(), "initial_noise": initial_noise[0].tolist(),
                    "predicted": predicted_a[0].tolist(), "denoising_history": correct_history,
                    "shuffled_condition": (test_c[permutation[0]] * 2).tolist(),
                    "shuffled_prediction": shuffled_a[0].tolist(),
                    "shuffled_denoising_history": shuffled_history},
        "hand_noise_example": {"alpha_bar": 0.64, "clean": [1, 0.5],
                               "noise": [0.5, -1], "noisy": example_noisy[0].tolist()},
        "development_note": "Default schedules and train_steps explored with seed-9 examples; "
                            "cosine/2000-step configuration fixed before final seed-1009 test. "
                            "For other --seed values the final test seed is seed+1002. "
                            "No robot success rate measured. Development records saved separately.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output / "loss_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "H": HORIZON, "noise_steps": noise_steps,
                "noise_schedule": "cosine", "alpha_bar": alpha_bar}, args.output / "noise_predictor.pt")
    figure, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    axes[0].semilogy([r["step"] for r in history],
                     [r["validation_noise_mse"] for r in history], label="fixed validation noise")
    axes[0].axvline(best_step, color="gray", linestyle="--", label="selected model")
    axes[0].set(xlabel="Optimizer steps", ylabel="Noise prediction MSE", title="Actual training")
    axes[0].legend()
    for axis, (title, path) in zip(axes[1:3], [("Correct condition", correct_history),
                                             ("Shuffled condition", shuffled_history)]):
        denoising = np.array(path)
        for action_index in range(HORIZON):
            line, = axis.plot(np.arange(noise_steps, -1, -1), denoising[:, action_index],
                              label=f"action {action_index + 1}")
            axis.axhline(float(test_a[0, action_index]), color=line.get_color(), linestyle="--", alpha=0.5)
        axis.invert_xaxis()
        axis.set(xlabel="Diffusion index k (100 to 0)", ylabel="Noisy action value", title=title)
        axis.legend(fontsize=8)
    axes[3].bar(["correct", "shuffled"], [action_mse, shuffled_mse],
                color=["#287db0", "#cc6942"])
    axes[3].set(ylabel="Held-out action block MSE", title="Same initial noise; 256 examples")
    figure.suptitle("CPU principle demo: dashed lines are reference actions, not inputs to the sampler", fontsize=10)
    figure.tight_layout()
    figure.savefig(args.output / "training_and_denoising.png", dpi=150)
    plt.close(figure)
    for key in ("train_condition_shape", "train_action_block_shape", "parameter_count", "best_step",
                "train_seconds", "alpha_bar_K", "initial_validation_noise_mse",
                "best_validation_noise_mse", "test_action_block_mse",
                "shuffled_condition_action_block_mse"):
        print(f"{key}: {metrics[key]}")
    print(f"results: {args.output}")


if __name__ == "__main__":
    main()
