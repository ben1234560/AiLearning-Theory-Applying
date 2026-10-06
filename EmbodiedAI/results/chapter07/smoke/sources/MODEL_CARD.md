---
license: apache-2.0
datasets:
- lerobot/pusht
tags:
- diffusion-policy
- model_hub_mixin
- pytorch_model_hub_mixin
- robotics
pipeline_tag: robotics
---
# Model Card for Diffusion Policy / PushT

Diffusion Policy (as per [Diffusion Policy: Visuomotor Policy
Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)) trained for the `PushT` environment from [gym-pusht](https://github.com/huggingface/gym-pusht).

## How to Get Started with the Model

See the [LeRobot library](https://github.com/huggingface/lerobot) (particularly the [evaluation script](https://github.com/huggingface/lerobot/blob/main/lerobot/scripts/eval.py)) for instructions on how to load and evaluate this model.

## Training Details

Trained with [LeRobot@3c0a209](https://github.com/huggingface/lerobot/tree/3c0a209f9fac4d2a57617e686a7f2a2309144ba2).

The model was trained using [LeRobot's training script](https://github.com/huggingface/lerobot/blob/main/lerobot/scripts/train.py) and with the [pusht](https://huggingface.co/datasets/lerobot/pusht) dataset, using this command:

```bash
python lerobot/scripts/train.py \
    --output_dir=outputs/train/diffusion_pusht \
    --policy.type=diffusion \
    --dataset.repo_id=lerobot/pusht \
    --seed=100000 \
    --env.type=pusht \
    --batch_size=64 \
    --steps=200000 \
    --eval_freq=25000 \
    --save_freq=25000 \
    --wandb.enable=true
```


The training curves may be found at https://wandb.ai/aliberts/lerobot/runs/s7elvf4r.
The current model corresponds to the checkpoint at 175k steps.

## Evaluation

The model was evaluated on the `PushT` environment from [gym-pusht](https://github.com/huggingface/gym-pusht) and compared to a similar model trained with the original [Diffusion Policy code](https://github.com/real-stanford/diffusion_policy). There are two evaluation metrics on a per-episode basis:

- Maximum overlap with target (seen as `eval/avg_max_reward` in the charts above). This ranges in [0, 1].
- Success: whether or not the maximum overlap is at least 95%.

Here are the metrics for 500 episodes worth of evaluation. The "Theirs" column is for an equivalent model trained on the original Diffusion Policy repository and evaluated on LeRobot (the model weights may be found in the [`original_dp_repo`](https://huggingface.co/lerobot/diffusion_pusht/tree/original_dp_repo) branch of this respository).

<blank>|Ours|Theirs
-|-|-
Average max. overlap ratio | 0.955 | 0.957
Success rate for 500 episodes (%) | 65.4 | 64.2

The results of each of the individual rollouts may be found in [eval_info.json](eval_info.json).
It was produced after training with this command:
```bash
python lerobot/scripts/eval.py \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/175000/pretrained_model \
    --output_dir=outputs/eval/diffusion_pusht/175000 \
    --env.type=pusht \
    --eval.n_episodes=500 \
    --eval.batch_size=50 \
    --device=cuda \
    --use_amp=false
```