"""SAC with multiple physics processes and one shared learner/replay buffer.

uv run --no-sync python tests/trainp2_sac_parallel.py --n-envs 4 --timesteps 300000
uv run --no-sync python tests/trainp2_sac_parallel.py --evaluate runs/door_sac_parallel/sac_door.zip --render

Timesteps count transitions across all workers. Default gradient_steps=-1
preserves one gradient update per collected transition after warmup.
"""

import json
import multiprocessing
from pathlib import Path

import torch

from rlinit.algorithmhand.door_env import DEFAULT_MODEL, DoorConfig
from trainp2_sac import build_parser, evaluate, train as train_sac


def train(*, n_envs=4, torch_threads=1, gradient_steps=-1,
          output_dir="runs/door_sac_parallel", **kwargs):
    """Limit learner CPU threads to leave cores available for physics workers."""
    if not isinstance(torch_threads, int) or torch_threads < 1:
        raise ValueError("torch_threads must be a positive integer")
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(torch_threads)
        return train_sac(n_envs=n_envs, gradient_steps=gradient_steps, output_dir=output_dir, **kwargs)
    finally:
        torch.set_num_threads(previous_threads)


def main():
    parser = build_parser()
    parser.description = __doc__
    parser.set_defaults(output_dir=Path("runs/door_sac_parallel"), gradient_steps=-1)
    parser.add_argument("--n-envs", type=int, default=4, help="Number of physics processes (default 4).")
    parser.add_argument("--torch-threads", type=int, default=1,
                        help="Learner CPU computation threads; workers each use 1 (default 1).")
    args = parser.parse_args()
    if args.evaluate:
        metrics = evaluate(args.evaluate, episodes=args.episodes, seed=args.seed,
                           device=args.device, render=args.render, model_path=args.model)
        print(json.dumps(metrics, indent=2))
    else:
        train(n_envs=args.n_envs, torch_threads=args.torch_threads,
              total_timesteps=args.timesteps, output_dir=args.output_dir, seed=args.seed,
              model_path=args.model or DEFAULT_MODEL,
              env_config=DoorConfig(max_steps=args.max_steps, action_scale=args.action_scale,
                                    initial_noise=args.initial_noise),
              learning_rate=args.learning_rate, buffer_size=args.buffer_size,
              learning_starts=args.learning_starts, batch_size=args.batch_size,
              train_freq=args.train_freq, gradient_steps=args.gradient_steps,
              gamma=args.gamma, tau=args.tau, ent_coef=args.ent_coef,
              eval_freq=args.eval_freq, eval_episodes=args.episodes, render=args.render, device=args.device)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
