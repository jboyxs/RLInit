"""Reference-reward SAC with larger Actor and twin Q networks (1024, 1024).

uv run --no-sync python tests/trainp2_sac_reference_large.py --timesteps 300000
uv run --no-sync python tests/trainp2_sac_reference_large.py --n-envs 4 --timesteps 300000
uv run --no-sync python tests/trainp2_sac_reference_large.py --evaluate runs/door_sac_reference_large/sac_door.zip --seed 20000 --render

All reward, environment and algorithm defaults match trainp2_sac_reference.py.
Only the network width and default output directory differ.
"""

import json
import multiprocessing
from pathlib import Path

from rlinit.algorithmhand.door_env import DEFAULT_MODEL, DoorConfig
from rlinit.algorithmhand.door_reference_env import ReferenceRewardConfig
from trainp2_sac import build_parser, evaluate
from trainp2_sac_reference import train as train_reference


def train(*, reward_config=None, output_dir="runs/door_sac_reference_large", n_envs=1, **kwargs):
    return train_reference(n_envs=n_envs, output_dir=output_dir, reward_config=reward_config,
                           policy_kwargs={"net_arch": [1024, 1024]}, **kwargs)


def main():
    parser = build_parser()
    parser.description = __doc__
    parser.set_defaults(output_dir=Path("runs/door_sac_reference_large"), gradient_steps=-1)
    parser.add_argument("--n-envs", type=int, default=1, help="Physics processes; 1 uses the local environment.")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--sparse", action="store_true", help="Only reward successful opening (reference's sparse mode).")
    args = parser.parse_args()
    if args.evaluate:
        metrics = evaluate(args.evaluate, episodes=args.episodes, seed=args.seed,
                           device=args.device, render=args.render, model_path=args.model)
        print(json.dumps(metrics, indent=2))
    else:
        train(total_timesteps=args.timesteps, output_dir=args.output_dir, seed=args.seed,
              n_envs=args.n_envs, torch_threads=args.torch_threads,
              model_path=args.model or DEFAULT_MODEL,
              env_config=DoorConfig(max_steps=args.max_steps, action_scale=args.action_scale,
                                    initial_noise=args.initial_noise),
              reward_config=ReferenceRewardConfig(reward_shaping=not args.sparse, reward_scale=args.reward_scale),
              learning_rate=args.learning_rate, buffer_size=args.buffer_size,
              learning_starts=args.learning_starts, batch_size=args.batch_size,
              train_freq=args.train_freq, gradient_steps=args.gradient_steps,
              gamma=args.gamma, tau=args.tau, ent_coef=args.ent_coef,
              eval_freq=args.eval_freq, eval_episodes=args.episodes, render=args.render, device=args.device)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
