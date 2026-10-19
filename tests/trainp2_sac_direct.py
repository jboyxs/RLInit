"""Train SAC with DoorEnv's features rewarded directly instead of differenced.

Position penalizes summed finger/handle surface distances; orientation rewards
alignment * exp(-distance / 0.08). Weights come from door_direct_env.py defaults.

uv run --no-sync python tests/trainp2_sac_direct.py --timesteps 300000
uv run --no-sync python tests/trainp2_sac_direct.py --evaluate runs/door_sac_direct/sac_door.zip --seed 20000 --render

SAC defaults, observations, actions and success rules match trainp2_sac.py.
"""

import json
import multiprocessing
from pathlib import Path

from rlinit.algorithmhand.door_env import DEFAULT_MODEL, DoorConfig
from trainp2_sac import build_parser, evaluate, train as train_sac


def train(*, output_dir="runs/door_sac_direct", **kwargs):
    return train_sac(output_dir=output_dir, direct_reward=True, **kwargs)


def main():
    parser = build_parser()
    parser.description = __doc__
    parser.set_defaults(output_dir=Path("runs/door_sac_direct"))
    parser.add_argument("--n-envs", type=int, default=1, help="Physics processes (default 1).")
    args = parser.parse_args()
    if args.evaluate:
        metrics = evaluate(args.evaluate, episodes=args.episodes, seed=args.seed,
                           device=args.device, render=args.render, model_path=args.model)
        print(json.dumps(metrics, indent=2))
    else:
        train(total_timesteps=args.timesteps, output_dir=args.output_dir, seed=args.seed,
              n_envs=args.n_envs, model_path=args.model or DEFAULT_MODEL,
              env_config=DoorConfig(max_steps=args.max_steps, action_scale=args.action_scale,
                                    initial_noise=args.initial_noise),
              learning_rate=args.learning_rate, buffer_size=args.buffer_size,
              learning_starts=args.learning_starts, batch_size=args.batch_size,
              train_freq=args.train_freq, gradient_steps=args.gradient_steps,
              gamma=args.gamma, tau=args.tau, ent_coef=args.ent_coef,
              eval_freq=args.eval_freq, eval_episodes=args.episodes,
              render=args.render, device=args.device)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
