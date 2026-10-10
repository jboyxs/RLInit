"""Train or evaluate the first fixed-base TidyBot reaching task."""

import argparse
import json
from pathlib import Path

from rlinit.algorithmhand.PPO import evaluate_ppo, train_ppo
from rlinit.algorithmhand.reach_env import DEFAULT_MODEL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/reach_ppo"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--evaluate", type=Path, help="Evaluate a saved .zip instead of training")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true", help="Show training or evaluation in the MuJoCo viewer")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    if args.evaluate:
        print(json.dumps(evaluate_ppo(args.evaluate, args.episodes, args.seed, args.model, args.render, device=args.device), indent=2))
    else:
        train_ppo(args.timesteps, args.output_dir, args.seed, args.model, args.n_steps, args.batch_size, render=args.render, device=args.device)


if __name__ == "__main__":
    main()
