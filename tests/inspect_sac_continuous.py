"""Independent physical trace of the XML-native continuous-joint SAC policy."""

import argparse
from pathlib import Path

from inspect_sac_learn import inspect
from train_sac_continuous import load_agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    inspect(args.checkpoint, args.output, episodes=args.episodes, seed=args.seed,
            device=args.device, render=args.render, loader=load_agent)
