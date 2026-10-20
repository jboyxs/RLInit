"""Train the two-stage SAC recipe with an optional live single-environment viewer.

  uv run --no-sync python train/train_sac_render.py --render
"""
import argparse
import json
import multiprocessing
from pathlib import Path

if __package__:
    from .sac_staged.config import ROOT
    from .sac_staged.pipeline import run
    from .sac_staged.evaluation import evaluate, inspect
else:
    from sac_staged.config import ROOT
    from sac_staged.pipeline import run
    from sac_staged.evaluation import evaluate, inspect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/door_sac_render")
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--press-progress-after", type=int, default=100000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-envs", type=int, choices=(1, 2, 4, 8),
                        help="Default: 1 with --render, otherwise 4")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--opening-device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--render", action="store_true", help="Show training or evaluation")
    parser.add_argument("--trace-output", type=Path, help="Save independent per-step evaluation traces")
    args = parser.parse_args()
    if args.trace_output is not None:
        if args.evaluate is None:
            parser.error("--trace-output requires --evaluate CHECKPOINT")
        inspect(args.evaluate, args.trace_output, episodes=args.episodes or 20,
                seed=args.seed if args.seed is not None else 20000,
                device=args.device or "cpu", render=args.render)
        return
    if args.evaluate:
        metrics = evaluate(args.evaluate, episodes=args.episodes or 20,
                           seed=args.seed if args.seed is not None else 20000,
                           device=args.device or "cpu", render=args.render)
        print(json.dumps(metrics, indent=2))
        return
    n_envs = args.n_envs if args.n_envs is not None else (1 if args.render else 4)
    if args.render and n_envs != 1:
        parser.error("Training with --render requires --n-envs 1; omit --n-envs to select it automatically")
    run(output_dir=args.output_dir, total_timesteps=args.timesteps,
        press_progress_after=args.press_progress_after, seed=args.seed if args.seed is not None else 67,
        n_envs=n_envs, device=args.device or "auto", opening_device=args.opening_device,
        eval_episodes=args.episodes or 5, render=args.render)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
