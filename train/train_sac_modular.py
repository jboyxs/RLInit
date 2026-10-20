"""Run the readable modular implementation of the unchanged SAC recipe.

  uv run --no-sync python train/train_sac_modular.py --device cuda --opening-device cpu
  uv run --no-sync python -m train.train_sac_modular --evaluate CHECKPOINT --render
"""
import argparse
import json
import multiprocessing
from pathlib import Path

# Direct script and `python -m` both work, including Windows spawn workers.
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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/door_sac_staged")
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--press-progress-after", type=int, default=100000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-envs", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--opening-device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--render", action="store_true", help="Render evaluation")
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
    else:
        if args.render:
            parser.error("Use --evaluate CHECKPOINT --render to inspect this parallel recipe")
        run(output_dir=args.output_dir, total_timesteps=args.timesteps,
            press_progress_after=args.press_progress_after, seed=args.seed if args.seed is not None else 67,
            n_envs=args.n_envs, device=args.device or "auto", opening_device=args.opening_device,
            eval_episodes=args.episodes or 5)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
