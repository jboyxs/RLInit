"""Automate the cold-start recipe used by the successful press-progress pilot.

Phase one learns grasping/pressing from random weights. Phase two resumes its
own paired periodic checkpoint and uniform replay, converting only the press
reward. Both phases use SAC and the unchanged raw eight-action joint interface.

  uv run --no-sync python tests/train_sac_staged.py --output-dir runs/door_sac_staged
  uv run --no-sync python tests/train_sac_staged.py --evaluate CHECKPOINT --episodes 100 --seed 70000

The experimentally promising pilot used CUDA for phase one and CPU for phase
two. This entry automates that boundary; its full cold run still needs validation.
"""

import argparse
from dataclasses import replace
import hashlib
import json
import multiprocessing
from pathlib import Path
import pickle
import shutil
import zipfile

from train_sac_learn import (
    ROOT, SACLearnSettings, LearnRewardConfig, evaluate, train,
)


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def periodic_pair(phase, *, timesteps, n_envs):
    """Reject stale/mismatched callback snapshots before reusing own experience.

    SB3 calls on_step before storing the latest vector transition and before its
    gradient updates. A normal final_model is therefore a different boundary.
    """
    phase = Path(phase)
    model = phase / "latest_model.zip"
    replay = phase / "latest_replay_buffer.pkl"
    with zipfile.ZipFile(model) as archive:
        if archive.testzip() is not None:
            raise ValueError("Incomplete periodic model snapshot")
        data = json.loads(archive.read("data"))
    if data["num_timesteps"] != timesteps:
        raise ValueError("Periodic model does not match the phase boundary")
    with replay.open("rb") as stream:
        buffer = pickle.load(stream)
    count = (buffer.buffer_size if buffer.full else buffer.pos) * buffer.n_envs
    if buffer.n_envs != n_envs or buffer.full or count != timesteps - n_envs:
        raise ValueError("Periodic replay does not match the model's callback boundary")
    metadata = dict(model=str(model), model_sha256=file_hash(model),
                    replay=str(replay), replay_sha256=file_hash(replay),
                    model_timesteps=data["num_timesteps"],
                    gradient_updates=data["_n_updates"], replay_transitions=count)
    return model, replay, metadata


def run(*, output_dir, total_timesteps=500000, press_progress_after=100000,
        seed=67, n_envs=4, device="auto", opening_device="cpu", eval_episodes=5):
    root = Path(output_dir).resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("Output directory must be empty")
    # Both periodic files are written at 20k boundaries. Keeping the prefix
    # inside the replay capacity also makes the snapshot pairing check exact.
    if (n_envs not in (1, 2, 4, 8) or not 0 < press_progress_after < total_timesteps
            or press_progress_after % 20000 or press_progress_after > 300000
            or total_timesteps % n_envs or eval_episodes < 1):
        raise ValueError("Use 1/2/4/8 envs, a 20k-multiple prefix <=300k, and a larger vector-aligned budget")
    root.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__).resolve(), ROOT / "tests/train_sac_learn.py",
               *[ROOT / "src/rlinit/algorithmhand" / name for name in (
                   "door_sac_learn_env.py", "door_sac_learn_replay.py", "door_sac_open_env.py",
                   "door_env.py", "door_direct_env.py", "door_sac_callback.py")],
               ROOT / "models/door_scene.xml", ROOT / "models/objects/door_lock_fixture.xml",
               ROOT / "models/stanford_tidybot/tidybot.xml"]
    source_hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        target = root / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        source_hashes[str(relative)] = file_hash(target)
    manifest = dict(mode="scratch-two-phase-SAC-periodic-pair", external_pretraining=False,
                    demonstrations=False, scripted_actions=False, seed=seed, n_envs=n_envs,
                    requested_timesteps=total_timesteps, press_progress_after=press_progress_after,
                    phase1_requested_device=device, phase2_requested_device=opening_device,
                    source_sha256=source_hashes, phase="learning_grasp_and_press")
    manifest_path = root / "pipeline.json"

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    save_manifest()
    options = dict(seed=seed, n_envs=n_envs, eval_freq=10000, eval_episodes=eval_episodes,
                   settings=replace(SACLearnSettings(), n_steps=3),
                   reward_config=LearnRewardConfig.guided(), save_replay_buffer=True)
    train(output_dir=root / "phase1", total_timesteps=press_progress_after,
          device=device, **options)
    first_config = json.loads((root / "phase1/config.json").read_text(encoding="utf-8"))
    if first_config["actual_timesteps"] != press_progress_after:
        manifest.update(phase="interrupted_in_phase1", actual_timesteps=first_config["actual_timesteps"])
        save_manifest()
        return None
    model, replay, snapshot = periodic_pair(root / "phase1", timesteps=press_progress_after, n_envs=n_envs)
    manifest.update(phase="learning_opening", phase1_snapshot=snapshot)
    save_manifest()
    train(output_dir=root / "phase2", total_timesteps=total_timesteps - press_progress_after,
          device=opening_device, resume_checkpoint=model, replay_buffer_path=replay,
          upgrade_press_progress=True, **options)
    choices = []
    for phase in ("phase1", "phase2"):
        folder = root / phase
        selection = folder / "best/selection.json"
        metrics = json.loads(selection.read_text(encoding="utf-8")) if selection.exists() else {}
        candidate = folder / "best/best_model.zip"
        if not candidate.exists():
            candidate = folder / "final_model.zip"
        # Full task success wins. Prefer phase two on ties across reward scales.
        choices.append((metrics.get("success_rate", 0), phase == "phase2", phase, candidate, selection))
    _, _, selected_phase, selected, selection = max(choices, key=lambda x: x[:2])
    destination = root / "best/best_model.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(selected, destination)
    if selection.exists():
        shutil.copyfile(selection, destination.parent / "selection.json")
    metrics = json.loads((root / selected_phase / "evaluation.json").read_text(encoding="utf-8"))
    metrics["checkpoint"] = str(destination)
    (root / "evaluation.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    actual = json.loads((root / "phase2/config.json").read_text(encoding="utf-8"))["actual_timesteps"]
    manifest.update(phase="training_finished" if actual >= total_timesteps else "interrupted_in_phase2",
                    actual_timesteps=actual, selected_phase=selected_phase,
                    selected_checkpoint=str(destination), selected_sha256=file_hash(destination))
    save_manifest()
    print(f"Selected own SAC policy: {destination}; independent success={metrics['success_rate']:.1%}")
    return destination


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
    args = parser.parse_args()
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
