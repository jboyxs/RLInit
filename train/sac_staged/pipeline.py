"""Two-stage orchestration using the original paired periodic boundary."""
from dataclasses import replace
import json
from pathlib import Path
import pickle
import shutil
import zipfile

from rlinit.algorithmhand.door_sac_learn_env import LearnRewardConfig
from .config import ROOT, SACLearnSettings, file_hash, recipe_sources
from .training import train

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
    sources = [*recipe_sources(),
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
