"""Checkpoint compatibility, deterministic evaluation and physical traces."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import numpy as np
from stable_baselines3 import SAC
import torch

from rlinit.algorithmhand.door_sac_learn_env import TASK, DEFAULT_MODEL, DoorConfig, LearnRewardConfig, SACLearnDoorEnv
from .config import FORMAT, resolve_device

def _saved_reward(spec):
    if spec.get("task") != TASK:
        raise ValueError("Only this standalone home-only SAC task is supported")
    return LearnRewardConfig(**spec["reward_config"])


def _reward_matches(source, target, upgrade_press_progress):
    if upgrade_press_progress:
        if not source.dense_guidance or source.press_progress_only:
            return False
        source = replace(source, press_progress_only=True)
    return source == target


def _validate_source(agent, env, *, allow_gamma_change=False, upgrade_press_progress=False):
    spec = agent.door_environment
    if spec.get("guidance_base_reward") != env.specification().get("guidance_base_reward"):
        raise ValueError("The original guidance reward changed since this checkpoint")
    if spec.get("model_source_sha256") != env.specification()["model_source_sha256"]:
        raise ValueError("Door or robot XML changed since this checkpoint")
    old_config, new_config = dict(spec["config"]), asdict(env.config)
    if allow_gamma_change:
        old_config.pop("gamma"); new_config.pop("gamma")
    if (old_config != new_config or not _reward_matches(_saved_reward(spec), env.reward_config, upgrade_press_progress)
            or agent.observation_space != env.observation_space or agent.action_space != env.action_space):
        raise ValueError("Checkpoint physics, observation or reward configuration differs")
    if getattr(agent, "door_training_reset_curriculum", None) is not None:
        raise ValueError("This recipe accepts home-only online checkpoints")
    if getattr(agent, "door_success_replay_fraction", 0):
        raise ValueError("This recipe uses uniform online replay")


def load_agent(checkpoint, *, device="cpu", model_path=DEFAULT_MODEL, render=False):
    agent = SAC.load(checkpoint, device=resolve_device(device), verbose=0)
    spec = getattr(agent, "door_environment", {})
    expected_format = FORMAT
    if getattr(agent, "door_task_format", None) != expected_format:
        raise ValueError("Unsupported SAC checkpoint format")
    reward = _saved_reward(spec)  # Reject controller, subtask and incompatible experiments.
    env = SACLearnDoorEnv(model_path, config=DoorConfig(**spec["config"]), reward_config=reward,
                         render_mode="human" if render else None)
    try:
        _validate_source(agent, env)
        if agent.gamma != env.config.gamma:
            raise ValueError("Checkpoint gamma differs from saved environment")
        expected_hash = spec.get("scene_sha256")
        if expected_hash is None:
            manifest = Path(checkpoint).parent / "manifest.json"
            if manifest.is_file():
                provenance = json.loads(manifest.read_text(encoding="utf-8"))
                if provenance.get("checkpoint_sha256") == hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest():
                    expected_hash = provenance.get("source_sha256", {}).get("models/door_scene.xml")
        old_path = Path(spec["model_path"])
        if expected_hash is None and old_path.is_file():
            expected_hash = hashlib.sha256(old_path.read_bytes()).hexdigest()
        if expected_hash != env.specification()["scene_sha256"]:
            raise ValueError("Scene XML differs or its source hash cannot be verified")
        return agent, env
    except Exception:
        env.close()
        raise


def evaluate(checkpoint, *, episodes=20, seed=20000, device="cpu", model_path=DEFAULT_MODEL, render=False):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    env = None
    records = []
    try:
        agent, env = load_agent(checkpoint, device=device, model_path=model_path, render=render)
        for offset in range(episodes):
            obs, _ = env.reset(seed=seed + offset)
            total, length = 0.0, 0
            while True:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, done, timeout, info = env.step(action)
                total += reward
                length += 1
                if done or timeout:
                    break
            records.append(dict(seed=seed + offset, success=info["is_success"],
                                grasp=info["grasp_seen"], press=info["press_seen"],
                                pressed_with_stable_grasp=info["pressed_with_stable_grasp"],
                                failure=info["failure_reason"], steps=length, return_=total,
                                final_door_deg=float(np.rad2deg(info["door_angle"])),
                                success_hold_seconds=info["success_hold_seconds"], reset_origin="home"))
    finally:
        if env is not None:
            env.close()
        torch.set_num_threads(previous_threads)
    return dict(episodes=episodes, first_seed=seed, deterministic=True, reset_origins=["home"],
                success_rate=float(np.mean([r["success"] for r in records])),
                grasp_rate=float(np.mean([r["grasp"] for r in records])),
                press_rate=float(np.mean([r["press"] for r in records])), records=records)


def inspect(checkpoint, output, *, episodes=5, seed=30000, device="cpu", render=False, loader=load_agent):
    torch.set_num_threads(1)
    agent, env = loader(checkpoint, device=device, render=render)
    records = []
    try:
        for offset in range(episodes):
            obs, _ = env.reset(seed=seed + offset)
            trajectory = []
            total = 0.
            grasp_steps = longest = door_hold_steps = 0
            pressed_step = None
            for step in range(1, env.config.max_steps + 1):
                action, _ = agent.predict(obs, deterministic=True)
                with torch.no_grad():
                    tensor, _ = agent.policy.obs_to_tensor(obs)
                    action_tensor = torch.as_tensor(action[None], device=agent.device)
                    critic_q = min(float(q.item()) for q in agent.critic(tensor, action_tensor))
                    _, latent_log_std, _ = agent.actor.get_action_dist_params(tensor)
                obs, reward, done, timeout, info = env.step(action)
                total += reward
                grasp_steps = grasp_steps + 1 if info["grasped"] and not info["failure_reason"] else 0
                longest = max(longest, grasp_steps)
                if (grasp_steps * env.dt + 1e-9 >= env.reward_config.stable_grasp_seconds
                        and info["handle_angle"] >= env.reward_config.press_target_angle):
                    pressed_step = pressed_step or step
                if (pressed_step is not None and info["grasped"] and not info["failure_reason"]
                        and info["door_angle"] >= env.config.success_angle):
                    door_hold_steps += 1
                else:
                    door_hold_steps = 0
                trajectory.append(dict(step=step, action=action.tolist(), reward=reward, critic_q=critic_q,
                    latent_log_std=latent_log_std.cpu().numpy().ravel().tolist(),
                    joint_tracking_error=(env.arm_target - env.data.qpos[env.qadr]).tolist(),
                    components=info["reward_components"], grasp=info["grasped"],
                    handle_deg=float(np.rad2deg(info["handle_angle"])),
                    door_deg=float(np.rad2deg(info["door_angle"])),
                    left_N=info["left_contact_N"], right_N=info["right_contact_N"],
                    penetration_m=info["penetration_m"], bad_contact_N=info["bad_contact_N"],
                    alignment=info["alignment"], center_m=info["center_distance_m"],
                    failure=info["failure_reason"]))
                if done or timeout:
                    break
            verified = (door_hold_steps * env.dt + 1e-9 >= env.config.success_hold_seconds
                        and not info["failure_reason"])
            if verified != info["is_success"]:
                raise RuntimeError("Environment and independent trace success disagree")
            finite_return = 0.
            for sample in reversed(trajectory):
                finite_return = sample["reward"] + agent.gamma * finite_return
                sample["finite_discounted_return"] = finite_return
            records.append(dict(seed=seed + offset, success=bool(verified), steps=step,
                return_=total, pressed_step=pressed_step, longest_grasp_steps=longest,
                initial_critic_q=trajectory[0]["critic_q"],
                finite_discounted_return=trajectory[0]["finite_discounted_return"],
                final_door_hold_seconds=door_hold_steps * env.dt, failure=info["failure_reason"],
                max_handle_deg=max(t["handle_deg"] for t in trajectory),
                max_door_deg=max(t["door_deg"] for t in trajectory), trajectory=trajectory))
    finally:
        env.close()
    result = dict(checkpoint=str(Path(checkpoint).resolve()),
        task=agent.door_environment["task"],
        trained_timesteps=agent.num_timesteps,
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        episodes=episodes, first_seed=seed, deterministic=True, reset_origin="home",
        success_rate=float(np.mean([r["success"] for r in records])),
        critic_comparison="Soft stochastic SAC critic versus finite deterministic rollout return; not an equality test",
        records=records)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    return result
