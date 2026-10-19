"""Train/evaluate the fixed-base door task with Stable-Baselines3 SAC.

uv run --no-sync python tests/trainp2_sac.py --timesteps 300000
uv run --no-sync python tests/trainp2_sac.py --evaluate runs/door_sac/sac_door.zip --seed 20000 --render

Use SB3's default MlpPolicy and algorithm settings unless explicitly overridden.
Reward-shaping gamma follows the actual SAC gamma, including during evaluation.
"""

from dataclasses import asdict, replace
import argparse
from functools import partial
import inspect
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from rlinit.algorithmhand.device import resolve_device
from rlinit.algorithmhand.door_direct_env import TASK as DIRECT_TASK, DirectRewardDoorEnv
from rlinit.algorithmhand.door_sac_grasp_env import (
    TASK as PRECISION_TASK, LEGACY_TASK as LEGACY_PRECISION_TASK,
    GraspTrainingConfig, PrecisionGraspDoorEnv,
)
from rlinit.algorithmhand.door_sac_replay import OwnSuccessReplayBuffer, configure_multistep_replay
from rlinit.algorithmhand.door_env import DEFAULT_MODEL, DoorConfig, DoorEnv
from rlinit.algorithmhand.door_reference_env import (
    TASK as REFERENCE_TASK, ReferenceRewardConfig, ReferenceRewardDoorEnv,
)
from rlinit.algorithmhand.evaluation import final_eval_seed


FORMAT = "rlinit-door-sac-v1"
DEFAULT_MAX_STEPS = 1000
MONITOR_KEYS = ("distance", "is_success", "grasp_seen", "press_seen",
                "door_angle", "max_door_angle", "failure_reason")


def monitor_keys(direct_reward):
    return MONITOR_KEYS + (("stage_success", "reset_origin")
                           if isinstance(direct_reward, GraspTrainingConfig) or direct_reward == "precision" else ())


def make_environment(model_path, config, *, render=False, reference_reward=None, direct_reward=False,
                     training_reset_bank=None, training_reset_probability=0.5,
                     adaptive_training_resets=False):
    if direct_reward and reference_reward is not None:
        raise ValueError("direct_reward and reference_reward cannot be combined")
    kwargs = {"config": config, "render_mode": "human" if render else None}
    if isinstance(direct_reward, GraspTrainingConfig) or direct_reward == "precision":
        environment = PrecisionGraspDoorEnv(model_path,
                                     training_config=direct_reward if isinstance(direct_reward, GraspTrainingConfig) else None,
                                     **kwargs)
        try:
            if training_reset_bank is not None:
                environment.enable_training_resets(training_reset_bank, training_reset_probability,
                                                    adaptive=adaptive_training_resets)
            return environment
        except Exception:
            environment.close()
            raise
    if training_reset_bank is not None:
        raise ValueError("Online-state training resets require the precision joint SAC task")
    if direct_reward:
        # New training uses the reward defaults defined in door_direct_env.py.
        return DirectRewardDoorEnv(model_path, **kwargs)
    if reference_reward is not None:
        return ReferenceRewardDoorEnv(model_path, reward_config=reference_reward, **kwargs)
    return DoorEnv(model_path, **kwargs)


def make_worker(model_path, config, monitor_path, reference_reward=None, direct_reward=False,
                training_reset_bank=None, training_reset_probability=0.5, adaptive_training_resets=False):
    """Construct physics state inside the spawned process, never in the parent."""
    torch.set_num_threads(1)
    return Monitor(make_environment(model_path, config, reference_reward=reference_reward, direct_reward=direct_reward,
                                    training_reset_bank=training_reset_bank,
                                    training_reset_probability=training_reset_probability,
                                    adaptive_training_resets=adaptive_training_resets),
                   str(monitor_path) if monitor_path is not None else None, info_keywords=monitor_keys(direct_reward))


def entropy_coefficient(value):
    """Accept a fixed nonnegative alpha, 'auto', or 'auto_<positive initial>'."""
    if value == "auto":
        return value
    if isinstance(value, str) and value.startswith("auto_"):
        initial = float(value.removeprefix("auto_"))
        if not np.isfinite(initial) or initial <= 0:
            raise ValueError("Automatic entropy coefficient needs a positive finite initial value")
        return value
    fixed = float(value)
    if not np.isfinite(fixed) or fixed < 0:
        raise ValueError("ent_coef must be nonnegative and finite, 'auto', or 'auto_<positive>'")
    return fixed


def sac_settings(**values):
    defaults = inspect.signature(SAC).parameters
    keys = ("learning_rate", "buffer_size", "learning_starts", "batch_size", "train_freq",
            "gradient_steps", "gamma", "tau", "ent_coef", "n_steps")
    unknown = values.keys() - set(keys)
    if unknown:
        raise ValueError(f"Unknown SAC settings: {sorted(unknown)}")
    overrides = {key: value for key, value in values.items() if value is not None}
    if "ent_coef" in overrides:
        overrides["ent_coef"] = entropy_coefficient(overrides["ent_coef"])
    actual = {key: overrides.get(key, defaults[key].default) for key in keys}
    for key in ("buffer_size", "batch_size", "train_freq", "n_steps"):
        if not isinstance(actual[key], int) or actual[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(actual["learning_starts"], int) or actual["learning_starts"] < 0:
        raise ValueError("learning_starts must be a nonnegative integer")
    gradient_steps = actual["gradient_steps"]
    if not isinstance(gradient_steps, int) or (gradient_steps != -1 and gradient_steps < 1):
        raise ValueError("gradient_steps must be positive or -1 (one update per collected step)")
    for key in ("learning_rate", "gamma", "tau"):
        if not np.isfinite(actual[key]) or actual[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if actual["gamma"] > 1 or actual["tau"] > 1:
        raise ValueError("gamma and tau must be in (0, 1]")
    return overrides, actual


def reject_legacy_precision(specification):
    if specification.get("task") == LEGACY_PRECISION_TASK:
        raise ValueError("Legacy precision checkpoint uses removed controller/action filtering semantics; "
                         "start a fresh joint-action SAC run, without its weights or replay buffer")


def collect_online_policy_warmup(agent, steps):
    """Add the current SAC actor's own rollouts without changing its weights."""
    vector_env = agent.get_env()
    observation = vector_env.reset()
    collected = 0
    while collected < steps:
        actions, _ = agent.predict(observation, deterministic=False)
        next_observation, rewards, dones, infos = vector_env.step(actions)
        replay_next = next_observation.copy()
        for index, done in enumerate(dones):
            if done and "terminal_observation" in infos[index]:
                replay_next[index] = infos[index]["terminal_observation"]
        agent.replay_buffer.add(observation, replay_next, agent.policy.scale_action(actions),
                                rewards, dones, infos)
        observation = next_observation
        collected += vector_env.num_envs
        agent.num_timesteps += vector_env.num_envs
        agent._episode_num += int(np.sum(dones))
    agent._last_obs = observation
    agent._last_episode_starts = dones
    return collected


def load_agent(checkpoint, *, device="auto", render=False, model_path=None, full_task=False):
    agent = SAC.load(checkpoint, device=resolve_device(device))
    if getattr(agent, "door_task_format", None) != FORMAT:
        raise ValueError("Expected a door-task .zip checkpoint from trainp2_sac.py")
    specification = agent.door_environment
    reject_legacy_precision(specification)
    direct_reward = specification["task"] in (DIRECT_TASK, PRECISION_TASK)
    if specification["task"] == REFERENCE_TASK:
        reference_reward = ReferenceRewardConfig(**specification["reward_config"])
    elif specification["task"] in ("fixed_base_door_v1", DIRECT_TASK, PRECISION_TASK):
        reference_reward = None
    else:
        raise ValueError(f"Unsupported SAC door task: {specification['task']}")
    path = model_path or specification["model_path"]
    config = DoorConfig(**specification["config"])
    if specification["task"] == PRECISION_TASK:
        training_config = GraspTrainingConfig(**specification["training_config"])
        if full_task:
            training_config = replace(training_config, training_stage="full")
        env = PrecisionGraspDoorEnv(path, config=config,
                                   training_config=training_config,
                                   render_mode="human" if render else None)
    elif direct_reward:
        # Evaluation restores checkpoint values rather than new-training defaults.
        env = DirectRewardDoorEnv(
            path, config=config, render_mode="human" if render else None,
            position_reward=specification.get("position_reward", "exponential_reach"),
            orientation_reward=specification.get("orientation_reward", "distance_weighted_alignment"),
            distance_metric=specification.get("distance_metric", "grasp_target"),
            distance_weight=specification.get("distance_weight", 5.0),
            orientation_weight=specification.get("orientation_weight", 0.25),
        )
    else:
        env = make_environment(path, config, render=render, reference_reward=reference_reward)
    if (agent.observation_space != env.observation_space
            or agent.action_space != env.action_space or agent.gamma != env.config.gamma):
        env.close()
        raise ValueError("Checkpoint spaces/gamma do not match its environment")
    return agent, env


def evaluate(checkpoint, *, episodes=10, seed=20_000, device="auto", render=False, model_path=None,
             full_task=False):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    agent, env = load_agent(checkpoint, device=device, render=render, model_path=model_path, full_task=full_task)
    records = []
    try:
        for episode in range(episodes):
            observation, reset_info = env.reset(seed=seed + episode)
            episode_return = 0.0
            length = 0
            grasp_streak = longest_grasp = 0
            minimum_distance = float("inf")
            while True:
                action, _ = agent.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                length += 1
                grasp_streak = grasp_streak + 1 if info["grasped"] else 0
                longest_grasp = max(longest_grasp, grasp_streak)
                minimum_distance = min(minimum_distance, info["distance"])
                if terminated or truncated:
                    break
            records.append({
                "seed": seed + episode, "return": episode_return, "length": length,
                "reset_origin": reset_info.get("reset_origin", "home"),
                "is_success": info["is_success"], "grasp_seen": info["grasp_seen"],
                "stage_success": info.get("stage_success", info["is_success"]),
                "press_seen": info["press_seen"], "final_door_deg": float(np.rad2deg(info["door_angle"])),
                "max_door_deg": float(np.rad2deg(info["max_door_angle"])),
                "final_distance_m": info["distance"], "failure_reason": info["failure_reason"],
                "minimum_distance_m": minimum_distance,
                "longest_grasp_seconds": longest_grasp * env.dt,
                "truncated": bool(truncated),
            })
        return {
            "episodes": episodes, "first_seed": seed,
            "full_task_override": full_task,
            "reset_origins": sorted({r["reset_origin"] for r in records}),
            "success_rate": float(np.mean([r["is_success"] for r in records])),
            "stage_success_rate": float(np.mean([r["stage_success"] for r in records])),
            "training_stage": getattr(getattr(env, "training_config", None), "training_stage", "full"),
            "grasp_rate": float(np.mean([r["grasp_seen"] for r in records])),
            "press_rate": float(np.mean([r["press_seen"] for r in records])),
            "mean_return": float(np.mean([r["return"] for r in records])),
            "mean_final_door_deg": float(np.mean([r["final_door_deg"] for r in records])),
            "mean_max_door_deg": float(np.mean([r["max_door_deg"] for r in records])),
            "mean_longest_grasp_seconds": float(np.mean([r["longest_grasp_seconds"] for r in records])),
            "records": records,
        }
    finally:
        env.close()


def train(*, total_timesteps=300_000, output_dir="runs/door_sac", seed=0,
          model_path=DEFAULT_MODEL, env_config=None, learning_rate=None, buffer_size=None,
          learning_starts=None, batch_size=None, train_freq=None, gradient_steps=None,
          gamma=None, tau=None, ent_coef=None, n_steps=None, eval_freq=10_000, eval_episodes=10,
          render=False, device="auto", n_envs=1, reference_reward=None, policy_kwargs=None,
          replay_buffer_initializer=None, save_replay_buffer=False, direct_reward=False,
          eval_callback_class=EvalCallback, resume_checkpoint=None, replay_buffer_path=None,
          initialize_from=None, initialize_critic_scale=1.0, action_noise=None,
          initialize_without_force_inputs=False, training_reset_bank=None,
          training_reset_probability=0.5, adaptive_training_resets=False, policy_warmup_steps=0,
          success_replay_fraction=None):
    if replay_buffer_path is not None and resume_checkpoint is None and initialize_from is None:
        raise ValueError("replay_buffer_path requires a resume_checkpoint or online initialize_from")
    if initialize_from is not None and resume_checkpoint is not None:
        raise ValueError("initialize_from and resume_checkpoint cannot be combined")
    if not np.isfinite(initialize_critic_scale) or initialize_critic_scale <= 0:
        raise ValueError("initialize_critic_scale must be positive and finite")
    if initialize_critic_scale != 1.0 and initialize_from is None:
        raise ValueError("initialize_critic_scale requires initialize_from")
    if initialize_without_force_inputs and initialize_from is None:
        raise ValueError("initialize_without_force_inputs requires initialize_from")
    if training_reset_bank is not None and (not np.isfinite(training_reset_probability)
                                            or not 0 < training_reset_probability <= 1):
        raise ValueError("training_reset_probability must be in (0, 1]")
    if adaptive_training_resets and training_reset_bank is None:
        raise ValueError("adaptive_training_resets requires a training_reset_bank")
    if success_replay_fraction is not None and (not np.isfinite(success_replay_fraction)
                                                or not 0 <= success_replay_fraction <= 0.5):
        raise ValueError("success_replay_fraction must be in [0, 0.5]")
    if direct_reward and reference_reward is not None:
        raise ValueError("direct_reward and reference_reward cannot be combined")
    if min(total_timesteps, eval_freq, eval_episodes) < 1:
        raise ValueError("timesteps/eval_freq/eval_episodes must be positive")
    if not isinstance(policy_warmup_steps, int) or not 0 <= policy_warmup_steps < total_timesteps:
        raise ValueError("policy_warmup_steps must be a nonnegative integer below timesteps")
    if not isinstance(n_envs, int) or n_envs < 1:
        raise ValueError("n_envs must be a positive integer")
    if render and n_envs > 1:
        raise ValueError("Parallel training runs headless; use --evaluate CHECKPOINT --render to view the policy")
    overrides, actual = sac_settings(
        learning_rate=learning_rate, buffer_size=buffer_size, learning_starts=learning_starts,
        batch_size=batch_size, train_freq=train_freq, gradient_steps=gradient_steps,
        gamma=gamma, tau=tau, ent_coef=ent_coef, n_steps=n_steps,
    )
    if actual["buffer_size"] < n_envs:
        raise ValueError("buffer_size must hold at least one transition per environment")
    task_config = replace(env_config or DoorConfig(max_steps=DEFAULT_MAX_STEPS), gamma=actual["gamma"])
    selected_device = resolve_device(device)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    eval_seed = final_eval_seed(seed)
    env = eval_env = logger = None
    try:
        if resume_checkpoint is not None:
            saved_agent = SAC.load(resume_checkpoint, device="cpu")
            saved_spec = getattr(saved_agent, "door_environment", {})
            reject_legacy_precision(saved_spec)
            if saved_spec.get("task") == PRECISION_TASK:
                direct_reward = GraspTrainingConfig(**saved_spec["training_config"])
                print(f"Resuming saved precision settings: stage={direct_reward.training_stage}", flush=True)
            del saved_agent
        if n_envs == 1:
            env = Monitor(
                make_environment(model_path, task_config, render=render,
                                 reference_reward=reference_reward, direct_reward=direct_reward,
                                 training_reset_bank=training_reset_bank,
                                 training_reset_probability=training_reset_probability,
                                 adaptive_training_resets=adaptive_training_resets),
                str(output / "train"), info_keywords=monitor_keys(direct_reward),
            )
            specification = env.unwrapped.specification()
        else:
            workers = output / "workers"
            workers.mkdir(exist_ok=True)
            env = SubprocVecEnv([
                partial(make_worker, str(Path(model_path).resolve()), task_config,
                        workers / str(rank), reference_reward, direct_reward,
                        training_reset_bank, training_reset_probability, adaptive_training_resets)
                for rank in range(n_envs)
            ], start_method="spawn")
            specification = env.env_method("specification", indices=0)[0]
        if n_envs == 1:
            eval_env = Monitor(make_environment(model_path, task_config,
                                                reference_reward=reference_reward, direct_reward=direct_reward))
            eval_env.reset(seed=seed + 10_000)
        else:
            eval_env = SubprocVecEnv([
                partial(make_worker, str(Path(model_path).resolve()), task_config, None, reference_reward, direct_reward)
            ], start_method="spawn")
            eval_env.seed(seed + 10_000)
        if resume_checkpoint is None:
            agent = SAC("MlpPolicy", env, seed=seed, device=selected_device, verbose=1,
                        policy_kwargs=policy_kwargs, action_noise=action_noise, **overrides)
            if initialize_from is not None:
                source = SAC.load(initialize_from, device=selected_device)
                source_spec = getattr(source, "door_environment", {})
                reject_legacy_precision(source_spec)
                source_training = source_spec.get("training_config", {})
                target_training = specification.get("training_config", {})
                observation_encoding_matches = (
                    source_training.get("ready_distance") == target_training.get("ready_distance")
                    and (initialize_without_force_inputs
                         or source_training.get("contact_observation", "linear")
                         == target_training.get("contact_observation", "linear")))
                source_physics, target_physics = dict(source_spec.get("config", {})), dict(specification["config"])
                source_physics.pop("gamma", None)
                target_physics.pop("gamma", None)
                if (getattr(source, "door_task_format", None) != FORMAT
                        or source_spec.get("task") != PRECISION_TASK
                        or specification["task"] != PRECISION_TASK
                        or source_physics != target_physics
                        or source_spec["model_path"] != specification["model_path"]
                        or source.observation_space != agent.observation_space
                        or source.action_space != agent.action_space or not observation_encoding_matches):
                    raise ValueError("Initialization requires a precision SAC checkpoint with matching physics/actions")
                agent.policy.load_state_dict(source.policy.state_dict())
                if initialize_without_force_inputs:
                    # Preserve the online learner's zero-contact predictions
                    # while allowing a different encoding of the two forces.
                    networks = [agent.actor.latent_pi, *agent.critic.q_networks,
                                *agent.critic_target.q_networks]
                    with torch.no_grad():
                        for network in networks:
                            first_layer = next(layer for layer in network.modules()
                                               if isinstance(layer, torch.nn.Linear))
                            first_layer.weight[:, 62:64].zero_()
                if initialize_critic_scale != 1.0:
                    # Rescale Q predictions when changing reward units, while
                    # preserving the learned actor and critic feature maps.
                    with torch.no_grad():
                        for critic in (agent.critic, agent.critic_target):
                            for network in critic.q_networks:
                                network[-1].weight.mul_(initialize_critic_scale)
                                network[-1].bias.mul_(initialize_critic_scale)
                if replay_buffer_path is not None:
                    # Reuse only this task's own online experience, with exact
                    # target reward/observation/physics semantics. No relabeling
                    # or expert replay is accepted through this path.
                    replay_run = json.loads((Path(replay_buffer_path).parent / "config.json").read_text(encoding="utf-8"))
                    replay_spec = dict(replay_run.get("environment", {}))
                    if replay_spec.get("task") == PRECISION_TASK:
                        replay_spec["training_config"] = asdict(GraspTrainingConfig(**replay_spec["training_config"]))
                        # Unlike potential shaping, this task's stored rewards
                        # do not depend on the SAC discount factor.
                        replay_spec["config"] = dict(replay_spec["config"], gamma=specification["config"]["gamma"])
                    if (replay_run.get("algorithm") != "stable_baselines3.SAC"
                            or "demonstrations" in replay_run or replay_spec != specification):
                        raise ValueError("Initialization replay must be matching online precision SAC experience")
                    agent.load_replay_buffer(replay_buffer_path)
                    if agent.replay_buffer.n_envs != n_envs:
                        raise ValueError("Initialization replay requires the same number of environments")
                del source
        else:
            noise_override = {"action_noise": action_noise} if action_noise is not None else {}
            agent = SAC.load(resume_checkpoint, env=env, seed=seed, device=selected_device,
                             **noise_override, **overrides)
            restored_specification = dict(agent.door_environment)
            if restored_specification["task"] == PRECISION_TASK:
                restored_specification["training_config"] = asdict(
                    GraspTrainingConfig(**restored_specification["training_config"]))
            if restored_specification != specification:
                raise ValueError("Resume environment differs from checkpoint; retain its environment settings")
            if replay_buffer_path is not None:
                agent.load_replay_buffer(replay_buffer_path)
        # These attributes are embedded in final and best checkpoints, allowing
        # evaluation without config.json and with a model path override on a new host.
        agent.door_task_format = FORMAT
        agent.door_environment = specification
        replay_fraction = (getattr(agent, "door_success_replay_fraction", 0.0)
                           if success_replay_fraction is None else success_replay_fraction)
        if agent.n_steps > 1 and replay_fraction:
            raise ValueError("Choose uniform multi-step replay or one-step own-success replay")
        configure_multistep_replay(agent)
        if replay_fraction or isinstance(agent.replay_buffer, OwnSuccessReplayBuffer):
            if (specification["task"] != PRECISION_TASK
                    or specification["training_config"]["training_stage"] != "full"
                    or training_reset_bank is not None):
                raise ValueError("Own-success replay requires the full home-only precision task")
            if replay_buffer_path is not None:
                replay_run = json.loads((Path(replay_buffer_path).parent / "config.json").read_text(encoding="utf-8"))
                if replay_run.get("training_reset_curriculum") is not None or "demonstrations" in replay_run:
                    raise ValueError("Own-success replay cannot infer home episodes from curriculum or demonstration data")
            hinge = env.unwrapped.hinge if n_envs == 1 else env.get_attr("hinge")[0]
            replay_settings = dict(success_fraction=replay_fraction, door_index=hinge,
                                   door_threshold=task_config.success_angle / 0.4,
                                   grasp_threshold=task_config.contact_threshold / specification["training_config"]["contact_reference"])
            agent.replay_buffer = OwnSuccessReplayBuffer.from_buffer(agent.replay_buffer, **replay_settings)
            agent.replay_buffer_class = OwnSuccessReplayBuffer
            agent.replay_buffer_kwargs = dict(agent.replay_buffer_kwargs or {}, **replay_settings)
        agent.door_success_replay_fraction = replay_fraction
        # Checkpoints evaluate with the default home initialization. Preserve
        # the distinct training reset distribution as separate provenance.
        reset_specification = None
        if training_reset_bank is not None:
            reset_specification = (env.unwrapped.training_reset_specification() if n_envs == 1
                                   else env.env_method("training_reset_specification", indices=0)[0])
        agent.door_training_reset_curriculum = reset_specification
        logger = configure(str(output / "logs"), ["stdout", "csv"])
        agent.set_logger(logger)
        replay_metadata = replay_buffer_initializer(agent) if replay_buffer_initializer is not None else None
        restored_settings = {key: agent.train_freq.frequency if key == "train_freq" else getattr(agent, key)
                             for key in actual}
        settings = dict(restored_settings, net_arch=agent.policy.net_arch,
                        activation=agent.policy.activation_fn.__name__,
                        optimizer=agent.policy.optimizer_class.__name__,
                        n_critics=agent.critic.n_critics, target_entropy=agent.target_entropy)
        run_config = {
            "algorithm": "stable_baselines3.SAC", "environment": agent.door_environment,
            "training_action_noise": str(agent.action_noise),
            "initialization_without_force_inputs": initialize_without_force_inputs,
            "training_reset_curriculum": reset_specification,
            "success_replay_fraction": replay_fraction,
            "agent": settings, "algorithm_overrides": overrides, "seed": seed,
            "final_eval_seed": eval_seed, "requested_timesteps": total_timesteps,
            "actual_timesteps": 0, "eval_freq": eval_freq, "eval_episodes": eval_episodes,
            "render": render, "device": str(selected_device),
            "n_envs": n_envs, "torch_threads": torch.get_num_threads(),
            "worker_seeds": [seed + rank for rank in range(n_envs)],
            "vector_backend": "SubprocVecEnv(spawn)" if n_envs > 1 else "DummyVecEnv",
        }
        if isinstance(agent.replay_buffer, OwnSuccessReplayBuffer):
            run_config["success_replay"] = agent.replay_buffer.specification()
            print(f"Own online success replay: {run_config['success_replay']}", flush=True)
        # A callback is invoked once per vector step, which yields n_envs
        # transitions. Round up so evaluation never runs earlier than requested.
        callback_freq = max(1, (eval_freq + n_envs - 1) // n_envs)
        run_config["effective_eval_freq"] = callback_freq * n_envs
        if replay_metadata is not None:
            run_config["demonstrations"] = replay_metadata
        if initialize_from is not None:
            run_config["policy_initialization"] = str(Path(initialize_from).resolve())
            run_config["initialization_source"] = (
                "previous online SAC weights; matching online replay; fresh optimizers" if replay_buffer_path is not None
                else "previous online SAC weights; empty replay buffer; fresh optimizers")
            if replay_buffer_path is not None:
                run_config["initialization_online_replay"] = str(Path(replay_buffer_path).resolve())
            run_config["critic_initialization_scale"] = initialize_critic_scale
        config_path = output / "config.json"
        config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
        print(f"SAC n_envs={n_envs}, network={settings['net_arch']}, activation={settings['activation']}, "
              f"gamma={agent.gamma}, ent_coef={agent.ent_coef}; overrides={overrides}", flush=True)
        if agent.num_timesteps + total_timesteps <= agent.learning_starts:
            print("Budget does not exceed learning_starts: this run collects data without SAC updates.", flush=True)
        callback = eval_callback_class(
            eval_env, best_model_save_path=str(output / "best"), log_path=str(output / "eval"),
            eval_freq=callback_freq, n_eval_episodes=eval_episodes, deterministic=True,
        )
        started = time.perf_counter()
        first_timestep = agent.num_timesteps
        warmed = collect_online_policy_warmup(agent, policy_warmup_steps) if policy_warmup_steps else 0
        if warmed >= total_timesteps:
            raise ValueError("Rounded policy warmup must leave at least one training step")
        run_config["online_policy_warmup_transitions"] = warmed
        callback.n_calls += warmed // n_envs
        agent.learn(total_timesteps=total_timesteps - warmed, callback=callback,
                    reset_num_timesteps=resume_checkpoint is None and not warmed)
        training_seconds = time.perf_counter() - started
        # Flush the last update's diagnostics even if no episode just ended.
        agent.logger.dump(step=agent.num_timesteps)
        checkpoint = output / "sac_door.zip"
        if adaptive_training_resets:
            final_reset_workers = ([env.unwrapped.training_reset_specification()] if n_envs == 1
                                   else env.env_method("training_reset_specification"))
            agent.door_training_reset_curriculum = dict(reset_specification, final_workers=final_reset_workers)
            run_config["training_reset_curriculum"] = agent.door_training_reset_curriculum
        agent.save(checkpoint)
        if adaptive_training_resets:
            pool_directory = output / "adaptive_resets"
            pool_directory.mkdir(exist_ok=True)
            exports = []
            for rank in range(n_envs):
                pool_path = str(pool_directory / f"worker_{rank}.npz")
                exported = (env.unwrapped.save_adaptive_training_states(pool_path, str(checkpoint)) if n_envs == 1
                            else env.env_method("save_adaptive_training_states", pool_path, str(checkpoint), indices=rank)[0])
                if exported is not None:
                    exports.append(exported)
            run_config["adaptive_training_reset_exports"] = exports
        if save_replay_buffer:
            replay_path = output / "replay_buffer.pkl"
            agent.save_replay_buffer(replay_path)
            run_config["replay_buffer_path"] = str(replay_path)
        run_config.update(actual_timesteps=agent.num_timesteps, gradient_updates=agent._n_updates,
                          training_seconds=training_seconds,
                          transitions_per_second=(agent.num_timesteps - first_timestep) / max(training_seconds, 1e-9))
        if isinstance(agent.replay_buffer, OwnSuccessReplayBuffer):
            run_config["success_replay"] = agent.replay_buffer.specification()
        config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    finally:
        if logger is not None:
            logger.close()
        if env is not None:
            env.close()
        if eval_env is not None:
            eval_env.close()
    print(f"Final evaluation seed: {eval_seed}", flush=True)
    metrics = evaluate(checkpoint, episodes=eval_episodes, seed=eval_seed,
                       model_path=model_path, device=selected_device)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {checkpoint}; success={metrics['success_rate']:.1%}, grasp={metrics['grasp_rate']:.1%}")
    return checkpoint


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/door_sac"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--buffer-size", type=int)
    parser.add_argument("--learning-starts", type=int, help="Random-action warmup steps (SB3 default 100).")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--train-freq", type=int, help="Environment steps collected between updates.")
    parser.add_argument("--gradient-steps", type=int, help="Updates per collection; -1 matches collected steps.")
    parser.add_argument("--n-steps", type=int, help="SB3 SAC return horizon (default 1); uniform online replay only.")
    parser.add_argument("--gamma", type=float, help="SAC discount factor (SB3 default 0.99).")
    parser.add_argument("--tau", type=float)
    parser.add_argument("--ent-coef", help="auto (default), auto_0.1, or a fixed nonnegative alpha.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                        help="Maximum actions per episode (default 1000).")
    parser.add_argument("--action-scale", type=float, default=0.03)
    parser.add_argument("--initial-noise", type=float, default=0.02)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--evaluate", type=Path)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main():
    args = build_parser().parse_args()
    if args.evaluate:
        metrics = evaluate(args.evaluate, episodes=args.episodes, seed=args.seed,
                           device=args.device, render=args.render, model_path=args.model)
        print(json.dumps(metrics, indent=2))
    else:
        train(total_timesteps=args.timesteps, output_dir=args.output_dir, seed=args.seed,
              model_path=args.model or DEFAULT_MODEL,
              env_config=DoorConfig(max_steps=args.max_steps, action_scale=args.action_scale,
                                    initial_noise=args.initial_noise),
              learning_rate=args.learning_rate, buffer_size=args.buffer_size,
              learning_starts=args.learning_starts, batch_size=args.batch_size,
              train_freq=args.train_freq, gradient_steps=args.gradient_steps,
              gamma=args.gamma, tau=args.tau, ent_coef=args.ent_coef, n_steps=args.n_steps,
              eval_freq=args.eval_freq, eval_episodes=args.episodes, render=args.render, device=args.device)


if __name__ == "__main__":
    main()
