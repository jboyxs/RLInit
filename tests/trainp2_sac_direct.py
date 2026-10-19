"""Train SAC with DoorEnv's features rewarded directly instead of differenced.

Position penalizes summed finger/handle surface distances; orientation rewards
alignment * exp(-distance / 0.08). Weights come from door_direct_env.py defaults.

uv run --no-sync python tests/trainp2_sac_direct.py --timesteps 300000
uv run --no-sync python tests/trainp2_sac_direct.py --evaluate runs/door_sac_direct/sac_door.zip --seed 20000 --render

The default profile matches trainp2_sac.py's observation/action spaces.
--precision-grasp adds explicit fingertip observations and bounded shaping;
both profiles use the original seven joint increments and gripper action.
There is no Cartesian controller, inverse kinematics or scripted action policy.
Both profiles train online SAC; evaluation always uses noisy home starts.
"""

import json
import multiprocessing
from pathlib import Path
import numpy as np
import torch
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from rlinit.algorithmhand.door_env import DEFAULT_MODEL, DoorConfig
from rlinit.algorithmhand.door_sac_callback import DoorSuccessEvalCallback
from rlinit.algorithmhand.door_sac_grasp_env import GraspTrainingConfig
from trainp2_sac import build_parser, evaluate, train as train_sac


def train(*, output_dir="runs/door_sac_direct", precision_grasp=False, torch_threads=1,
          orientation_cost_weight=0.0, grasp_reward_profile="balanced", training_stage="full",
          gripper_exploration_std=0.0, contact_observation="linear", **kwargs):
    if torch_threads < 1:
        raise ValueError("torch_threads must be positive")
    if not np.isfinite(gripper_exploration_std) or gripper_exploration_std < 0:
        raise ValueError("gripper_exploration_std must be nonnegative and finite")
    if gripper_exploration_std:
        if "action_noise" in kwargs:
            raise ValueError("Choose gripper_exploration_std or action_noise")
        # Task-independent stochastic exploration, with ~1 second correlation.
        # Arm actions remain unchanged. SAC evaluation uses the actor alone.
        sigma = np.zeros(8)
        sigma[7] = np.sqrt(2.0) * gripper_exploration_std
        kwargs["action_noise"] = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(8), sigma=sigma, theta=1.0, dt=0.02)
    if {"cartesian_actions", "handle_pose_actions"} & kwargs.keys():
        raise ValueError("Controller action options were removed; SAC must command joint increments directly")
    if orientation_cost_weight and not precision_grasp:
        raise ValueError("precision_grasp is required for its action/reward options")
    if contact_observation != "linear" and not precision_grasp:
        raise ValueError("contact_observation requires precision_grasp")
    if kwargs.get("training_reset_bank") is not None and not precision_grasp:
        raise ValueError("training_reset_bank requires precision_grasp")
    if precision_grasp and kwargs.get("replay_buffer_initializer") is not None:
        raise ValueError("The precision SAC task uses online experience only")
    if grasp_reward_profile not in ("balanced", "strong", "opening", "joint", "safe_opening"):
        raise ValueError("Unknown grasp_reward_profile")
    if grasp_reward_profile != "balanced" and not precision_grasp:
        raise ValueError("grasp_reward_profile requires precision_grasp")
    if training_stage != "full" and not precision_grasp:
        raise ValueError("training_stage requires precision_grasp")
    configuration = {"balanced": GraspTrainingConfig, "strong": GraspTrainingConfig.strong,
                     "opening": GraspTrainingConfig.opening,
                     "joint": GraspTrainingConfig.joint,
                     "safe_opening": GraspTrainingConfig.safe_opening}[grasp_reward_profile]
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(torch_threads)
        if precision_grasp:
            kwargs.setdefault("eval_callback_class", DoorSuccessEvalCallback)
        return train_sac(output_dir=output_dir,
                         direct_reward=configuration(orientation_cost_weight=orientation_cost_weight,
                                                     contact_observation=contact_observation,
                                                     training_stage=training_stage)
                         if precision_grasp else True, **kwargs)
    finally:
        torch.set_num_threads(previous_threads)


def main():
    parser = build_parser()
    parser.description = __doc__
    parser.set_defaults(output_dir=Path("runs/door_sac_direct"))
    parser.add_argument("--n-envs", type=int, default=1, help="Physics processes (default 1).")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--gripper-exploration-std", type=float, default=0.0,
                        help="Training-only correlated gripper action noise; evaluation uses policy actions alone.")
    parser.add_argument("--contact-observation", choices=("linear", "log"), default="linear",
                        help="Precision force input encoding, saved in the model.")
    parser.add_argument("--training-reset-bank", type=Path,
                        help="Training-only simulator states collected from this joint SAC learner; evaluation remains home-only.")
    parser.add_argument("--training-reset-probability", type=float, default=0.5,
                        help="Fraction of training episodes starting from the learner's saved states.")
    parser.add_argument("--expand-training-reset-bank", action="store_true",
                        help="Replay safe grasped opening progress generated by the online learner; never alters actions or evaluation resets.")
    parser.add_argument("--policy-warmup-steps", type=int, default=0,
                        help="Collect the initialized SAC actor's own stochastic rollouts before updates; included in timesteps.")
    parser.add_argument("--success-replay-fraction", type=float,
                        help="Home-only full task: sample up to 0.5 of each batch from own successful episodes, with unchanged SAC losses.")
    parser.add_argument("--precision-grasp", action="store_true",
                        help="Online joint-action SAC with fingertip observations and centering/balance costs.")
    parser.add_argument("--orientation-cost-weight", type=float, default=0.0,
                        help="Additional global orientation cost in the precision task.")
    parser.add_argument("--grasp-reward-profile", choices=("balanced", "strong", "opening", "joint", "safe_opening"), default="balanced",
                        help="Precision reward coefficients; saved completely in checkpoints.")
    parser.add_argument("--training-stage", choices=("full", "reach", "grasp", "press"), default="full",
                        help="Online SAC subtask target, always from home; only full counts as opening success.")
    parser.add_argument("--full-task", action="store_true", help="Evaluate a subtask policy using the original full opening rule.")
    parser.add_argument("--resume", type=Path, help="Continue an online SAC checkpoint with matching environment settings.")
    parser.add_argument("--init-from", type=Path,
                        help="Initialize networks from a previous online precision SAC run, with an empty replay buffer.")
    parser.add_argument("--init-critic-scale", type=float, default=1.0,
                        help="Scale initial Q outputs when changing reward units; leaves the learned actor unchanged.")
    parser.add_argument("--init-without-force-inputs", action="store_true",
                        help="Zero only first-layer force columns when transferring an online joint policy to a new force encoding.")
    parser.add_argument("--replay-buffer", type=Path,
                        help="Online SAC replay matching the resumed or initialized target task and reward configuration.")
    parser.add_argument("--save-replay-buffer", action="store_true")
    args = parser.parse_args()
    if args.evaluate:
        if args.torch_threads < 1:
            parser.error("--torch-threads must be positive")
        torch.set_num_threads(args.torch_threads)
        metrics = evaluate(args.evaluate, episodes=args.episodes, seed=args.seed,
                           device=args.device, render=args.render, model_path=args.model,
                           full_task=args.full_task)
        print(json.dumps(metrics, indent=2))
    else:
        train(total_timesteps=args.timesteps, output_dir=args.output_dir, seed=args.seed,
              precision_grasp=args.precision_grasp, torch_threads=args.torch_threads,
              gripper_exploration_std=args.gripper_exploration_std,
              contact_observation=args.contact_observation,
              training_reset_bank=args.training_reset_bank,
              training_reset_probability=args.training_reset_probability,
              adaptive_training_resets=args.expand_training_reset_bank,
              policy_warmup_steps=args.policy_warmup_steps,
              success_replay_fraction=args.success_replay_fraction,
              orientation_cost_weight=args.orientation_cost_weight,
              grasp_reward_profile=args.grasp_reward_profile,
              training_stage=args.training_stage,
              resume_checkpoint=args.resume, replay_buffer_path=args.replay_buffer,
              initialize_from=args.init_from,
              initialize_critic_scale=args.init_critic_scale,
              initialize_without_force_inputs=args.init_without_force_inputs,
              save_replay_buffer=args.save_replay_buffer,
              n_envs=args.n_envs, model_path=args.model or DEFAULT_MODEL,
              env_config=DoorConfig(max_steps=args.max_steps, action_scale=args.action_scale,
                                    initial_noise=args.initial_noise),
              learning_rate=args.learning_rate, buffer_size=args.buffer_size,
              learning_starts=args.learning_starts, batch_size=args.batch_size,
              train_freq=args.train_freq, gradient_steps=args.gradient_steps,
              gamma=args.gamma, tau=args.tau, ent_coef=args.ent_coef, n_steps=args.n_steps,
              eval_freq=args.eval_freq, eval_episodes=args.episodes,
              render=args.render, device=args.device)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
