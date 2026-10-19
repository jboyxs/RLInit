"""Evaluate door SAC by task completion first, with resumable snapshots."""

import json
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


class DoorSuccessEvalCallback(BaseCallback):
    def __init__(self, eval_env, *, best_model_save_path, log_path, eval_freq,
                 n_eval_episodes=5, deterministic=True, replay_save_freq=20_000,
                 eval_seed=10_000):
        super().__init__()
        self.eval_env = eval_env
        self.best_path = Path(best_model_save_path)
        self.log_path = Path(log_path)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.replay_save_freq = replay_save_freq
        self.eval_seed = eval_seed
        self.last_replay_step = 0
        self.best_rank = (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -np.inf)
        self.records = []

    def _init_callback(self):
        self.best_path.mkdir(parents=True, exist_ok=True)
        self.log_path.mkdir(parents=True, exist_ok=True)
        history = self.log_path / "task_evaluations.json"
        if history.exists():
            self.records = json.loads(history.read_text(encoding="utf-8"))
            if self.records:
                self.best_rank = max(self._rank(r) for r in self.records)

    @staticmethod
    def _rank(record):
        stage = record.get("training_stage", "full")
        return (record["success_rate"], record.get("stage_success_rate", record["success_rate"]),
                record["press_rate"] if stage in ("full", "press") else 0.0,
                record.get("mean_max_grasped_door_deg", 0.0) if stage == "full" else 0.0,
                record.get("mean_longest_grasp_steps", 0.0), record["grasp_rate"], record["mean_return"])

    def _on_step(self):
        if self.n_calls % 250 == 0 and (self.best_path.parent / "STOP_TRAINING").exists():
            print("STOP_TRAINING requested: saving final online SAC checkpoint.", flush=True)
            return False
        if self.n_calls % self.eval_freq:
            return True
        terminal = []
        nearest = []
        grasp_lengths = []
        grasped_openings = []
        current_nearest = float("inf")
        grasp_streak = longest_grasp = 0
        maximum_grasped_opening = 0.0

        if hasattr(self.eval_env, "seed"):
            self.eval_env.seed(self.eval_seed)
        else:
            self.eval_env.reset(seed=self.eval_seed)

        def collect(locals_, globals_):
            nonlocal current_nearest, grasp_streak, longest_grasp, maximum_grasped_opening
            info = locals_["info"]
            current_nearest = min(current_nearest, info["distance"])
            grasp_streak = grasp_streak + 1 if info["grasped"] else 0
            longest_grasp = max(longest_grasp, grasp_streak)
            if grasp_streak >= 5 and not info["failure_reason"]:
                maximum_grasped_opening = max(maximum_grasped_opening, info["door_angle"])
            if locals_["done"]:
                terminal.append(dict(info))
                nearest.append(current_nearest)
                grasp_lengths.append(longest_grasp)
                grasped_openings.append(maximum_grasped_opening)
                current_nearest = float("inf")
                grasp_streak = longest_grasp = 0
                maximum_grasped_opening = 0.0

        rewards, lengths = evaluate_policy(
            self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic, return_episode_rewards=True, callback=collect,
        )
        success = float(np.mean([r["is_success"] for r in terminal]))
        stage_success = float(np.mean([r.get("stage_success", r["is_success"]) for r in terminal]))
        press = float(np.mean([r["press_seen"] for r in terminal]))
        grasp = float(np.mean([r["grasp_seen"] for r in terminal]))
        record = {
            "timesteps": self.num_timesteps, "success_rate": success,
            "stage_success_rate": stage_success,
            "training_stage": terminal[0].get("training_stage", "full"),
            "evaluation_reset_origins": sorted({r.get("reset_origin", "home") for r in terminal}),
            "press_rate": press, "grasp_rate": grasp, "mean_return": float(np.mean(rewards)),
            "mean_length": float(np.mean(lengths)),
            "mean_distance_m": float(np.mean([r["distance"] for r in terminal])),
            "mean_minimum_distance_m": float(np.mean(nearest)),
            "mean_final_alignment": float(np.mean([r["alignment"] for r in terminal])),
            "mean_longest_grasp_steps": float(np.mean(grasp_lengths)),
            "mean_max_grasped_door_deg": float(np.rad2deg(np.mean(grasped_openings))),
            "failures": sum(bool(r["failure_reason"]) for r in terminal),
        }
        rank = self._rank(record)
        record["rank"] = list(rank)
        self.records.append(record)
        (self.log_path / "task_evaluations.json").write_text(
            json.dumps(self.records, indent=2) + "\n", encoding="utf-8")
        for key in ("success_rate", "stage_success_rate", "press_rate", "grasp_rate", "mean_return", "mean_distance_m"):
            self.logger.record(f"eval/{key}", record[key])
        self.logger.dump(self.num_timesteps)
        output = self.best_path.parent
        curriculum = getattr(self.model, "door_training_reset_curriculum", None)
        if curriculum is not None:
            self.model.door_training_reset_curriculum = dict(
                curriculum, current_workers=self.training_env.env_method("training_reset_specification"))
        self.model.save(output / "latest_model.zip")
        if rank > self.best_rank:
            self.best_rank = rank
            self.model.save(self.best_path / "best_model.zip")
            (self.best_path / "selection.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        if self.num_timesteps - self.last_replay_step >= self.replay_save_freq:
            self.model.save_replay_buffer(output / "latest_replay_buffer.pkl")
            self.last_replay_step = self.num_timesteps
        print(f"Task evaluation @{self.num_timesteps}: success={success:.0%}, press={press:.0%}, "
              f"grasp={grasp:.0%}, stage={record['training_stage']}:{stage_success:.0%}, "
              f"longest_grasp={record['mean_longest_grasp_steps']:.1f} steps, return={rank[-1]:.3f}", flush=True)
        return True
