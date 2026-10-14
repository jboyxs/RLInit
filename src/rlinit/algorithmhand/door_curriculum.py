"""Algorithm-independent graduation and evaluation for the door curriculum."""

from dataclasses import dataclass

import numpy as np

from .door_curriculum_env import CurriculumConfig, DoorCurriculumEnv, STAGES
from .door_env import DoorConfig


@dataclass
class CurriculumProgress:
    stage: str = "reach"
    home_probability: float = 0.0
    passed_checks: int = 0
    completed: bool = False

    def advance(self, success_rate, threshold, patience, home_step):
        if self.completed:
            return False
        self.passed_checks = self.passed_checks + 1 if success_rate >= threshold else 0
        if self.passed_checks < patience:
            return False
        self.passed_checks = 0
        if self.stage in ("reach", "full") and self.home_probability < 1:
            self.home_probability = min(1.0, self.home_probability + home_step)
        elif self.stage == "full":
            self.completed = True
        else:
            self.stage = STAGES[STAGES.index(self.stage) + 1]
            self.home_probability = home_step if self.stage == "full" else 0.0
        return True


def evaluate_policy(action_fn, specification, *, episodes=10, seed=20_000,
                    render=False, stage=None, home_probability=None):
    """Evaluate a deterministic action function in a separate physical env.

    Both PPO trainers use this evaluator. Full-task success only counts when
    the full stage succeeds in an episode whose initial state was home.
    """
    if episodes < 1:
        raise ValueError("episodes must be positive")
    env = DoorCurriculumEnv(
        specification["model_path"], config=DoorConfig(**specification["config"]),
        curriculum=CurriculumConfig(**specification["curriculum"]),
        stage=specification["stage"] if stage is None else stage,
        home_probability=specification["home_probability"] if home_probability is None else home_probability,
        render_mode="human" if render else None,
    )
    records = []
    try:
        for episode in range(episodes):
            obs, initial = env.reset(seed=seed + episode)
            total = 0.0
            min_distance = initial["distance"]
            for length in range(1, env.config.max_steps + 1):
                obs, reward, terminated, truncated, info = env.step(action_fn(obs))
                total += reward
                min_distance = min(min_distance, info["distance"])
                if terminated or truncated:
                    break
            records.append({
                "seed": seed + episode, "return": total, "length": length,
                "episode_start": info["episode_start"], "is_success": info["is_success"],
                "full_task_success": info["full_task_success"], "grasp_seen": info["grasp_seen"],
                "press_seen": info["press_seen"], "min_distance_m": min_distance,
                "final_distance_m": info["distance"], "alignment": info["alignment"],
                "final_door_deg": float(np.rad2deg(info["door_angle"])),
                "failure_reason": info["failure_reason"], "truncated": bool(truncated),
            })
        home = [r for r in records if r["episode_start"] == "home"]
        return {
            "stage": env.stage, "home_probability": env.home_probability,
            "episodes": episodes, "first_seed": seed,
            "success_rate": float(np.mean([r["is_success"] for r in records])),
            "full_task_success_rate": float(np.mean([r["full_task_success"] for r in records])),
            "home_episodes": len(home),
            "home_success_rate": float(np.mean([r["is_success"] for r in home])) if home else None,
            "mean_return": float(np.mean([r["return"] for r in records])),
            "mean_min_distance_m": float(np.mean([r["min_distance_m"] for r in records])),
            "mean_final_door_deg": float(np.mean([r["final_door_deg"] for r in records])),
            "records": records,
        }
    finally:
        env.close()
