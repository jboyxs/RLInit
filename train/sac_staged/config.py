"""Shared recipe constants, device resolution and source provenance."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
FORMAT = "rlinit-sac-learn-v1"
MONITOR_KEYS = ("is_success", "grasp_seen", "press_seen", "door_angle", "failure_reason")

def recipe_sources():
    """Archive the implementation actually used, including the entry point."""
    return (ROOT / "train/train_sac_modular.py",
            ROOT / "train/train_sac_render.py",
            *sorted(Path(__file__).resolve().parent.glob("*.py")),
            ROOT / "src/rlinit/algorithmhand/door_sac_learn_env.py")

@dataclass(frozen=True)
class SACLearnSettings:
    learning_rate: float = 0.0001
    buffer_size: int = 300000
    batch_size: int = 256
    gamma: float = 0.995
    tau: float = 0.005
    ent_coef: float = 0.005
    n_steps: int = 1
    multistep_after: int = 400000
    final_n_steps: int = 3
    gripper_exploration_std: float = 0.3
    scratch_learning_starts: int = 5000

    def __post_init__(self):
        if self.multistep_after < 0 or self.final_n_steps < 1:
            raise ValueError("Invalid multistep schedule")
        for key in ("buffer_size", "batch_size", "n_steps"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not isinstance(self.scratch_learning_starts, int) or self.scratch_learning_starts < 0:
            raise ValueError("scratch_learning_starts must be nonnegative")
        for key in ("learning_rate", "gamma", "tau", "ent_coef", "gripper_exploration_std"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.learning_rate <= 0 or not 0 < self.gamma <= 1 or not 0 < self.tau <= 1:
            raise ValueError("Invalid learning rate, gamma or tau")


def resolve_device(device="auto"):
    """Auto prefers CUDA; explicit CUDA fails clearly when unavailable."""
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(device)
    if selected.type not in ("cpu", "cuda"):
        raise ValueError("device 必须是 auto、cpu 或 cuda（可指定 cuda:0）")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用：请安装 CUDA 版 PyTorch 并检查 NVIDIA 驱动；或使用 --device cpu")
    return selected


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
