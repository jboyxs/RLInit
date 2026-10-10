"""Select a training device independently of the checkpoint's original host."""

import torch


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
