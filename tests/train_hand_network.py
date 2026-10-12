"""仅更换 SB3 式双层 Tanh 网络的 PPOhand 训练/评估入口。

训练结束后的自动评估起始种子统一为训练 seed + 20000。

训练：uv run --no-sync python tests/train_hand_network.py --timesteps 300000
评估：uv run --no-sync python tests/train_hand_network.py --evaluate runs/reach_ppohand_network/ppo_hand.pt --render
"""

from rlinit.algorithmhand.PPOhand_network import PPO
from train_hand import main


if __name__ == "__main__":
    main(agent_class=PPO, default_output_dir="runs/reach_ppohand_network",
         default_hidden_dim=64, description=__doc__)
