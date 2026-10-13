"""旧小批次实验的兼容入口；当前更新方式与 PPOhand.py 一致。

训练结束后的自动评估起始种子统一为训练 seed + 20000。

训练：uv run --no-sync python tests/train_hand_minibatch.py --timesteps 300000 --batch-size 64
评估：uv run --no-sync python tests/train_hand_minibatch.py --evaluate runs/reach_ppohand_minibatch/ppo_hand.pt --render
"""

from rlinit.algorithmhand.PPOhand_minibatch import PPO
from train_hand import main


if __name__ == "__main__":
    main(agent_class=PPO, default_output_dir="runs/reach_ppohand_minibatch",
         default_hidden_dim=128, description=__doc__)
