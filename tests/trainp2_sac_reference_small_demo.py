"""Reference-reward SAC with scripted replay prefill and smaller (256, 256) networks.

uv run --no-sync python tests/trainp2_sac_reference_small_demo.py --demo-episodes 10 --timesteps 300000
uv run --no-sync python tests/trainp2_sac_reference_small_demo.py --n-envs 4 --timesteps 300000
uv run --no-sync python tests/trainp2_sac_reference_small_demo.py --evaluate runs/door_sac_reference_small_demo/sac_door.zip --render

Actor and twin Q networks use two 256-unit ReLU layers. Demonstration collection,
uniform replay, rewards and all other defaults match the large-demo entry point.
"""

import multiprocessing

from trainp2_sac_reference_large_demo import main as main_demo, train as train_demo


DEFAULT_OUTPUT = "runs/door_sac_reference_small_demo"


def train(*, output_dir=DEFAULT_OUTPUT, **kwargs):
    return train_demo(output_dir=output_dir, net_arch=(256, 256), **kwargs)


def main():
    main_demo(train_fn=train, output_dir=DEFAULT_OUTPUT, description=__doc__)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
