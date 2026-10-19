# Verified joint-action SAC door policy

The frozen neural policy completed 97 of 100 independent noisy-home episodes.
Grasp and press were observed in all 100. Three episodes failed the unchanged
penetration check. Seeds: 70000–70019 and 71000–71079. This result applies to
the current simulated scene and original initial-noise distribution (0.02).

The policy directly outputs seven joint increments and one gripper action.
No imitation learning, expert actions, Cartesian controller, IK, or Jacobian
mapping is used. Original task success requires bilateral grasp and door angle
at least 0.30 radians for 0.30 seconds, without an original physics failure.

From the RLInit project root:

```powershell
uv run --no-sync python tests/trainp2_sac_direct.py --evaluate runs/door_sac_joint_v2_verified/sac_door.zip --model models/door_scene.xml --episodes 1 --seed 70000 --device cpu --render
```

`sac_door.zip` is the selected 140000-step checkpoint of the multistep stage,
initialized from earlier online joint SAC weights and same-task self-generated
experience. It is not a policy trained from scratch in only 140000 steps.
Later checkpoints regressed; use this fixed model for the reported result.

Training used standard SB3 SAC losses, uniform 3-step returns, gamma 0.995,
alpha 0.005, learning rate 0.0001, batch 256, four environments and one update
per collected transition. Architecture: [256, 256] ReLU. Reward profile: joint;
global orientation cost: 0.5; contact observations: log. Training gripper OU
noise: standard deviation 0.3; inference has no added noise. This run used
only home resets. It used neither success-priority replay nor the optional
safe-opening reward or state-reset curriculum.

- `evaluation.json`: all 100 results and aggregate metrics.
- `evaluation_20.json`, `evaluation_cli_80.json`: original evaluation outputs.
- `manifest.json`: checkpoint/source hashes and provenance.
- `selection.json`: the periodic evaluation used to select the frozen model.
- `source_run_config.json`: originating training-run configuration.
- `opening.gif`: actual deterministic simulator rollout, seed 70000.
- `video_metrics.json`: that video's original success predicate and model hash.
- `record_video.py`: regenerates the video from the checkpoint.

Evaluation only needs the model and the project's scene/code, not the original
training replay or reset banks. The root README includes the exact multistep
training command and the two independent evaluation commands.
