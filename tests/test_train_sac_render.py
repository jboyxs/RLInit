"""Viewer CLI routing, two-stage handoff and real SAC updates with viewer hooks."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import zipfile

import numpy as np

from train import train_sac_render as entry
from train.sac_staged import pipeline, training


class RenderTrainingTests(unittest.TestCase):
    def test_cli_selects_one_rendered_environment_and_four_without_render(self):
        for arguments, count, render in (([], 4, False), (["--render"], 1, True),
                                        (["--render", "--n-envs", "1"], 1, True),
                                        (["--n-envs", "2"], 2, False)):
            with self.subTest(arguments=arguments), patch.object(sys, "argv", ["entry", *arguments]), \
                    patch.object(entry, "run") as run:
                entry.main()
                self.assertEqual(run.call_args.kwargs["n_envs"], count)
                self.assertEqual(run.call_args.kwargs["render"], render)

    def test_multiple_rendered_environments_are_rejected_before_training(self):
        with patch.object(sys, "argv", ["entry", "--render", "--n-envs", "4"]), \
                patch.object(entry, "run") as run, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                entry.main()
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()
        with tempfile.TemporaryDirectory() as folder, patch.object(pipeline, "train") as train:
            with self.assertRaisesRegex(ValueError, "n_envs=1"):
                pipeline.run(output_dir=Path(folder) / "run", render=True)
            train.assert_not_called()

    def test_evaluation_and_trace_keep_their_viewer_options(self):
        for trace in (False, True):
            arguments = ["entry", "--evaluate", "model.zip", "--render", "--episodes", "2"]
            if trace:
                arguments += ["--trace-output", "trace.json"]
            with patch.object(sys, "argv", arguments), patch.object(entry, "run") as run, \
                    patch.object(entry, "evaluate", return_value={}) as evaluate, \
                    patch.object(entry, "inspect") as inspect, redirect_stdout(io.StringIO()):
                entry.main()
                selected = inspect if trace else evaluate
                self.assertTrue(selected.call_args.kwargs["render"])
                self.assertEqual(selected.call_args.kwargs["episodes"], 2)
                run.assert_not_called()

    def test_pipeline_passes_render_to_both_stages_and_keeps_own_replay(self):
        calls = []

        def fake_train(**kwargs):
            calls.append(kwargs)
            root = kwargs["output_dir"]
            root.mkdir(parents=True)
            first = root.name == "phase1"
            count = 20000 if first else 20004
            (root / "config.json").write_text(json.dumps(dict(actual_timesteps=count)))
            with zipfile.ZipFile(root / "latest_model.zip", "w") as archive:
                archive.writestr("data", json.dumps(dict(num_timesteps=count, _n_updates=14999)))
            with (root / "latest_replay_buffer.pkl").open("wb") as stream:
                pickle.dump(SimpleNamespace(buffer_size=300000, pos=count - 1, n_envs=1, full=False), stream)
            (root / "best").mkdir()
            (root / "best/best_model.zip").write_bytes(b"first" if first else b"second")
            metrics = json.dumps(dict(success_rate=0.0))
            (root / "best/selection.json").write_text(metrics)
            (root / "evaluation.json").write_text(metrics)

        with tempfile.TemporaryDirectory() as folder, patch.object(pipeline, "train", fake_train), \
                redirect_stdout(io.StringIO()):
            root = Path(folder) / "run"
            selected = pipeline.run(output_dir=root, total_timesteps=20004, press_progress_after=20000,
                                    n_envs=1, render=True, device="cuda", opening_device="cuda")
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(call["render"] and call["n_envs"] == 1 for call in calls))
            self.assertEqual(calls[1]["resume_checkpoint"], root / "phase1/latest_model.zip")
            self.assertEqual(calls[1]["replay_buffer_path"], root / "phase1/latest_replay_buffer.pkl")
            self.assertTrue(calls[1]["upgrade_press_progress"])
            self.assertEqual(selected.read_bytes(), b"second")
            manifest = json.loads((root / "pipeline.json").read_text())
            self.assertTrue(manifest["render"])
            self.assertEqual(manifest["phase1_snapshot"]["replay_transitions"], 19999)
            self.assertIn("train/train_sac_render.py", {key.replace("\\", "/") for key in manifest["source_sha256"]})

    def test_rendered_worker_really_updates_sac_and_closes_viewer(self):
        viewer = MagicMock()
        viewer.cam.lookat = np.zeros(3)
        viewer.is_running.return_value = True
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()), \
                patch("mujoco.viewer.launch_passive", return_value=viewer) as launch:
            recipe = replace(training.SACLearnSettings(), n_steps=3, buffer_size=64,
                             batch_size=4, scratch_learning_starts=4)
            checkpoint = training.train(output_dir=folder, total_timesteps=16, n_envs=1, render=True,
                settings=recipe, reward_config=training.LearnRewardConfig.guided(), device="cpu",
                env_config=replace(training.default_door_config(), max_steps=4), eval_freq=8, eval_episodes=1)
            agent, env = training.load_agent(checkpoint)
            try:
                self.assertGreater(agent._n_updates, 0)
                self.assertEqual(agent.num_timesteps, 16)
                self.assertEqual(agent.n_envs, 1)
            finally:
                env.close()
            launch.assert_called_once()
            self.assertGreaterEqual(viewer.sync.call_count, 16)
            viewer.close.assert_called_once()
            self.assertTrue(json.loads((Path(folder) / "config.json").read_text())["render"])


if __name__ == "__main__":
    unittest.main()
