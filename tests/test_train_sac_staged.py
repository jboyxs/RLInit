"""Check callback-boundary pairing and cold-only orchestration; no task claims."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import train.train_sac_staged as staged


def model_file(path, timesteps):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data", json.dumps(dict(num_timesteps=timesteps, _n_updates=14996)))


class StagedRecipeTests(unittest.TestCase):
    def write_pair(self, root, *, model_steps=20000, count=19996, full=False):
        model_file(root / "latest_model.zip", model_steps)
        with (root / "latest_replay_buffer.pkl").open("wb") as stream:
            pickle.dump(SimpleNamespace(buffer_size=75000, pos=count // 4,
                                        n_envs=4, full=full), stream)

    def test_callback_pair_requires_pre_storage_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_pair(root)
            model, replay, metadata = staged.periodic_pair(root, timesteps=20000, n_envs=4)
            self.assertEqual(model.name, "latest_model.zip")
            self.assertEqual(replay.name, "latest_replay_buffer.pkl")
            self.assertEqual(metadata["replay_transitions"], 19996)
            self.write_pair(root, count=20000)
            with self.assertRaisesRegex(ValueError, "replay"):
                staged.periodic_pair(root, timesteps=20000, n_envs=4)

    def test_stale_or_full_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_pair(root, model_steps=10000)
            with self.assertRaisesRegex(ValueError, "model"):
                staged.periodic_pair(root, timesteps=20000, n_envs=4)
            self.write_pair(root, full=True)
            with self.assertRaisesRegex(ValueError, "replay"):
                staged.periodic_pair(root, timesteps=20000, n_envs=4)

    def test_pipeline_uses_own_periodic_pair_and_preserves_successful_early_model(self):
        calls = []

        def fake_train(**kwargs):
            calls.append(kwargs)
            root = kwargs["output_dir"]
            root.mkdir(parents=True)
            first = root.name == "phase1"
            (root / "config.json").write_text(json.dumps(dict(actual_timesteps=20000 if first else 20004)))
            self.write_pair(root)
            model_file(root / "final_model.zip", 20000 if first else 20004)
            (root / "best").mkdir()
            (root / "best/best_model.zip").write_bytes(b"mock-first" if first else b"mock-second")
            selection = dict(success_rate=.8 if first else .2)
            (root / "best/selection.json").write_text(json.dumps(selection))
            (root / "evaluation.json").write_text(json.dumps(selection))

        with tempfile.TemporaryDirectory() as folder, patch.object(staged, "train", fake_train):
            root = Path(folder) / "run"
            with redirect_stdout(io.StringIO()):
                selected = staged.run(output_dir=root, total_timesteps=20004,
                                      press_progress_after=20000, device="cuda", opening_device="cpu")
            self.assertNotIn("resume_checkpoint", calls[0])
            self.assertEqual(calls[0]["device"], "cuda")
            self.assertEqual(calls[1]["device"], "cpu")
            self.assertEqual(calls[1]["resume_checkpoint"], root / "phase1/latest_model.zip")
            self.assertEqual(calls[1]["replay_buffer_path"], root / "phase1/latest_replay_buffer.pkl")
            self.assertTrue(calls[1]["upgrade_press_progress"])
            self.assertEqual(selected.read_bytes(), b"mock-first")
            manifest = json.loads((root / "pipeline.json").read_text())
            self.assertEqual(manifest["selected_phase"], "phase1")
            self.assertEqual(manifest["phase1_snapshot"]["replay_transitions"], 19996)


if __name__ == "__main__":
    unittest.main()
