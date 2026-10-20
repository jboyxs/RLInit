"""Numerical-code parity and modular spawn collection."""
import ast
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from train import train_sac_modular, train_sac_staged
from train.sac_staged import config, pipeline, training

ROOT = Path(__file__).resolve().parents[1]


class AlgorithmOnly(ast.NodeTransformer):
    """Exclude changed provenance paths, never an update/reward/selection step."""
    def visit_Assign(self, node):
        for target in node.targets:
            if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                    and target.value.id == "run" and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "recipe_sha256"):
                return None
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if node.name == "run":
            # Optional viewer routing has its own integration tests. With the
            # default render=False, the original training path is unchanged.
            if node.args.kwonlyargs[-1].arg == "render":
                node.args.kwonlyargs.pop()
                node.args.kw_defaults.pop()
            node.body = [item for item in node.body if not (
                isinstance(item, ast.If) and isinstance(item.test, ast.Name) and item.test.id == "render")]
            node.body = [item for item in node.body if not (isinstance(item, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "sources" for target in item.targets))]
        return self.generic_visit(node)


def definitions(paths):
    return {node.name: node for path in paths
            for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


class ModularSACCodeTests(unittest.TestCase):
    def test_all_algorithm_definitions_match_retained_reference(self):
        original = definitions([ROOT / "train/train_sac_staged.py"])
        modular = definitions([*sorted((ROOT / "train/sac_staged").glob("*.py")),
                               ROOT / "train/train_sac_modular.py"])
        for name, node in original.items():
            expected = ast.dump(AlgorithmOnly().visit(node), include_attributes=False)
            with self.subTest(definition=name):
                self.assertIn(name, modular)
                self.assertEqual(expected, ast.dump(AlgorithmOnly().visit(modular[name]),
                                                   include_attributes=False))

    def test_recipe_constants_and_defaults_are_identical(self):
        self.assertEqual(config.ROOT, train_sac_staged.ROOT)
        self.assertEqual(config.FORMAT, train_sac_staged.FORMAT)
        self.assertEqual(config.MONITOR_KEYS, train_sac_staged.MONITOR_KEYS)
        self.assertEqual(asdict(config.SACLearnSettings()), asdict(train_sac_staged.SACLearnSettings()))

    def test_new_implementations_do_not_import_the_reference(self):
        paths = [*sorted((ROOT / "train/sac_staged").glob("*.py")),
                 ROOT / "train/train_sac_modular.py"]
        for path in paths:
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotEqual((node.module or "").split(".")[-1], "train_sac_staged")
                elif isinstance(node, ast.Import):
                    self.assertFalse(any(alias.name.split(".")[-1] == "train_sac_staged" for alias in node.names))

    def test_cli_training_defaults_and_overrides_match(self):
        for arguments in ([], ["--seed", "31", "--timesteps", "20004", "--n-envs", "1",
                               "--press-progress-after", "20000", "--device", "cpu", "--episodes", "1"]):
            calls = []
            for entry in (train_sac_staged, train_sac_modular):
                with patch.object(sys, "argv", ["entry.py", *arguments]), patch.object(entry, "run") as run:
                    entry.main()
                    calls.append(run.call_args)
            self.assertEqual(calls[0], calls[1])

    def test_provenance_records_the_executed_implementation(self):
        paths = config.recipe_sources()
        self.assertIn(ROOT / "train/train_sac_modular.py", paths)
        self.assertIn(ROOT / "train/sac_staged/training.py", paths)
        self.assertNotIn(ROOT / "train/train_sac_staged.py", paths)


class ModularSACExecutionTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_updates_match_reference_networks_and_optimizers(self):
        states = []
        for variant in (train_sac_staged, training):
            with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
                recipe = replace(variant.SACLearnSettings(), n_steps=3, buffer_size=64,
                                 batch_size=4, scratch_learning_starts=4)
                checkpoint = variant.train(output_dir=folder, total_timesteps=16, n_envs=1,
                    settings=recipe, reward_config=variant.LearnRewardConfig.guided(), device="cuda",
                    env_config=replace(variant.default_door_config(), max_steps=4), eval_freq=8, eval_episodes=1)
                agent, env = variant.load_agent(checkpoint, device="cpu")
                try:
                    self.assertGreater(agent._n_updates, 0)
                    states.append((agent.policy.state_dict(), agent.actor.optimizer.state_dict(),
                                   agent.critic.optimizer.state_dict(), agent.num_timesteps, agent._n_updates))
                finally:
                    env.close()

        def equal(first, second):
            if isinstance(first, torch.Tensor):
                self.assertTrue(torch.equal(first, second))
            elif isinstance(first, dict):
                self.assertEqual(first.keys(), second.keys())
                for key in first:
                    equal(first[key], second[key])
            elif isinstance(first, (list, tuple)):
                self.assertEqual(len(first), len(second))
                for left, right in zip(first, second):
                    equal(left, right)
            else:
                self.assertEqual(first, second)

        equal(states[0], states[1])

    def test_modular_trainer_really_updates_with_four_spawned_environments(self):
        for variant in (training,):
            with self.subTest(variant=variant.__name__), tempfile.TemporaryDirectory() as folder:
                recipe = replace(variant.SACLearnSettings(), n_steps=3, buffer_size=64,
                                 batch_size=4, scratch_learning_starts=4)
                with redirect_stdout(io.StringIO()):
                    checkpoint = variant.train(output_dir=folder, total_timesteps=16, n_envs=4,
                        settings=recipe, reward_config=variant.LearnRewardConfig.guided(), device="cpu",
                        env_config=replace(variant.default_door_config(), max_steps=4), eval_freq=8, eval_episodes=1)
                agent, env = variant.load_agent(checkpoint)
                try:
                    self.assertEqual(agent.num_timesteps, 16)
                    self.assertEqual(agent.n_envs, 4)
                    self.assertEqual(agent.n_steps, 3)
                    self.assertGreater(agent._n_updates, 0)
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()
