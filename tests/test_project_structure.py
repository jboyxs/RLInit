"""Guard the two-task layout and standalone SAC import boundary."""
import ast
from pathlib import Path
import unittest

from rlinit.algorithmhand.reach_env import ReachEnv
from train import train_hand, train_sb3, train_sac_staged

ROOT = Path(__file__).resolve().parents[1]


class ProjectStructureTests(unittest.TestCase):
    def test_reaching_trainers_share_exactly_the_same_environment(self):
        self.assertIs(train_hand.ReachEnv, ReachEnv)
        self.assertIs(train_sb3.ReachEnv, ReachEnv)

    def test_training_entry_points_are_in_train(self):
        self.assertEqual(
            {path.name for path in (ROOT / "train").glob("*.py")},
            {"__init__.py", "train_hand.py", "train_sb3.py", "train_sac_staged.py",
             "train_sac_modular.py"},
        )
        self.assertFalse((ROOT / "train.py").exists())
        self.assertFalse(list((ROOT / "tests").glob("train*.py")))

    def test_sac_environment_has_no_local_module_dependencies(self):
        path = ROOT / "src/rlinit/algorithmhand/door_sac_learn_env.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0)
                self.assertFalse((node.module or "").startswith(("rlinit", "train", "tests")))
            elif isinstance(node, ast.Import):
                self.assertFalse(any(item.name.startswith(("rlinit", "train", "tests")) for item in node.names))

    def test_sac_trainer_only_imports_its_single_environment(self):
        tree = ast.parse(Path(train_sac_staged.__file__).read_text(encoding="utf-8"))
        local = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0)
                if (node.module or "").startswith(("rlinit", "train", "tests")):
                    local.append(node.module)
            elif isinstance(node, ast.Import):
                local.extend(item.name for item in node.names
                             if item.name.startswith(("rlinit", "train", "tests")))
        self.assertEqual(local, ["rlinit.algorithmhand.door_sac_learn_env"])


if __name__ == "__main__":
    unittest.main()
