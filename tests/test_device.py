"""Device selection without requiring a GPU on the test host."""

import unittest
from unittest.mock import patch

from rlinit.algorithmhand.device import resolve_device


class DeviceTests(unittest.TestCase):
    def test_auto_cpu_fallback_and_explicit_cuda_error(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(str(resolve_device()), "cpu")
            with self.assertRaises(RuntimeError):
                resolve_device("cuda")

    def test_auto_cuda_and_cpu_override(self):
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(str(resolve_device()), "cuda")
            self.assertEqual(str(resolve_device("cuda:0")), "cuda:0")
            self.assertEqual(str(resolve_device("cpu")), "cpu")


if __name__ == "__main__":
    unittest.main()
