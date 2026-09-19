from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from pickle import dump

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research_impl"))
from validate_data import validate_episode
from nttg_label_buffer import label_episode


class DataContractTest(unittest.TestCase):
    def test_nttg_progress_and_failure(self):
        success = label_episode([{"rewards": 0.0}, {"rewards": 0.0}, {"rewards": 1.0}])
        self.assertEqual([row["raw_norm"] for row in success], [0.0, 0.5, 1.0])
        failure = label_episode([{"rewards": 0.0}, {"rewards": 0.0}])
        self.assertEqual([row["raw_value"] for row in failure], [-500, -500])

    def test_valid_fold_towel_episode(self):
        image = np.zeros((1, 8, 8, 3), dtype=np.uint8)
        obs = {"external": image, "wrist": image, "state": np.zeros(9, dtype=np.float32)}
        step = {"observations": obs, "next_observations": obs,
                "actions": np.zeros(7, dtype=np.float32), "rewards": 1.0}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "transitions_test.pkl"
            with path.open("wb") as f:
                dump([step], f)
            self.assertEqual(validate_episode(path), (1, True))

    def test_rejects_six_dimensional_action(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        obs = {"external": image, "wrist": image, "state": np.zeros(9, dtype=np.float32)}
        step = {"observations": obs, "next_observations": obs,
                "actions": np.zeros(6, dtype=np.float32), "rewards": 1.0}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "transitions_test.pkl"
            with path.open("wb") as f:
                dump([step], f)
            with self.assertRaisesRegex(ValueError, "7D action"):
                validate_episode(path)


if __name__ == "__main__":
    unittest.main()
