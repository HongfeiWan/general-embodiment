from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tools.data_chain import analyze_quality as quality


class AnalyzeQualityTests(unittest.TestCase):
    def test_source_date_prefers_raw_episode_id(self) -> None:
        row = {"teleop_stack_metadata": {"raw_episode_id": "episode_20260624T090839Z_000000"}}
        self.assertEqual(quality.source_date(row), "2026-06-24")

    def test_fingerprint_changes_with_analysis_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "episode.parquet"
            video_path = Path(tmp) / "episode.mp4"
            data_path.write_text("data", encoding="utf-8")
            video_path.write_text("video", encoding="utf-8")
            row = {"episode_index": 1, "length": 2, "teleop_stack_metadata": {"source": "a"}}
            first = quality.episode_fingerprint(row, data_path, {"ego_view": video_path}, {"skip_video": True})
            second = quality.episode_fingerprint(row, data_path, {"ego_view": video_path}, {"skip_video": False})
            self.assertNotEqual(first, second)

    def test_estimate_lag_recovers_known_shift(self) -> None:
        x = np.sin(np.linspace(0, 8 * np.pi, 120)).reshape(-1, 1)
        action = np.cumsum(x, axis=0)
        state = np.vstack([np.zeros((3, 1)), action[:-3]])
        lag, corr = quality.estimate_lag(action, state, max_lag=10)
        self.assertEqual(lag, 3)
        self.assertIsNotNone(corr)
        self.assertGreater(float(corr), 0.9)

    def test_hand_metrics_handles_empty_arrays(self) -> None:
        row: dict[str, object] = {}
        quality.hand_metrics(row, np.zeros((0, 0)), np.zeros((0, 0)), fps=10.0)
        self.assertIn("finger_angle_mean", row)
        self.assertTrue(np.isnan(row["finger_angle_mean"]))
        self.assertEqual(row["closing_duration"], 0.0)

    def test_issue_scores_surface_bad_episodes(self) -> None:
        episodes = pd.DataFrame(
            [
                {
                    "episode_index": 1,
                    "source_date": "2026-06-24",
                    "status": "ok",
                    "parquet_rows": 100,
                    "frame_count_match": True,
                    "video_frame_count_match": True,
                    "finite_state": True,
                    "finite_action": True,
                },
                {
                    "episode_index": 2,
                    "source_date": "2026-06-24",
                    "status": "ok",
                    "parquet_rows": 100,
                    "frame_count_match": False,
                    "video_frame_count_match": True,
                    "finite_state": True,
                    "finite_action": True,
                },
            ]
        )
        issues = quality.build_issue_rows(episodes, pd.DataFrame(), pd.DataFrame())
        scored = quality.apply_quality_scores(episodes, issues)
        bad = scored.loc[scored["episode_index"] == 2].iloc[0]
        good = scored.loc[scored["episode_index"] == 1].iloc[0]
        self.assertEqual(int(bad["error_count"]), 1)
        self.assertLess(float(bad["quality_score"]), float(good["quality_score"]))
        self.assertIn("state_action_frame_mismatch", str(bad["issue_summary"]))


if __name__ == "__main__":
    unittest.main()
