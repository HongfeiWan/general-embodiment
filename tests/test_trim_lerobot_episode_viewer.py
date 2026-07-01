from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from tools.data_chain.trim_lerobot_episode_viewer import _eef9d_payload


def _new_state_frame_info() -> dict[str, object]:
    row_major_names = [
        "arm_eef_pos.x",
        "arm_eef_pos.y",
        "arm_eef_pos.z",
        "arm_eef_rot6d.r00",
        "arm_eef_rot6d.r01",
        "arm_eef_rot6d.r02",
        "arm_eef_rot6d.r10",
        "arm_eef_rot6d.r11",
        "arm_eef_rot6d.r12",
    ]
    return {
        "features": {
            "observation.state": {"names": row_major_names},
            "action": {
                "names": [
                    name.replace("arm_eef_pos.", "arm_eef_pos_target.").replace(
                        "arm_eef_rot6d.", "arm_eef_rot6d_target."
                    )
                    for name in row_major_names
                ]
            },
        },
        "teleop_stack": {
            "arm_action_semantics": "absolute_wrist_pose_xyz_rot6d_target_in_state_frame",
            "rot6d_convention": "row_major_first_two_rows_[r00,r01,r02,r10,r11,r12]",
            "action_rot6d_frame_transform": {"formula": "R_state ~= L @ R_action @ R"},
        },
    }


class TrimLeRobotEpisodeViewerTests(unittest.TestCase):
    def test_eef9d_payload_reads_direct_modality_alias(self) -> None:
        df = pd.DataFrame(
            {
                "observation.state": [np.arange(16, dtype=np.float64), np.arange(16, dtype=np.float64) + 1.0],
                "action": [
                    [10.0, 11.0, 12.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [11.0, 12.0, 13.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                ],
                "timestamp": [0.0, 0.1],
            }
        )
        payload = _eef9d_payload(
            df,
            {
                "state": {"eef_9d": {"start": 7, "end": 16}},
                "action": {"eef_9d": {"start": 0, "end": 9}},
            },
            _new_state_frame_info(),
        )

        self.assertIsNone(payload["error"])
        np.testing.assert_allclose(payload["state"][0], np.arange(7, 16, dtype=np.float64))
        np.testing.assert_allclose(payload["action"][0], [10.0, 11.0, 12.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        self.assertAlmostEqual(payload["rmse"][0], 3.0)

    def test_eef9d_payload_falls_back_to_contiguous_pos_and_rot_slices(self) -> None:
        df = pd.DataFrame(
            {
                "observation.state": [np.arange(16, dtype=np.float64)],
                "action": [[0.0, 0.0, 0.0, 3.0, 4.0, 5.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
            }
        )
        payload = _eef9d_payload(
            df,
            {
                "state": {
                    "arm_eef_pos": {"start": 7, "end": 10},
                    "arm_eef_rot6d": {"start": 10, "end": 16},
                },
                "action": {
                    "arm_eef_pos_target": {"start": 3, "end": 6},
                    "arm_eef_rot6d_target": {"start": 6, "end": 12},
                },
            },
            _new_state_frame_info(),
        )

        self.assertIsNone(payload["error"])
        self.assertEqual(payload["state_slice"], [7, 16])
        self.assertEqual(payload["action_slice"], [3, 12])
        np.testing.assert_allclose(payload["state"][0], np.arange(7, 16, dtype=np.float64))
        np.testing.assert_allclose(payload["action"][0], [3.0, 4.0, 5.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    def test_eef9d_payload_standardizes_old_column_major_action_rot6d(self) -> None:
        df = pd.DataFrame(
            {
                "observation.state": [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
                "action": [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
            }
        )
        info = {
            "features": {
                "observation.state": {
                    "names": [
                        "arm_eef_pos.x",
                        "arm_eef_pos.y",
                        "arm_eef_pos.z",
                        "arm_eef_rot6d.r00",
                        "arm_eef_rot6d.r01",
                        "arm_eef_rot6d.r02",
                        "arm_eef_rot6d.r10",
                        "arm_eef_rot6d.r11",
                        "arm_eef_rot6d.r12",
                    ]
                },
                "action": {
                    "names": [
                        "arm_eef_pos_target.x",
                        "arm_eef_pos_target.y",
                        "arm_eef_pos_target.z",
                        "arm_eef_rot6d_target.r00",
                        "arm_eef_rot6d_target.r10",
                        "arm_eef_rot6d_target.r20",
                        "arm_eef_rot6d_target.r01",
                        "arm_eef_rot6d_target.r11",
                        "arm_eef_rot6d_target.r21",
                    ]
                },
            }
        }
        payload = _eef9d_payload(
            df,
            {
                "state": {"eef_9d": {"start": 0, "end": 9}},
                "action": {"eef_9d": {"start": 0, "end": 9}},
            },
            info,
        )

        self.assertIsNone(payload["error"])
        self.assertEqual(payload["action_rot6d_convention"], "column_major")
        self.assertFalse(payload["action_already_in_state_frame"])
        np.testing.assert_allclose(payload["action"][0, 3:9], [1.0, 0.0, 0.0, 0.0, -1.0, 0.0])

    def test_eef9d_payload_does_not_remap_new_state_frame_action_rot6d_twice(self) -> None:
        df = pd.DataFrame(
            {
                "observation.state": [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
                "action": [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0]],
            }
        )
        payload = _eef9d_payload(
            df,
            {
                "state": {"eef_9d": {"start": 0, "end": 9}},
                "action": {"eef_9d": {"start": 0, "end": 9}},
            },
            _new_state_frame_info(),
        )

        self.assertIsNone(payload["error"])
        self.assertEqual(payload["action_rot6d_convention"], "row_major")
        self.assertTrue(payload["action_already_in_state_frame"])
        np.testing.assert_allclose(payload["action"][0, 3:9], [1.0, 0.0, 0.0, 0.0, -1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
