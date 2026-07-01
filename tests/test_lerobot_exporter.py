from __future__ import annotations

import unittest

from teleop_stack.data_capture.exporters.lerobot import (
    L10_CANONICAL_JOINT_ORDER,
    _action_position_xyz_to_state_frame,
    _action_quat_xyzw_to_state_frame_rot6d,
    _quat_xyzw_to_rot6d,
    _rokae_linker_l10_action_vector_impl,
    _rokae_linker_l10_state_vector_impl,
    _select_video,
    _video_frame_records,
)


class LeRobotExporterTests(unittest.TestCase):
    def test_video_frame_records_falls_back_to_authoritative_video_log(self) -> None:
        records = _video_frame_records(
            frames=[{"monotonic_ts_s": 1.0, "videos": []}],
            video_log_records=[
                {
                    "monotonic_ts_s": 1.00,
                    "camera_name": "realsense_head",
                    "relative_path": "video/realsense_head.mp4",
                    "frame_index": 10,
                    "width": 1280,
                    "height": 800,
                },
                {
                    "monotonic_ts_s": 1.02,
                    "camera_name": "realsense_wrist",
                    "relative_path": "video/wrist_d405_rgb.mp4",
                    "frame_index": 20,
                    "width": 640,
                    "height": 480,
                },
            ],
            camera_names=("realsense_head", "realsense_wrist"),
        )

        self.assertEqual(len(records), 1)
        self.assertIsNotNone(_select_video(records[0]["videos"], "realsense_head"))
        self.assertIsNotNone(_select_video(records[0]["videos"], "realsense_wrist"))
        self.assertAlmostEqual(float(records[0]["monotonic_ts_s"]), 1.01)

    def test_video_frame_records_requires_all_selected_cameras(self) -> None:
        records = _video_frame_records(
            frames=[],
            video_log_records=[
                {
                    "monotonic_ts_s": 1.00,
                    "camera_name": "realsense_head",
                    "relative_path": "video/realsense_head.mp4",
                    "frame_index": 10,
                    "width": 1280,
                    "height": 800,
                },
            ],
            camera_names=("realsense_head", "realsense_wrist"),
        )

        self.assertEqual(records, [])

    def test_state_rot6d_uses_groot_row_major_first_two_rows(self) -> None:
        # 90 degrees around x: [[1,0,0], [0,0,-1], [0,1,0]]
        rot6d = _quat_xyzw_to_rot6d([2**-0.5, 0.0, 0.0, 2**-0.5])

        self.assertEqual(len(rot6d), 6)
        expected = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        for actual, wanted in zip(rot6d, expected):
            self.assertAlmostEqual(actual, wanted)

    def test_action_rot6d_is_mapped_into_state_frame_before_export(self) -> None:
        rot6d = _action_quat_xyzw_to_state_frame_rot6d([0.0, 0.0, 0.0, 1.0])

        self.assertEqual(len(rot6d), 6)
        expected = [1.0, 0.0, 0.0, 0.0, -1.0, 0.0]
        for actual, wanted in zip(rot6d, expected):
            self.assertAlmostEqual(actual, wanted)

    def test_action_position_xyz_is_mapped_into_state_frame_before_export(self) -> None:
        self.assertEqual(_action_position_xyz_to_state_frame([1.0, 2.0, 3.0]), [3.0, 1.0, 2.0])

    def test_full_orientation_component_names_are_row_major(self) -> None:
        hand_payload = {
            "joint_names": list(L10_CANONICAL_JOINT_ORDER),
            "joint_positions": [0.0] * len(L10_CANONICAL_JOINT_ORDER),
        }
        state = _rokae_linker_l10_state_vector_impl(
            {
                "arm_joint_positions": [0.0] * 7,
                "arm_ee_pose": {"position_xyz": [0.0, 0.0, 0.0], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]},
                "hand_joint_positions": [0.0] * len(L10_CANONICAL_JOINT_ORDER),
                "raw_snapshot": {"hand_state": hand_payload},
            },
            include_rot6d=True,
        )
        action = _rokae_linker_l10_action_vector_impl(
            {
                "safe_action": {
                    "ee_target": {
                        "position_xyz": [1.0, 2.0, 3.0],
                        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                    "hand_target": hand_payload,
                },
            },
            "safe_action",
            include_rot6d=True,
        )

        self.assertEqual(
            state.names[10:16],
            [
                "arm_eef_rot6d.r00",
                "arm_eef_rot6d.r01",
                "arm_eef_rot6d.r02",
                "arm_eef_rot6d.r10",
                "arm_eef_rot6d.r11",
                "arm_eef_rot6d.r12",
            ],
        )
        self.assertEqual(
            action.names[3:9],
            [
                "arm_eef_rot6d_target.r00",
                "arm_eef_rot6d_target.r01",
                "arm_eef_rot6d_target.r02",
                "arm_eef_rot6d_target.r10",
                "arm_eef_rot6d_target.r11",
                "arm_eef_rot6d_target.r12",
            ],
        )
        self.assertEqual(action.values[:3], [3.0, 1.0, 2.0])
        self.assertEqual(action.values[3:9], [1.0, 0.0, 0.0, 0.0, -1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
