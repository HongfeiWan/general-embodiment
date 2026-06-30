from __future__ import annotations

import unittest

from teleop_stack.data_capture.exporters.lerobot import _select_video, _video_frame_records


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


if __name__ == "__main__":
    unittest.main()
