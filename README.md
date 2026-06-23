# general-embodiment
致力于打造一套通用的具身pipline

## Mission2 数据链路

Nero L10 mission2 的 raw、LeRobot v2、trimmed 测试数据和关键处理代码已经迁入本仓库。

- 数据入口：`missions/nero/mission2/`
- 数据组织脚本：`tools/data_chain/organize_mission_data.py`
- raw -> LeRobot v2 导出：`tools/data_chain/export_lerobot_dataset.py`
- 视频裁切工具：`tools/data_chain/trim_lerobot_episode_viewer.py`
- trimmed -> smooth 平滑生成：`tools/data_chain/smooth_action_commands.py`
- smooth 日期质量分析：`tools/data_chain/analyze_smooth_by_date.py`
- smooth 对比分析：`tools/data_chain/plot_trimmed_vs_smooth_action.py`、`tools/data_chain/compare_trimmed_vs_smooth_relative.py`
- Agent 操作说明：`Agent.md`
