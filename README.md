# general-embodiment
致力于打造一套通用的具身pipline

## Mission2 数据链路

Nero L10 mission2 的 raw、LeRobot v2、trimmed 测试数据和关键处理代码已经迁入本仓库。

- 数据入口：`missions/nero/mission2/`
- 数据组织脚本：`tools/data_chain/organize_mission_data.py`
- raw -> LeRobot v2 导出：`tools/data_chain/export_lerobot_dataset.py`
- 视频裁切工具：`tools/data_chain/trim_lerobot_episode_viewer.py`
- trimmed -> smooth 平滑生成：`tools/data_chain/smooth_action_commands.py`
- 统一 smooth 数据质量分析：`tools/data_chain/analyze_quality.py`
- smooth 对比分析：`tools/data_chain/plot_trimmed_vs_smooth_action.py`、`tools/data_chain/compare_trimmed_vs_smooth_relative.py`
- 数据质量标准：`docs/embodied_data_quality_standard.md`
- Agent 操作说明：`Agent.md`

示例：对 canonical smooth 数据集做统一质量检查。报告、缓存、增量日志都会写入 `missions/nero/mission2/quality/`：

```bash
python3 tools/data_chain/analyze_quality.py \
  --dataset-dir missions/nero/mission2/smooth \
  --output-dir missions/nero/mission2/quality \
  --raw-root missions/nero/mission2/raw
```

主报告输出到 `missions/nero/mission2/quality/index.html`。再次运行时会复用 `quality/cache/processed_episodes.jsonl`，只分析新增或变更过的 episode；需要全量重算时加 `--force`。

快速检查可以先跳过视频和 raw 时间同步扫描：

```bash
python3 tools/data_chain/analyze_quality.py --skip-video --skip-timing
```

旧的 `analyze_smooth_*` 和 `check_lerobot_batch_quality.py` 脚本仍保留用于读取历史报告或兼容临时流程，但新的质量入口统一使用 `analyze_quality.py`。
