from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from marbleracer.training_monitor import build_dashboard_summary, build_generation_index


def write_metrics(report_dir: Path) -> None:
    rows = [
        {
            "phase": "phase2_default_full",
            "generation": 1,
            "timesteps": 100000,
            "checkpoint_path": str(report_dir / "phase2_default_full/checkpoints/phase2_default_full_gen_001_0100000.zip"),
            "finished": False,
            "finish_time": None,
            "max_progress": 72.5,
            "reward": 40.0,
            "steps": 3200,
            "resets": 1,
            "termination_reason": "off_track",
        },
        {
            "phase": "phase2_default_full",
            "generation": 2,
            "timesteps": 200000,
            "checkpoint_path": str(report_dir / "phase2_default_full/checkpoints/phase2_default_full_gen_002_0200000.zip"),
            "finished": False,
            "finish_time": None,
            "max_progress": 77.1,
            "reward": 58.0,
            "steps": 3400,
            "resets": 1,
            "termination_reason": "off_track",
        },
        {
            "phase": "phase2_default_full",
            "generation": 3,
            "timesteps": 300000,
            "checkpoint_path": str(report_dir / "phase2_default_full/checkpoints/phase2_default_full_gen_003_0300000.zip"),
            "finished": False,
            "finish_time": None,
            "max_progress": 74.4,
            "reward": 61.0,
            "steps": 3100,
            "resets": 0,
            "termination_reason": "stall",
        },
    ]
    (report_dir / "phase2_default_full/checkpoints").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(report_dir / "checkpoint_metrics.csv", index=False)
    (report_dir / "summary.json").write_text(json.dumps({"device": "mps", "workers": 4}, indent=2) + "\n")


def test_build_generation_index_marks_latest_and_best(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    write_metrics(report_dir)

    index = build_generation_index(report_dir)

    assert index["checkpoint_count"] == 3
    assert index["latest_checkpoint"].endswith("gen_003_0300000.zip")
    assert index["best_progress_checkpoint"].endswith("gen_002_0200000.zip")
    assert index["best_reward_checkpoint"].endswith("gen_003_0300000.zip")
    latest = index["generations"][-1]
    assert latest["is_latest"] is True
    assert latest["is_best_reward"] is True
    assert latest["termination_reason"] == "stall"


def test_build_dashboard_summary_merges_report_and_snapshot(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    write_metrics(report_dir)

    run_dir = tmp_path / "live_latest"
    (run_dir / "replay_3d_latest").mkdir(parents=True)
    (run_dir / "summary.json").write_text(json.dumps({"plots_dir": "ignored-by-summary"}, indent=2) + "\n")

    summary = build_dashboard_summary(run_dir, report_dir)

    assert summary["device"] == "mps"
    assert summary["workers"] == 4
    assert summary["checkpoint_count"] == 3
    assert summary["latest_generation"] == 3
    assert summary["best_progress_generation"] == 2
    assert summary["latest_termination_reason"] == "stall"
    assert summary["latest_replay_video"].endswith("replay_3d_latest/training_ghosts_latest_active.mp4")
