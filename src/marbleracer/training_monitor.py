from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pandas as pd


SECTION_MARKERS = (23.0, 57.5, 71.5, 79.5, 89.0, 101.0)


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else {}


def _to_bool(value: object) -> bool:
    return bool(value)


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    if math.isnan(number):
        return None
    return number


def _to_int(value: object) -> int | None:
    if value is None:
        return None
    number = float(value)
    if math.isnan(number):
        return None
    return int(number)


def load_metrics(report_dir: Path) -> pd.DataFrame:
    metrics_csv = report_dir / "checkpoint_metrics.csv"
    if not metrics_csv.exists():
        return pd.DataFrame()
    return pd.read_csv(metrics_csv).sort_values(["timesteps", "phase"]).reset_index(drop=True)


def _load_live_training_state(report_dir: Path) -> dict[str, object]:
    log_path = report_dir.parent / "overnight.log"
    live_status_path = report_dir / "live_status.json"
    summary = _load_json(report_dir / "summary.json")
    checkpoint_interval = _to_int(summary.get("checkpoint_interval"))
    live_status = _load_json(live_status_path)
    if live_status:
        return {
            "log_path": str(log_path),
            "current_timesteps": _to_int(live_status.get("current_timesteps")),
            "current_fps": _to_float(live_status.get("current_fps")),
            "checkpoint_interval": _to_int(live_status.get("checkpoint_interval")) or checkpoint_interval,
            "in_progress_generation": _to_int(live_status.get("in_progress_generation")),
            "latest_saved_timesteps": _to_int(live_status.get("latest_saved_timesteps")),
        }
    if not log_path.exists():
        return {
            "log_path": str(log_path),
            "current_timesteps": None,
            "current_fps": None,
            "checkpoint_interval": checkpoint_interval,
            "in_progress_generation": None,
            "latest_saved_timesteps": None,
        }

    return {
        "log_path": str(log_path),
        "current_timesteps": None,
        "current_fps": None,
        "checkpoint_interval": checkpoint_interval,
        "in_progress_generation": None,
        "latest_saved_timesteps": None,
    }


def build_generation_index(report_dir: Path) -> dict[str, object]:
    metrics = load_metrics(report_dir)
    if metrics.empty:
        return {
            "checkpoint_count": 0,
            "latest_checkpoint": None,
            "best_progress_checkpoint": None,
            "best_reward_checkpoint": None,
            "generations": [],
        }

    default_rows = metrics.loc[metrics["phase"].astype(str).str.contains("phase2_default_full")].copy()
    if default_rows.empty:
        default_rows = metrics.copy()
    default_rows = default_rows.sort_values("timesteps").reset_index(drop=True)
    latest = default_rows.iloc[-1]
    best_progress = default_rows.sort_values(["max_progress", "reward"], ascending=[False, False]).iloc[0]
    best_reward = default_rows.sort_values(["reward", "max_progress"], ascending=[False, False]).iloc[0]

    generations: list[dict[str, object]] = []
    latest_path = str(latest["checkpoint_path"])
    best_progress_path = str(best_progress["checkpoint_path"])
    best_reward_path = str(best_reward["checkpoint_path"])
    for _, row in default_rows.iterrows():
        max_progress = float(row["max_progress"])
        entry = {
            "phase": str(row["phase"]),
            "generation": int(row["generation"]),
            "timesteps": int(row["timesteps"]),
            "checkpoint_path": str(row["checkpoint_path"]),
            "checkpoint_name": Path(str(row["checkpoint_path"])).name,
            "finished": _to_bool(row.get("finished", False)),
            "finish_time": _to_float(row.get("finish_time")),
            "max_progress": max_progress,
            "reward": float(row["reward"]),
            "steps": int(row["steps"]),
            "resets": int(row["resets"]),
            "termination_reason": str(row["termination_reason"]),
            "is_latest": str(row["checkpoint_path"]) == latest_path,
            "is_best_progress": str(row["checkpoint_path"]) == best_progress_path,
            "is_best_reward": str(row["checkpoint_path"]) == best_reward_path,
        }
        for marker in SECTION_MARKERS:
            entry[f"section_reached_{marker:.1f}m"] = max_progress >= marker
        generations.append(entry)

    return {
        "checkpoint_count": len(generations),
        "latest_checkpoint": latest_path,
        "best_progress_checkpoint": best_progress_path,
        "best_reward_checkpoint": best_reward_path,
        "generations": generations,
    }


def trainer_active(report_dir: Path) -> bool:
    try:
        result = subprocess.run(
            ["ps", "aux"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    absolute_needle = str(report_dir.resolve())
    relative_needle = report_dir.as_posix()
    report_name = report_dir.name
    for line in result.stdout.splitlines():
        if "train_bot_policy.py" not in line:
            continue
        if absolute_needle in line or relative_needle in line:
            return True
        if "--report-dir" in line and report_name in line:
            return True
    return False


def build_dashboard_summary(run_dir: Path, report_dir: Path) -> dict[str, object]:
    snapshot_summary = _load_json(run_dir / "summary.json")
    report_summary = _load_json(report_dir / "summary.json")
    generation_index = build_generation_index(report_dir)
    live_state = _load_live_training_state(report_dir)
    is_active = trainer_active(report_dir)
    latest_generation = generation_index["generations"][-1] if generation_index["generations"] else None
    best_generation = None
    for entry in generation_index["generations"]:
        if entry.get("is_best_progress"):
            best_generation = entry
            break

    current_timesteps = _to_int(live_state.get("current_timesteps"))
    checkpoint_interval = _to_int(live_state.get("checkpoint_interval"))
    current_fps = _to_float(live_state.get("current_fps"))
    latest_timesteps = None if latest_generation is None else int(latest_generation["timesteps"])
    latest_generation_id = None if latest_generation is None else int(latest_generation["generation"])
    next_checkpoint_timesteps = None
    timesteps_to_next_checkpoint = None
    in_progress_generation = _to_int(live_state.get("in_progress_generation"))
    eta_seconds = None
    if (
        current_timesteps is not None
        and latest_timesteps is not None
        and checkpoint_interval is not None
        and checkpoint_interval > 0
        and current_timesteps >= latest_timesteps
    ):
        completed_intervals = max(0, (current_timesteps - latest_timesteps) // checkpoint_interval)
        next_checkpoint_timesteps = latest_timesteps + checkpoint_interval * (completed_intervals + 1)
        timesteps_to_next_checkpoint = max(0, next_checkpoint_timesteps - current_timesteps)
        if in_progress_generation is None and latest_generation_id is not None:
            in_progress_generation = latest_generation_id + completed_intervals + 1
        if current_fps and timesteps_to_next_checkpoint is not None:
            eta_seconds = timesteps_to_next_checkpoint / max(1, current_fps)

    summary = {
        "trainer_active": is_active,
        "device": report_summary.get("device"),
        "workers": _to_int(report_summary.get("workers")),
        "checkpoint_count": generation_index["checkpoint_count"],
        "latest_checkpoint": None if latest_generation is None else latest_generation["checkpoint_path"],
        "latest_timesteps": None if latest_generation is None else latest_generation["timesteps"],
        "latest_generation": None if latest_generation is None else latest_generation["generation"],
        "latest_max_progress": None if latest_generation is None else latest_generation["max_progress"],
        "latest_reward": None if latest_generation is None else latest_generation["reward"],
        "latest_finish_time": None if latest_generation is None else latest_generation["finish_time"],
        "latest_termination_reason": None if latest_generation is None else latest_generation["termination_reason"],
        "current_timesteps": current_timesteps,
        "current_fps": current_fps,
        "checkpoint_interval": checkpoint_interval,
        "in_progress_generation": in_progress_generation,
        "next_checkpoint_timesteps": next_checkpoint_timesteps,
        "timesteps_to_next_checkpoint": timesteps_to_next_checkpoint,
        "eta_seconds_to_next_checkpoint": eta_seconds,
        "best_progress_checkpoint": None if best_generation is None else best_generation["checkpoint_path"],
        "best_progress_generation": None if best_generation is None else best_generation["generation"],
        "best_progress_value": None if best_generation is None else best_generation["max_progress"],
        "snapshot_plots_dir": str(run_dir / "plots"),
        "snapshot_replay_dir": str(run_dir / "replay_3d_latest"),
        "latest_replay_video": str(run_dir / "replay_3d_latest" / "training_ghosts_latest_active.mp4"),
        "latest_replay_frame": str(run_dir / "replay_3d_latest" / "training_ghosts_latest_active.png"),
        "report_metrics_csv": str(report_dir / "checkpoint_metrics.csv"),
        "training_log": str(live_state["log_path"]),
        "run_dir": str(run_dir),
        "report_dir": str(report_dir),
        "asset_version": None if latest_generation is None else f"{latest_generation['timesteps']}-{latest_generation['generation']}",
    }
    for key, value in snapshot_summary.items():
        if key not in summary:
            summary[key] = value
    if not is_active:
        summary["current_timesteps"] = None
        summary["current_fps"] = None
        summary["in_progress_generation"] = None
        summary["next_checkpoint_timesteps"] = None
        summary["timesteps_to_next_checkpoint"] = None
        summary["eta_seconds_to_next_checkpoint"] = None
    return summary
