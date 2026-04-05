from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from marbleracer.training_monitor import build_generation_index

SECTION_MARKERS = (23.0, 57.5, 71.5, 79.5, 89.0, 101.0)


def write_dashboard(metrics: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    specs = (
        ("reward", "Reward", "#2563eb"),
        ("max_progress", "Max Progress", "#16a34a"),
        ("steps", "Episode Steps", "#ea580c"),
        ("resets", "Resets", "#dc2626"),
    )
    for axis, (column, title, color) in zip(axes.ravel(), specs, strict=True):
        axis.plot(metrics["timesteps"], metrics[column], marker="o", linewidth=2.0, color=color)
        axis.set_title(title)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "checkpoint_dashboard.png", dpi=180)
    plt.close(fig)


def write_curve(metrics: pd.DataFrame, *, column: str, title: str, ylabel: str, color: str, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(metrics["timesteps"], metrics[column], marker="o", linewidth=2.0, color=color)
    axis.set_title(title)
    axis.set_xlabel("Timesteps")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_section_success(default_rows: pd.DataFrame, output_dir: Path) -> None:
    if default_rows.empty:
        return
    fig, axis = plt.subplots(figsize=(9, 5))
    for marker in SECTION_MARKERS:
        column = f"section_reached_{marker:.1f}m"
        axis.plot(default_rows["timesteps"], default_rows[column], marker="o", linewidth=2.0, label=f"{marker:.1f}m")
    axis.set_title("Default Track Section Success")
    axis.set_xlabel("Timesteps")
    axis.set_ylabel("Reached")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "section_success_curve.png", dpi=180)
    plt.close(fig)


def write_failure_histogram(default_rows: pd.DataFrame, output_dir: Path) -> None:
    if default_rows.empty:
        return
    counts = default_rows.groupby("termination_reason").size().sort_values(ascending=False)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    counts.plot(kind="bar", ax=axis, color="#dc2626")
    axis.set_title("Default Checkpoint Terminations")
    axis.set_xlabel("Termination Reason")
    axis.set_ylabel("Count")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "failure_histogram.png", dpi=180)
    plt.close(fig)
    (output_dir / "latest_failure_counts.json").write_text(json.dumps(counts.to_dict(), indent=2) + "\n")


def render_3d_snapshot(*, python_bin: str, metrics_csv: Path, output_dir: Path, width: int, height: int, fps: int, playback_speed: float) -> None:
    subprocess.run(
        [
            python_bin,
            "scripts/render_training_rollouts_3d.py",
            "--metrics-csv",
            str(metrics_csv),
            "--output-dir",
            str(output_dir),
            "--width",
            str(width),
            "--height",
            str(height),
            "--fps",
            str(fps),
            "--playback-speed",
            str(playback_speed),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Update training plots and the latest 3D replay snapshot from current checkpoint metrics.")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-bin", type=str, default=".venv/bin/python")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--playback-speed", type=float, default=0.65)
    args = parser.parse_args()

    metrics_csv = args.report_dir / "checkpoint_metrics.csv"
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {metrics_csv}")

    metrics = pd.read_csv(metrics_csv).sort_values(["timesteps", "phase"]).reset_index(drop=True)
    if metrics.empty:
        raise RuntimeError("Checkpoint metrics CSV is empty.")

    output_dir = args.output_dir
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    default_rows = metrics.loc[metrics["phase"].astype(str).str.contains("phase2_default_full")].copy()
    for marker in SECTION_MARKERS:
        default_rows[f"section_reached_{marker:.1f}m"] = (default_rows["max_progress"] >= marker).astype(float)

    metrics.to_csv(output_dir / "checkpoint_metrics_snapshot.csv", index=False)
    default_rows.to_csv(output_dir / "default_checkpoint_metrics_snapshot.csv", index=False)
    generation_index = build_generation_index(args.report_dir)
    (output_dir / "generation_index.json").write_text(json.dumps(generation_index, indent=2) + "\n")
    write_dashboard(metrics, plots_dir)
    write_curve(default_rows, column="reward", title="Checkpoint Reward", ylabel="Reward", color="#2563eb", output_path=plots_dir / "reward_curve.png")
    write_curve(default_rows, column="max_progress", title="Checkpoint Max Progress", ylabel="Meters", color="#16a34a", output_path=plots_dir / "max_progress_curve.png")
    write_section_success(default_rows, plots_dir)
    write_failure_histogram(default_rows, plots_dir)

    render_3d_snapshot(
        python_bin=args.python_bin,
        metrics_csv=metrics_csv,
        output_dir=output_dir / "replay_3d_latest",
        width=args.width,
        height=args.height,
        fps=args.fps,
        playback_speed=args.playback_speed,
    )

    latest_row = metrics.sort_values("timesteps").iloc[-1].to_dict()
    best_default = default_rows.sort_values(["max_progress", "reward"], ascending=[False, False]).iloc[0].to_dict() if not default_rows.empty else None
    summary = {
        "checkpoint_count": int(len(metrics)),
        "latest_checkpoint": str(latest_row["checkpoint_path"]),
        "latest_timesteps": int(latest_row["timesteps"]),
        "latest_max_progress": float(latest_row["max_progress"]),
        "latest_reward": float(latest_row["reward"]),
        "best_default_checkpoint": None if best_default is None else str(best_default["checkpoint_path"]),
        "best_default_max_progress": None if best_default is None else float(best_default["max_progress"]),
        "generation_index": str(output_dir / "generation_index.json"),
        "plots_dir": str(plots_dir),
        "replay_dir": str(output_dir / "replay_3d_latest"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
