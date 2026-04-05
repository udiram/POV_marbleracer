from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from stable_baselines3 import PPO

from marbleracer.rl_env import MarbleBotTrainingEnv, default_training_config

CHECKPOINT_PATTERN = re.compile(r"(?P<phase>.+)_gen_(?P<generation>\d+)_(?P<timesteps>\d+)\.zip$")
SECTION_MARKERS = (23.0, 57.5, 71.5, 79.5, 89.0, 101.0)


def level_config_for_phase(phase_name: str):
    if "time_trial_short" in phase_name:
        return default_training_config(with_obstacles=False, level_name="time-trial-short")
    return default_training_config(with_obstacles=True, level_name="default")


def sample_track_points(config, *, count: int = 300) -> tuple[np.ndarray, np.ndarray]:
    from marbleracer.physics import ramp_surface_point

    distances = np.linspace(0.0, config.length, count)
    points = np.array([ramp_surface_point(config, float(distance)) for distance in distances], dtype=np.float32)
    return points[:, 0], points[:, 1]


def baseline_action(kind: str, observation: np.ndarray) -> np.ndarray:
    if kind == "coast":
        return np.array([0.0, 0.0], dtype=np.float32)
    lane_error = float(observation[5]) * 0.5
    lateral_speed = float(observation[3]) * 6.0
    speed = float(observation[1]) * 12.0
    target_speed = float(observation[22]) * 12.0
    steer = np.clip(3.0 * lane_error - 1.2 * lateral_speed, -1.0, 1.0)
    brake = np.clip((speed - target_speed) * 0.08, 0.0, 1.0)
    return np.array([steer, brake], dtype=np.float32)


def collect_rollout(config, *, seed: int, policy: PPO | None = None, baseline: str | None = None) -> tuple[pd.DataFrame, dict[str, object]]:
    env = MarbleBotTrainingEnv(config=config)
    observation, _ = env.reset(seed=seed)
    frames: list[dict[str, float | str | int]] = []
    info: dict[str, object] = {"termination_reason": "running"}
    for _ in range(4000):
        if policy is not None:
            action = np.asarray(policy.predict(observation, deterministic=True)[0], dtype=np.float32)
        else:
            action = baseline_action(str(baseline), observation)
        observation, _, terminated, truncated, info = env.step(action)
        snapshot = env.simulation.snapshot()
        frames.append(
            {
                "time": float(snapshot.time),
                "x": float(snapshot.position.x),
                "y": float(snapshot.position.y),
                "z": float(snapshot.position.z),
                "progress": float(info["progress"]),
                "speed": float(snapshot.speed),
                "signed_lateral_offset": float(info["signed_lateral_offset"]),
                "termination_reason": str(info["termination_reason"]),
            }
        )
        if terminated or truncated:
            break
    env.close()
    return pd.DataFrame(frames), info


def write_dashboard(metrics: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("reward", "Reward"),
        ("max_progress", "Max Progress"),
        ("finish_time", "Finish Time"),
        ("resets", "Resets"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, (column, title) in zip(axes.ravel(), specs, strict=True):
        axis.plot(metrics["timesteps"], metrics[column], marker="o", linewidth=2.0, color="#2563eb")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "checkpoint_dashboard.png", dpi=180)
    plt.close(fig)


def write_section_success(default_rows: pd.DataFrame, output_dir: Path) -> None:
    if default_rows.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    columns = [column for column in default_rows.columns if column.startswith("section_reached_")]
    for column in columns:
        ax.plot(default_rows["timesteps"], default_rows[column], marker="o", linewidth=2.0, label=column.replace("section_reached_", ""))
    ax.set_title("Default Track Section Success")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Reached")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "section_success_curve.png", dpi=180)
    plt.close(fig)


def write_failure_histogram(default_rows: pd.DataFrame, output_dir: Path) -> None:
    if default_rows.empty:
        return
    counts = default_rows.groupby("termination_reason").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts.plot(kind="bar", ax=ax, color="#dc2626")
    ax.set_title("Default Checkpoint Terminations")
    ax.set_xlabel("Termination Reason")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "failure_histogram.png", dpi=180)
    plt.close(fig)
    (output_dir / "latest_failure_counts.json").write_text(json.dumps(counts.to_dict(), indent=2) + "\n")


def write_pendulum_trace(frame_df: pd.DataFrame, output_dir: Path) -> None:
    pendulum = frame_df[(frame_df["progress"] >= 65.0) & (frame_df["progress"] <= 80.0)].copy()
    if pendulum.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(pendulum["progress"], pendulum["speed"], color="#2563eb", linewidth=2.0)
    axes[0].set_ylabel("Speed")
    axes[0].grid(alpha=0.2)
    axes[1].plot(pendulum["progress"], pendulum["signed_lateral_offset"], color="#ea580c", linewidth=2.0)
    axes[1].set_xlabel("Progress")
    axes[1].set_ylabel("Lateral Offset")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "pendulum_trace.png", dpi=180)
    plt.close(fig)
    pendulum.to_csv(output_dir / "pendulum_trace.csv", index=False)


def render_comparison_video(tracks: list[tuple[str, pd.DataFrame, str]], *, config, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    track_x, track_y = sample_track_points(config)
    fig, axes = plt.subplots(1, len(tracks), figsize=(5.2 * len(tracks), 4.6), sharex=True, sharey=True)
    if len(tracks) == 1:
        axes = [axes]
    artists = []
    frame_count = min(90, max(len(frames) for _, frames, _ in tracks))
    indices = [
        np.linspace(0, max(0, len(frames) - 1), frame_count, dtype=int) if not frames.empty else np.array([0])
        for _, frames, _ in tracks
    ]
    for axis, (label, frames, color) in zip(axes, tracks, strict=True):
        axis.plot(track_x, track_y, color="#1f2937", linewidth=8, alpha=0.55, solid_capstyle="round")
        axis.plot(track_x, track_y, color="#cbd5e1", linewidth=1.5, alpha=0.85)
        for obstacle in config.obstacles:
            axis.scatter([obstacle.distance_along_ramp], [obstacle.lateral_offset], s=55, c="#f97316", alpha=0.4)
        marble = axis.scatter([], [], s=110, c=color, edgecolors="white", linewidths=0.8, zorder=5)
        trail, = axis.plot([], [], color=color, linewidth=1.5, alpha=0.7)
        text = axis.text(0.02, 0.98, "", transform=axis.transAxes, va="top", ha="left")
        axis.set_title(label)
        axis.set_xlim(-2.0, config.length + 2.0)
        axis.set_ylim(-config.width * 1.1, config.width * 1.1)
        axis.grid(alpha=0.18)
        artists.append((marble, trail, text))
    axes[0].set_ylabel("Lateral Offset")
    fig.suptitle(title)
    fig.tight_layout()

    def _update(frame_idx: int):
        out = []
        for idx, (_label, frames, _color) in enumerate(tracks):
            sample_idx = int(indices[idx][min(frame_idx, len(indices[idx]) - 1)])
            frame = frames.iloc[sample_idx]
            marble, trail, text = artists[idx]
            marble.set_offsets(np.array([[frame["x"], frame["y"]]], dtype=np.float32))
            trail_points = frames.iloc[: sample_idx + 1][["x", "y"]].to_numpy(dtype=np.float32)
            trail.set_data(trail_points[:, 0], trail_points[:, 1])
            text.set_text(f"t={frame['time']:.2f}s\np={frame['progress']:.2f}\nv={frame['speed']:.2f}\n{frame['termination_reason']}")
            out.extend((marble, trail, text))
        return out

    anim = animation.FuncAnimation(fig, _update, frames=frame_count, interval=50, blit=False)
    anim.save(output_path, writer=animation.FFMpegWriter(fps=20, bitrate=1800))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a lightweight qualitative and quantitative evidence pack.")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    checkpoint_paths = sorted(args.report_dir.glob("**/checkpoints/*.zip"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found under {args.report_dir}")

    metrics_path = args.report_dir / "checkpoint_metrics.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    rows: list[dict[str, object]] = []
    default_records: list[tuple[int, PPO, Path]] = []

    for checkpoint_path in checkpoint_paths:
        match = CHECKPOINT_PATTERN.match(checkpoint_path.name)
        if match is None:
            continue
        phase_name = match.group("phase")
        if metrics.empty:
            model = PPO.load(str(checkpoint_path))
            config = level_config_for_phase(phase_name)
            frames, info = collect_rollout(config, seed=args.seed, policy=model)
            row = {
                "phase": phase_name,
                "generation": int(match.group("generation")),
                "timesteps": int(match.group("timesteps")),
                "finish_time": info.get("finish_time"),
                "max_progress": float(frames["progress"].max()),
                "reward": float(frames.iloc[-1]["progress"]) if not frames.empty else 0.0,
                "resets": int(info.get("resets", 0)),
                "termination_reason": str(info["termination_reason"]),
            }
        else:
            match_rows = metrics.loc[metrics["checkpoint_path"] == str(checkpoint_path)]
            if match_rows.empty:
                continue
            row = match_rows.iloc[0].to_dict()
        if "default" in phase_name:
            config = level_config_for_phase(phase_name)
            model = PPO.load(str(checkpoint_path))
            frames, info = collect_rollout(config, seed=args.seed, policy=model)
            for marker in SECTION_MARKERS:
                row[f"section_reached_{marker:.1f}m"] = float(float(frames["progress"].max()) >= marker)
            default_records.append((int(row["timesteps"]), model, checkpoint_path))
        rows.append(row)

    result_df = pd.DataFrame(rows).sort_values(["timesteps", "phase"]).reset_index(drop=True)
    default_df = result_df[result_df["phase"].astype(str).str.contains("default")].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    videos_dir = args.output_dir / "videos"
    plots_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(args.output_dir / "checkpoint_evidence.csv", index=False)
    write_dashboard(result_df, plots_dir)
    write_section_success(default_df, plots_dir)
    write_failure_histogram(default_df, plots_dir)

    default_config = default_training_config(with_obstacles=True, level_name="default")
    default_records.sort(key=lambda item: item[0])
    if default_records:
        early = default_records[0]
        mid = default_records[len(default_records) // 2]
        late = default_records[-1]
        late_frames, late_info = collect_rollout(default_config, seed=args.seed, policy=late[1])
        write_pendulum_trace(late_frames, plots_dir)
        (plots_dir / "pendulum_trace_summary.json").write_text(json.dumps({"termination_reason": late_info["termination_reason"]}, indent=2) + "\n")

        early_frames, _ = collect_rollout(default_config, seed=args.seed, policy=early[1])
        mid_frames, _ = collect_rollout(default_config, seed=args.seed, policy=mid[1])
        render_comparison_video(
            [("Early", early_frames, "#ef4444"), ("Mid", mid_frames, "#f59e0b"), ("Late", late_frames, "#2563eb")],
            config=default_config,
            output_path=videos_dir / "early_vs_mid_vs_late.mp4",
            title="Early vs Mid vs Late Default Checkpoints",
        )

        coast_frames, _ = collect_rollout(default_config, seed=args.seed, baseline="coast")
        pd_frames, _ = collect_rollout(default_config, seed=args.seed, baseline="center_pd")
        render_comparison_video(
            [("Coast", coast_frames, "#6b7280"), ("Center PD", pd_frames, "#16a34a"), ("Best Policy", late_frames, "#2563eb")],
            config=default_config,
            output_path=videos_dir / "baseline_vs_policy.mp4",
            title="Baseline vs Policy on Default Track",
        )

    summary = {
        "checkpoint_evidence_csv": str(args.output_dir / "checkpoint_evidence.csv"),
        "plots_dir": str(plots_dir),
        "videos_dir": str(videos_dir),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
