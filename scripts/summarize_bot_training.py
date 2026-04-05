from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from stable_baselines3 import PPO

from marbleracer.bot_controller import default_policy_manifest_path, export_sb3_policy
from marbleracer.rl_env import MarbleBotTrainingEnv, default_training_config, evaluate_policy_actions

CHECKPOINT_PATTERN = re.compile(r"(?P<phase>.+)_gen_(?P<generation>\d+)_(?P<timesteps>\d+)\.zip$")
DEFAULT_EVAL_SEEDS = (7, 13, 23)
DEFAULT_SECTION_MARKERS = (23.0, 57.5, 71.5, 79.5, 89.0, 101.0)


def sample_track_points(config, *, count: int = 300) -> tuple[np.ndarray, np.ndarray]:
    from marbleracer.physics import ramp_surface_point

    distances = np.linspace(0.0, config.length, count)
    points = np.array([ramp_surface_point(config, float(distance)) for distance in distances], dtype=np.float32)
    return points[:, 0], points[:, 1]


def write_plots(metrics: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = [
        ("reward", "Total Reward", "Checkpoint Reward", "reward_curve.png"),
        ("max_progress", "Max Progress", "Checkpoint Progress", "progress_curve.png"),
        ("finish_time", "Finish Time (s)", "Checkpoint Finish Time", "finish_time_curve.png"),
        ("resets", "Reset Count", "Checkpoint Resets", "reset_curve.png"),
    ]
    for column, ylabel, title, filename in plot_specs:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(metrics["timesteps"], metrics[column], marker="o", linewidth=2.0, color="#2563eb")
        ax.set_title(title)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    for axis, (column, ylabel, _, _) in zip(axes, plot_specs):
        axis.plot(metrics["timesteps"], metrics[column], marker="o", linewidth=2.0, color="#0f766e")
        axis.set_title(ylabel)
        axis.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_dir / "dashboard.png", dpi=180)
    fig.savefig(output_dir / "checkpoint_dashboard.png", dpi=180)
    plt.close(fig)


def level_config_for_phase(phase_name: str):
    if "time_trial_short" in phase_name:
        return default_training_config(with_obstacles=False, level_name="time-trial-short")
    return default_training_config(with_obstacles=True, level_name="default")


def rollout_action(name: str, observation: np.ndarray, info: dict[str, object]) -> np.ndarray:
    if name == "coast":
        return np.array([0.0, 0.0], dtype=np.float32)
    lane_error = float(observation[5]) * 0.5
    lateral_speed = float(observation[3]) * 6.0
    speed = float(observation[1]) * 12.0
    target_speed = float(observation[22]) * 12.0
    steer = np.clip(3.0 * lane_error - 1.2 * lateral_speed, -1.0, 1.0)
    brake = np.clip((speed - target_speed) * 0.08, 0.0, 1.0)
    return np.array([steer, brake], dtype=np.float32)


def collect_rollout(
    *,
    config,
    seed: int,
    action_provider: Callable[[np.ndarray, dict[str, object]], np.ndarray],
) -> tuple[list[dict[str, float | int | str]], dict[str, object]]:
    env = MarbleBotTrainingEnv(config=config)
    observation, _ = env.reset(seed=seed)
    frames: list[dict[str, float | int | str]] = []
    info: dict[str, object] = {}
    for _ in range(4000):
        action = action_provider(observation, info)
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
                "rail_contacts": int(snapshot.rail_contacts),
                "obstacle_contacts": int(sum(snapshot.obstacle_contacts)),
                "reward": float(info["episode_reward"]),
                "termination_reason": str(info["termination_reason"]),
            }
        )
        if terminated or truncated:
            break
    env.close()
    return frames, info


def render_rollout(model: PPO, phase_name: str, csv_path: Path, animation_path: Path) -> dict[str, object]:
    config = level_config_for_phase(phase_name)
    frames, info = collect_rollout(
        config=config,
        seed=7,
        action_provider=lambda observation, _: np.asarray(model.predict(observation, deterministic=True)[0], dtype=np.float32),
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(frames).to_csv(csv_path, index=False)

    track_x, track_y = sample_track_points(config)
    animation_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(track_x, track_y, color="#1f2937", linewidth=8, alpha=0.55, solid_capstyle="round")
    ax.plot(track_x, track_y, color="#cbd5e1", linewidth=1.5, alpha=0.85)
    for boost_pad in config.boost_pads:
        ax.scatter([boost_pad.distance_along_ramp], [boost_pad.lateral_offset], s=90, c="#06b6d4", alpha=0.7)
    for obstacle in config.obstacles:
        ax.scatter([obstacle.distance_along_ramp], [obstacle.lateral_offset], s=85, c="#f97316", alpha=0.6)
    marble_artist = ax.scatter([], [], s=120, c="#2563eb", edgecolors="white", linewidths=0.8, zorder=5)
    trail_artist, = ax.plot([], [], color="#2563eb", linewidth=1.5, alpha=0.6)
    title = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")
    ax.set_title(f"{phase_name} rollout")
    ax.set_xlabel("Track Distance")
    ax.set_ylabel("Lateral Offset")
    ax.grid(alpha=0.18)
    ax.set_xlim(-2.0, config.length + 2.0)
    ax.set_ylim(-config.width * 1.1, config.width * 1.1)

    def _update(index: int):
        frame = frames[index]
        marble_artist.set_offsets(np.array([[frame["x"], frame["y"]]], dtype=np.float32))
        trail = np.array([[sample["x"], sample["y"]] for sample in frames[: index + 1]], dtype=np.float32)
        trail_artist.set_data(trail[:, 0], trail[:, 1])
        title.set_text(
            f"t={frame['time']:5.2f}s progress={frame['progress']:6.2f} "
            f"speed={frame['speed']:4.2f} rails={frame['rail_contacts']} obs={frame['obstacle_contacts']}"
        )
        return marble_artist, trail_artist, title

    step = max(1, len(frames) // 220)
    sampled = frames[::step] if step > 1 else frames
    anim = animation.FuncAnimation(fig, lambda i: _update(i), frames=len(sampled), interval=40, blit=False)
    writer = animation.FFMpegWriter(fps=24, bitrate=1800)
    anim.save(animation_path, writer=writer)
    plt.close(fig)
    return info


def evaluate_checkpoint_seeds(model: PPO, *, config, phase_name: str, seeds: tuple[int, ...]) -> tuple[dict[str, object], list[dict[str, object]]]:
    section_markers = (*DEFAULT_SECTION_MARKERS, config.length - max(0.18, config.marble_radius * 0.8)) if "default" in phase_name else ()
    seed_rows: list[dict[str, object]] = []
    for seed in seeds:
        result = evaluate_policy_actions(model, config=config, seed=seed)
        row = {
            "phase": phase_name,
            "seed": seed,
            **asdict(result),
        }
        for marker in section_markers:
            row[f"section_reached_{marker:.1f}m"] = float(result.max_progress >= marker)
        seed_rows.append(row)
    seed_df = pd.DataFrame(seed_rows)
    metrics: dict[str, object] = {
        "finished": bool(seed_df["finished"].any()),
        "finish_rate": float(seed_df["finished"].mean()),
        "finish_time": float(seed_df.loc[seed_df["finished"], "finish_time"].median()) if seed_df["finished"].any() else None,
        "max_progress": float(seed_df["max_progress"].median()),
        "reward": float(seed_df["reward"].median()),
        "steps": int(seed_df["steps"].median()),
        "resets": float(seed_df["resets"].mean()),
        "termination_reason": str(seed_df.groupby("termination_reason").size().sort_values(ascending=False).index[0]),
    }
    for marker in section_markers:
        metrics[f"section_reached_{marker:.1f}m"] = float(seed_df[f"section_reached_{marker:.1f}m"].mean())
    return metrics, seed_rows


def write_section_success_curve(metrics: pd.DataFrame, output_dir: Path) -> None:
    default_metrics = metrics[metrics["phase"].str.contains("default")].copy()
    section_columns = [column for column in default_metrics.columns if column.startswith("section_reached_")]
    if default_metrics.empty or not section_columns:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for column in section_columns:
        label = column.replace("section_reached_", "").replace("m", " m")
        ax.plot(default_metrics["timesteps"], default_metrics[column], marker="o", linewidth=2.0, label=label)
    ax.set_title("Default Track Section Success")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "section_success_curve.png", dpi=180)
    plt.close(fig)


def write_failure_histogram(seed_metrics: pd.DataFrame, output_dir: Path) -> None:
    default_seed_metrics = seed_metrics[seed_metrics["phase"].str.contains("default")].copy()
    if default_seed_metrics.empty:
        return
    latest_timesteps = default_seed_metrics["timesteps"].max()
    latest = default_seed_metrics[default_seed_metrics["timesteps"] == latest_timesteps]
    counts = latest.groupby("termination_reason").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts.plot(kind="bar", ax=ax, color="#dc2626")
    ax.set_title("Latest Default Checkpoint Failure Histogram")
    ax.set_xlabel("Termination Reason")
    ax.set_ylabel("Count Across Seeds")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "failure_histogram.png", dpi=180)
    plt.close(fig)
    (output_dir / "latest_failure_counts.json").write_text(json.dumps(counts.to_dict(), indent=2) + "\n")


def write_pendulum_trace(model: PPO, output_dir: Path) -> None:
    config = default_training_config(with_obstacles=True, level_name="default")
    frames, info = collect_rollout(
        config=config,
        seed=7,
        action_provider=lambda observation, _: np.asarray(model.predict(observation, deterministic=True)[0], dtype=np.float32),
    )
    frame_df = pd.DataFrame(frames)
    pendulum = frame_df[(frame_df["progress"] >= 65.0) & (frame_df["progress"] <= 80.0)].copy()
    if pendulum.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(pendulum["progress"], pendulum["speed"], color="#2563eb", linewidth=2.0)
    axes[0].set_ylabel("Speed")
    axes[0].set_title("Pendulum Section Trace")
    axes[0].grid(alpha=0.2)
    axes[1].plot(pendulum["progress"], pendulum["signed_lateral_offset"], color="#ea580c", linewidth=2.0)
    axes[1].set_xlabel("Progress")
    axes[1].set_ylabel("Lateral Offset")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "pendulum_trace.png", dpi=180)
    plt.close(fig)
    pendulum.to_csv(output_dir / "pendulum_trace.csv", index=False)
    (output_dir / "pendulum_trace_summary.json").write_text(json.dumps({"termination_reason": info["termination_reason"]}, indent=2) + "\n")


def render_comparison_video(
    tracks: list[tuple[str, list[dict[str, float | int | str]], str]],
    *,
    config,
    output_path: Path,
    title: str,
    max_frames: int,
) -> None:
    track_x, track_y = sample_track_points(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(tracks), figsize=(5.4 * len(tracks), 4.8), sharex=True, sharey=True)
    if len(tracks) == 1:
        axes = [axes]
    artists: list[tuple[plt.Axes, object, object, object]] = []
    max_len = max(len(frames) for _, frames, _ in tracks)
    sampled_len = min(max_frames, max_len)
    indices = [np.linspace(0, max(0, len(frames) - 1), sampled_len, dtype=int) if frames else np.array([0]) for _, frames, _ in tracks]

    for axis, (label, frames, color) in zip(axes, tracks, strict=True):
        axis.plot(track_x, track_y, color="#1f2937", linewidth=8, alpha=0.55, solid_capstyle="round")
        axis.plot(track_x, track_y, color="#cbd5e1", linewidth=1.5, alpha=0.85)
        for obstacle in config.obstacles:
            axis.scatter([obstacle.distance_along_ramp], [obstacle.lateral_offset], s=60, c="#f97316", alpha=0.45)
        marble = axis.scatter([], [], s=100, c=color, edgecolors="white", linewidths=0.8, zorder=5)
        trail, = axis.plot([], [], color=color, linewidth=1.5, alpha=0.7)
        text = axis.text(0.02, 0.98, "", transform=axis.transAxes, va="top", ha="left")
        axis.set_title(label)
        axis.set_xlim(-2.0, config.length + 2.0)
        axis.set_ylim(-config.width * 1.1, config.width * 1.1)
        axis.grid(alpha=0.18)
        axis.set_xlabel("Track Distance")
        artists.append((axis, marble, trail, text))
    axes[0].set_ylabel("Lateral Offset")
    fig.suptitle(title)
    fig.tight_layout()

    def _update(frame_index: int):
        updated = []
        for idx, (_, frames, _color) in enumerate(tracks):
            sampled_index = int(indices[idx][min(frame_index, len(indices[idx]) - 1)])
            frame = frames[sampled_index]
            _axis, marble, trail, text = artists[idx]
            marble.set_offsets(np.array([[frame["x"], frame["y"]]], dtype=np.float32))
            trail_points = np.array([[sample["x"], sample["y"]] for sample in frames[: sampled_index + 1]], dtype=np.float32)
            trail.set_data(trail_points[:, 0], trail_points[:, 1])
            text.set_text(
                f"t={frame['time']:.2f}s\np={frame['progress']:.2f}\nv={frame['speed']:.2f}\n{frame['termination_reason']}"
            )
            updated.extend((marble, trail, text))
        return updated

    anim = animation.FuncAnimation(fig, _update, frames=sampled_len, interval=40, blit=False)
    anim.save(output_path, writer=animation.FFMpegWriter(fps=24, bitrate=2200))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize and render a checkpointed bot-training artifact directory.")
    parser.add_argument("--report-dir", type=Path, default=Path("artifacts/bot_training_report"))
    parser.add_argument("--export-manifest", type=Path, default=default_policy_manifest_path())
    parser.add_argument("--model-out", type=Path, default=Path("artifacts/default_bot_policy.zip"))
    parser.add_argument("--render-limit", type=int, default=4)
    parser.add_argument("--video-frames", type=int, default=140)
    parser.add_argument("--eval-seeds", type=int, nargs="*", default=list(DEFAULT_EVAL_SEEDS))
    args = parser.parse_args()

    checkpoint_paths = sorted(args.report_dir.glob("**/checkpoints/*.zip"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found under {args.report_dir}")

    rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    latest_checkpoint = checkpoint_paths[-1]
    latest_model = None
    default_checkpoint_records: list[tuple[int, Path, PPO]] = []
    for index, checkpoint_path in enumerate(checkpoint_paths):
        match = CHECKPOINT_PATTERN.match(checkpoint_path.name)
        if match is None:
            continue
        model = PPO.load(str(checkpoint_path))
        latest_model = model
        phase_name = match.group("phase")
        config = level_config_for_phase(phase_name)
        metrics, per_seed_rows = evaluate_checkpoint_seeds(
            model,
            config=config,
            phase_name=phase_name,
            seeds=tuple(int(seed) for seed in args.eval_seeds),
        )
        seed_rows.extend(
            [{**row, "timesteps": int(match.group("timesteps")), "generation": int(match.group("generation"))} for row in per_seed_rows]
        )
        phase_dir = checkpoint_path.parents[1]
        csv_path = phase_dir / "rollouts" / checkpoint_path.name.replace(".zip", ".csv")
        animation_path = phase_dir / "animations" / checkpoint_path.name.replace(".zip", ".mp4")
        if index < args.render_limit:
            render_rollout(model, phase_name, csv_path, animation_path)
        if "default" in phase_name:
            default_checkpoint_records.append((int(match.group("timesteps")), checkpoint_path, model))
        row = {
            "phase": phase_name,
            "generation": int(match.group("generation")),
            "timesteps": int(match.group("timesteps")),
            "checkpoint_path": str(checkpoint_path),
            "rollout_csv": str(csv_path),
            "animation_path": str(animation_path),
            **metrics,
        }
        rows.append(row)

    metrics_df = pd.DataFrame(rows).sort_values(["timesteps", "phase", "generation"]).reset_index(drop=True)
    seed_metrics_df = pd.DataFrame(seed_rows).sort_values(["timesteps", "phase", "seed"]).reset_index(drop=True)
    metrics_csv = args.report_dir / "checkpoint_metrics.csv"
    metrics_json = args.report_dir / "checkpoint_metrics.json"
    seed_metrics_csv = args.report_dir / "checkpoint_seed_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    metrics_df.to_json(metrics_json, orient="records", indent=2)
    seed_metrics_df.to_csv(seed_metrics_csv, index=False)
    write_plots(metrics_df, args.report_dir / "plots")
    write_section_success_curve(metrics_df, args.report_dir / "plots")
    write_failure_histogram(seed_metrics_df, args.report_dir / "plots")

    if latest_model is None:
        raise RuntimeError("No checkpoint models could be loaded.")
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    latest_model.save(str(args.model_out))
    export_path = export_sb3_policy(latest_model.policy, args.export_manifest)

    if default_checkpoint_records:
        default_checkpoint_records.sort(key=lambda item: item[0])
        best_default = max(default_checkpoint_records, key=lambda item: item[0])
        write_pendulum_trace(best_default[2], args.report_dir / "plots")

        selected = [default_checkpoint_records[0], default_checkpoint_records[len(default_checkpoint_records) // 2], default_checkpoint_records[-1]]
        comparison_tracks: list[tuple[str, list[dict[str, float | int | str]], str]] = []
        default_config = default_training_config(with_obstacles=True, level_name="default")
        colors = ("#ef4444", "#f59e0b", "#2563eb")
        labels = ("Early", "Mid", "Late")
        for (label, color), (_timesteps, _path, model) in zip(zip(labels, colors, strict=True), selected, strict=True):
            frames, _ = collect_rollout(
                config=default_config,
                seed=7,
                action_provider=lambda observation, _, model=model: np.asarray(
                    model.predict(observation, deterministic=True)[0], dtype=np.float32
                ),
            )
            comparison_tracks.append((label, frames, color))
        render_comparison_video(
            comparison_tracks,
            config=default_config,
            output_path=args.report_dir / "videos" / "early_vs_mid_vs_late.mp4",
            title="Early vs Mid vs Late Default Checkpoints",
            max_frames=args.video_frames,
        )

        baseline_tracks: list[tuple[str, list[dict[str, float | int | str]], str]] = []
        for label, color in (("Coast", "#6b7280"), ("Center PD", "#16a34a")):
            frames, _ = collect_rollout(
                config=default_config,
                seed=7,
                action_provider=lambda observation, info, label=label: rollout_action(
                    "coast" if label == "Coast" else "center_pd", observation, info
                ),
            )
            baseline_tracks.append((label, frames, color))
        best_frames, _ = collect_rollout(
            config=default_config,
            seed=7,
            action_provider=lambda observation, _: np.asarray(best_default[2].predict(observation, deterministic=True)[0], dtype=np.float32),
        )
        baseline_tracks.append(("Best Policy", best_frames, "#2563eb"))
        render_comparison_video(
            baseline_tracks,
            config=default_config,
            output_path=args.report_dir / "videos" / "baseline_vs_policy.mp4",
            title="Baseline vs Policy on Default Track",
            max_frames=args.video_frames,
        )

    summary = {
        "checkpoint_count": len(metrics_df),
        "latest_checkpoint": str(latest_checkpoint),
        "final_model": str(args.model_out),
        "export_manifest": str(export_path),
        "metrics_csv": str(metrics_csv),
        "metrics_json": str(metrics_json),
        "seed_metrics_csv": str(seed_metrics_csv),
        "plots_dir": str(args.report_dir / "plots"),
    }
    (args.report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
