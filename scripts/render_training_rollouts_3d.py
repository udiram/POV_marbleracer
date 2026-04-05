from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from panda3d.core import (
    AmbientLight,
    DirectionalLight,
    LineSegs,
    NodePath,
    PointLight,
    TransparencyAttrib,
    Vec3,
    Vec4,
    loadPrcFileData,
)
from stable_baselines3 import PPO

from marbleracer.app import MarbleRampApp
from marbleracer.physics import ramp_normal, ramp_side, ramp_tangent
from marbleracer.progress import GhostSample
from marbleracer.rl_env import MarbleBotTrainingEnv, default_training_config


CHECKPOINT_COLORS = {
    "Early": "#ef4444",
    "Best": "#16a34a",
    "Latest": "#2563eb",
}


@dataclass(frozen=True, slots=True)
class RolloutFrame:
    time: float
    position: tuple[float, float, float]
    progress: float
    speed: float
    linear_velocity: tuple[float, float, float]
    steer: float
    brake: float
    termination_reason: str


@dataclass(frozen=True, slots=True)
class RolloutTrajectory:
    label: str
    color_hex: str
    checkpoint_path: Path
    max_progress: float
    reward: float
    termination_reason: str
    frames: tuple[RolloutFrame, ...]

    @property
    def duration(self) -> float:
        if not self.frames:
            return 0.0
        return float(self.frames[-1].time)


def parse_checkpoint_rows(log_path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    buffer: list[str] = []
    collecting = False
    for line in log_path.read_text().splitlines():
        stripped = line.strip()
        if stripped == "{":
            collecting = True
            buffer = ["{"]
            continue
        if collecting:
            buffer.append(line)
            if stripped == "}":
                collecting = False
                try:
                    row = json.loads("\n".join(buffer))
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and "checkpoint_path" in row:
                    rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No checkpoint metrics found in {log_path}")
    return pd.DataFrame(rows).sort_values(["timesteps", "phase"]).reset_index(drop=True)


def load_checkpoint_metrics(*, metrics_csv: Path | None, log_path: Path | None) -> pd.DataFrame:
    if metrics_csv is not None:
        if not metrics_csv.exists():
            raise FileNotFoundError(f"Missing metrics CSV: {metrics_csv}")
        return pd.read_csv(metrics_csv).sort_values(["timesteps", "phase"]).reset_index(drop=True)
    if log_path is None:
        raise ValueError("Either --metrics-csv or --log-path is required.")
    return parse_checkpoint_rows(log_path)


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


def collect_trajectory(*, checkpoint_path: Path, label: str, color_hex: str, seed: int) -> RolloutTrajectory:
    model = PPO.load(str(checkpoint_path))
    config = default_training_config(with_obstacles=True, level_name="default")
    env = MarbleBotTrainingEnv(config=config)
    observation, _ = env.reset(seed=seed)
    frames: list[RolloutFrame] = []
    info: dict[str, object] = {"termination_reason": "running"}
    for _ in range(4000):
        action = np.asarray(model.predict(observation, deterministic=True)[0], dtype=np.float32)
        observation, reward, terminated, truncated, info = env.step(action)
        snapshot = env.simulation.snapshot()
        frames.append(
            RolloutFrame(
                time=float(snapshot.time),
                position=(float(snapshot.position.x), float(snapshot.position.y), float(snapshot.position.z)),
                progress=float(info["path_distance"]),
                speed=float(snapshot.speed),
                linear_velocity=(
                    float(snapshot.linear_velocity.x),
                    float(snapshot.linear_velocity.y),
                    float(snapshot.linear_velocity.z),
                ),
                steer=float(action[0]),
                brake=float(action[1]),
                termination_reason=str(info["termination_reason"]),
            )
        )
        if terminated or truncated:
            break
    env.close()
    return RolloutTrajectory(
        label=label,
        color_hex=color_hex,
        checkpoint_path=checkpoint_path,
        max_progress=float(max(frame.progress for frame in frames)) if frames else 0.0,
        reward=float(info.get("episode_reward", 0.0)),
        termination_reason=str(info["termination_reason"]),
        frames=tuple(frames),
    )


def hex_to_rgba(color_hex: str, alpha: float) -> Vec4:
    color_hex = color_hex.lstrip("#")
    red = int(color_hex[0:2], 16) / 255.0
    green = int(color_hex[2:4], 16) / 255.0
    blue = int(color_hex[4:6], 16) / 255.0
    return Vec4(red, green, blue, alpha)


def frame_at_time(trajectory: RolloutTrajectory, target_time: float) -> RolloutFrame:
    if not trajectory.frames:
        raise ValueError(f"No frames in trajectory {trajectory.label}")
    if len(trajectory.frames) == 1 or target_time <= 0.0:
        return trajectory.frames[0]
    if target_time >= trajectory.frames[-1].time:
        return trajectory.frames[-1]
    times = [frame.time for frame in trajectory.frames]
    upper_index = int(np.searchsorted(times, target_time, side="left"))
    upper_index = min(max(1, upper_index), len(trajectory.frames) - 1)
    lower = trajectory.frames[upper_index - 1]
    upper = trajectory.frames[upper_index]
    span = max(1e-6, upper.time - lower.time)
    alpha = float(np.clip((target_time - lower.time) / span, 0.0, 1.0))

    def lerp(a: float, b: float) -> float:
        return a * (1.0 - alpha) + b * alpha

    lower_pos = np.asarray(lower.position, dtype=np.float32)
    upper_pos = np.asarray(upper.position, dtype=np.float32)
    lower_vel = np.asarray(lower.linear_velocity, dtype=np.float32)
    upper_vel = np.asarray(upper.linear_velocity, dtype=np.float32)
    position = tuple(float(value) for value in (lower_pos * (1.0 - alpha) + upper_pos * alpha))
    velocity = tuple(float(value) for value in (lower_vel * (1.0 - alpha) + upper_vel * alpha))
    return RolloutFrame(
        time=target_time,
        position=position,
        progress=lerp(lower.progress, upper.progress),
        speed=lerp(lower.speed, upper.speed),
        linear_velocity=velocity,
        steer=lerp(lower.steer, upper.steer),
        brake=lerp(lower.brake, upper.brake),
        termination_reason=upper.termination_reason if alpha >= 0.5 else lower.termination_reason,
    )


def to_ghost_samples(trajectory: RolloutTrajectory) -> tuple[GhostSample, ...]:
    return tuple(
        GhostSample(time=frame.time, path_distance=frame.progress, position=frame.position)
        for frame in trajectory.frames
    )


def add_trail(app: MarbleRampApp, samples: tuple[GhostSample, ...], *, color: Vec4, z_offset: float) -> NodePath:
    line_segs = LineSegs("trajectory-trail")
    line_segs.setColor(color)
    line_segs.setThickness(4.0)
    if not samples:
        return app.scene_root.attachNewNode("empty-trajectory-trail")
    first = Vec3(*samples[0].position) + Vec3(0.0, 0.0, z_offset)
    line_segs.moveTo(first)
    for sample in samples[1:]:
        point = Vec3(*sample.position) + Vec3(0.0, 0.0, z_offset)
        line_segs.drawTo(point)
    node = app.scene_root.attachNewNode(line_segs.create())
    node.setTransparency(TransparencyAttrib.MAlpha)
    node.setDepthWrite(False)
    node.setBin("transparent", 25)
    return node


def add_marker(app: MarbleRampApp, *, color: Vec4, scale: float) -> NodePath:
    marker_root = app.scene_root.attachNewNode("trajectory-marker")
    marker_root.setTransparency(TransparencyAttrib.MAlpha)
    marker_root.setDepthWrite(False)
    marker_root.setBin("transparent", 35)
    app._load_styled_model(
        marker_root,
        "sphere",
        scale=Vec3(scale),
        color=color,
    )
    return marker_root


def install_replay_lighting(app: MarbleRampApp) -> list[NodePath]:
    rig_root = app.render.attachNewNode("replay-light-rig")
    ambient = AmbientLight("replay-ambient")
    ambient.setColor(Vec4(0.62, 0.66, 0.74, 1.0))
    ambient_np = rig_root.attachNewNode(ambient)
    app.render.setLight(ambient_np)

    key = DirectionalLight("replay-key")
    key.setColor(Vec4(1.18, 1.10, 0.96, 1.0))
    key_np = rig_root.attachNewNode(key)
    key_np.setHpr(-42.0, -34.0, 0.0)
    app.render.setLight(key_np)

    fill = DirectionalLight("replay-fill")
    fill.setColor(Vec4(0.62, 0.76, 0.98, 1.0))
    fill_np = rig_root.attachNewNode(fill)
    fill_np.setHpr(128.0, -12.0, 0.0)
    app.render.setLight(fill_np)

    chase = PointLight("replay-chase")
    chase.setColor(Vec4(0.86, 0.90, 1.06, 1.0))
    chase_np = rig_root.attachNewNode(chase)
    app.render.setLight(chase_np)
    return [rig_root, chase_np]


def update_director_camera(
    app: MarbleRampApp,
    current: RolloutFrame,
    *,
    smoothing_state: dict[str, Vec3],
    dt: float,
) -> None:
    progress = current.progress
    position = Vec3(*current.position)
    tangent = ramp_tangent(app.sim_config, progress + 0.8)
    side = ramp_side(app.sim_config, progress)
    normal = ramp_normal(app.sim_config, progress)
    for vector, fallback in ((tangent, Vec3(1.0, 0.0, 0.0)), (side, Vec3(0.0, 1.0, 0.0)), (normal, Vec3(0.0, 0.0, 1.0))):
        if vector.length_squared() <= 1e-9:
            vector.set(fallback)
        vector.normalize()

    lateral_bias = float(np.clip(current.steer * 1.6, -1.4, 1.4))
    speed_bias = float(np.clip(current.speed * 0.10, 0.0, 1.2))
    look_target = position + tangent * (4.2 + speed_bias) + normal * 0.55
    camera_target = position - tangent * (8.6 + speed_bias * 1.2) + side * (2.6 + lateral_bias) + normal * 3.8

    previous_camera = smoothing_state.get("camera_pos", camera_target)
    previous_look = smoothing_state.get("look_target", look_target)
    blend = float(np.clip(1.0 - np.exp(-dt * 5.0), 0.0, 1.0))
    camera_pos = previous_camera * (1.0 - blend) + camera_target * blend
    camera_look = previous_look * (1.0 - blend) + look_target * blend
    smoothing_state["camera_pos"] = camera_pos
    smoothing_state["look_target"] = camera_look

    app.camera.setPos(camera_pos)
    app.camera.lookAt(camera_look)
    app.camera.setR(float(np.clip(-current.steer * min(7.0, current.speed * 1.8), -6.0, 6.0)))
    app.camLens.setFov(42.0 + min(8.0, current.speed * 0.65))


def overlay_legend(
    image_path: Path,
    *,
    trajectories: list[RolloutTrajectory],
    active_label: str,
    title: str,
    subtitle: str,
    metrics: str,
) -> None:
    image = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    panel_left = 26
    panel_top = 26
    panel_width = 360
    panel_height = 200
    draw.rounded_rectangle(
        [(panel_left, panel_top), (panel_left + panel_width, panel_top + panel_height)],
        radius=18,
        fill=(10, 16, 24, 186),
        outline=(86, 104, 128, 180),
        width=2,
    )
    y = panel_top + 16
    draw.text((panel_left + 16, y), title, fill=(236, 240, 245, 255), font=font)
    y += 20
    draw.text((panel_left + 16, y), subtitle, fill=(162, 177, 193, 255), font=font)
    y += 28
    for trajectory in trajectories:
        rgba = hex_to_rgba(trajectory.color_hex, 1.0)
        swatch = (
            int(round(rgba.x * 255.0)),
            int(round(rgba.y * 255.0)),
            int(round(rgba.z * 255.0)),
            255,
        )
        draw.rectangle(
            [(panel_left + 16, y + 3), (panel_left + 32, y + 19)],
            fill=swatch,
        )
        label = f"{trajectory.label} {'(active)' if trajectory.label == active_label else '(ghost)'}"
        draw.text((panel_left + 40, y), label, fill=(236, 240, 245, 255), font=font)
        y += 18
        draw.text(
            (panel_left + 40, y),
            f"{trajectory.max_progress:5.2f} m   {trajectory.termination_reason}",
            fill=(162, 177, 193, 255),
            font=font,
        )
        y += 28
    draw.text((panel_left + 16, y), "Metrics", fill=(236, 240, 245, 255), font=font)
    y += 20
    for line in metrics.splitlines():
        draw.text((panel_left + 16, y), line, fill=(162, 177, 193, 255), font=font)
        y += 16
    image.save(image_path)


def control_state(current: RolloutFrame) -> str:
    if current.brake >= 0.35:
        return "SLOW"
    if current.steer >= 0.18:
        return "LEFT"
    if current.steer <= -0.18:
        return "RIGHT"
    return "NONE"


def overlay_controls(image_path: Path, *, current: RolloutFrame) -> None:
    image = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size
    panel_width = 250
    panel_height = 98
    panel_left = width - panel_width - 26
    panel_top = height - panel_height - 26
    draw.rounded_rectangle(
        [(panel_left, panel_top), (panel_left + panel_width, panel_top + panel_height)],
        radius=18,
        fill=(10, 16, 24, 186),
        outline=(86, 104, 128, 180),
        width=2,
    )
    draw.text((panel_left + 16, panel_top + 14), "Live Controls", fill=(236, 240, 245, 255), font=font)
    active = control_state(current)
    states = (
        ("LEFT", active == "LEFT"),
        ("RIGHT", active == "RIGHT"),
        ("SLOW", active == "SLOW"),
        ("NONE", active == "NONE"),
    )
    x = panel_left + 16
    y = panel_top + 44
    for label, is_active in states:
        fill = (37, 99, 235, 230) if is_active else (34, 44, 58, 210)
        outline = (122, 184, 255, 255) if is_active else (86, 104, 128, 160)
        text = (245, 249, 255, 255) if is_active else (162, 177, 193, 255)
        box_width = 48 if label != "RIGHT" else 54
        draw.rounded_rectangle(
            [(x, y), (x + box_width, y + 28)],
            radius=10,
            fill=fill,
            outline=outline,
            width=2,
        )
        draw.text((x + 10, y + 8), label, fill=text, font=font)
        x += box_width + 8
    image.save(image_path)


def render_video(
    trajectories: list[RolloutTrajectory],
    *,
    active_label: str,
    output_path: Path,
    hero_frame_path: Path,
    width: int,
    height: int,
    fps: int,
    playback_speed: float,
) -> None:
    loadPrcFileData("", "window-type offscreen")
    loadPrcFileData("", "audio-library-name null")
    loadPrcFileData("", f"win-size {width} {height}")

    app = MarbleRampApp(
        None,
        initial_level="default",
        menu_disabled=True,
        start_mode="time_trial",
        bot_controller_mode="mixed",
    )
    app.taskMgr.remove("marble-ramp-tick")
    app.render2d.hide()
    app.a2dTopLeft.hide()
    app.a2dBottomLeft.hide()
    app.a2dTopRight.hide()
    app.a2dBottomRight.hide()
    app.event_timer = 0.0
    app.event_text = ""
    app.event_subtitle = ""
    app.flash_alpha = 0.0
    app.boost_flash = 0.0
    app.finish_flash = 0.0
    app.start_burst = 0.0
    app.setBackgroundColor(0.78, 0.86, 0.96, 1.0)
    app.render.clearFog()
    app.scene_root.setColorScale(1.18, 1.18, 1.20, 1.0)
    replay_light_nodes = install_replay_lighting(app)
    chase_light = replay_light_nodes[-1]

    active = next(trajectory for trajectory in trajectories if trajectory.label == active_label)
    ghosts = [trajectory for trajectory in trajectories if trajectory.label != active_label]

    trail_nodes: list[NodePath] = []
    ghost_markers: list[tuple[RolloutTrajectory, NodePath]] = []
    for index, trajectory in enumerate(ghosts):
        color = hex_to_rgba(trajectory.color_hex, 0.70)
        samples = to_ghost_samples(trajectory)
        trail_nodes.append(add_trail(app, samples, color=color, z_offset=0.12 + 0.02 * index))
        marker = add_marker(app, color=color, scale=app.sim_config.marble_radius * 0.86)
        ghost_markers.append((trajectory, marker))

    active_color = hex_to_rgba(active.color_hex, 1.0)
    app.marble_root.setColorScale(active_color.x, active_color.y, active_color.z, 1.0)
    active_trail = add_trail(app, to_ghost_samples(active), color=hex_to_rgba(active.color_hex, 0.28), z_offset=0.06)
    trail_nodes.append(active_trail)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    hero_frame_path.parent.mkdir(parents=True, exist_ok=True)
    effective_playback_speed = max(0.05, playback_speed)
    output_duration = max(10.0, active.duration / effective_playback_speed)
    frame_count = max(2, int(np.ceil(output_duration * fps)))
    with tempfile.TemporaryDirectory(prefix="marbleracer-3d-rollout-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        smoothing_state: dict[str, Vec3] = {}
        for frame_idx in range(frame_count):
            playback_time = min(active.duration, (frame_idx / max(1, frame_count - 1)) * output_duration * effective_playback_speed)
            current = frame_at_time(active, playback_time)
            position = Vec3(*current.position)
            app.marble_root.setPos(position)
            app.marble_root.setHpr(0.0, 0.0, 0.0)
            app._animate_moving_obstacles(current.time)
            for ghost_trajectory, marker in ghost_markers:
                ghost_frame = frame_at_time(ghost_trajectory, playback_time)
                marker.setPos(Vec3(*ghost_frame.position))
            update_director_camera(app, current, smoothing_state=smoothing_state, dt=1.0 / fps)
            chase_light.setPos(app.camera.getPos() + Vec3(0.0, 0.0, 2.0))
            for _ in range(2):
                app.graphicsEngine.renderFrame()
            frame_path = temp_dir / f"frame_{frame_idx:04d}.png"
            app.win.saveScreenshot(str(frame_path))
            overlay_legend(
                frame_path,
                trajectories=trajectories,
                active_label=active_label,
                title="3D Training Replay",
                subtitle="Real Panda3D scene, ghost trajectory comparison",
                metrics=(
                    f"Active: {active.label}\n"
                    f"Progress: {current.progress:5.2f} m\n"
                    f"Speed:    {current.speed:5.2f} m/s\n"
                    f"Steer:    {current.steer:+0.2f}\n"
                    f"Brake:    {current.brake:0.2f}\n"
                    f"Time:     {current.time:0.2f} s\n"
                    f"Playback: {effective_playback_speed:0.2f}x"
                ),
            )
            overlay_controls(frame_path, current=current)
            if frame_idx == frame_count - 1:
                Image.open(frame_path).save(hero_frame_path)

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(temp_dir / "frame_%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
    for node in trail_nodes:
        if not node.isEmpty():
            node.removeNode()
    for _, marker in ghost_markers:
        if not marker.isEmpty():
            marker.removeNode()
    for node in replay_light_nodes:
        if not node.isEmpty():
            node.removeNode()
    app.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a true 3D Panda3D replay of training checkpoints with ghost trajectories.")
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--metrics-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--playback-speed", type=float, default=0.7)
    args = parser.parse_args()

    metrics = load_checkpoint_metrics(metrics_csv=args.metrics_csv, log_path=args.log_path)
    default_rows = metrics.loc[metrics["phase"].astype(str).str.contains("phase2_default_full")].copy()
    if default_rows.empty:
        raise RuntimeError("No default-track checkpoints found in the log.")
    default_rows = default_rows.sort_values("timesteps").reset_index(drop=True)
    early_row = default_rows.iloc[0]
    best_row = default_rows.sort_values(["max_progress", "reward"], ascending=[False, False]).iloc[0]
    latest_row = default_rows.iloc[-1]

    trajectories = [
        collect_trajectory(
            checkpoint_path=Path(str(early_row["checkpoint_path"])),
            label="Early",
            color_hex=CHECKPOINT_COLORS["Early"],
            seed=args.seed,
        ),
        collect_trajectory(
            checkpoint_path=Path(str(best_row["checkpoint_path"])),
            label="Best",
            color_hex=CHECKPOINT_COLORS["Best"],
            seed=args.seed,
        ),
        collect_trajectory(
            checkpoint_path=Path(str(latest_row["checkpoint_path"])),
            label="Latest",
            color_hex=CHECKPOINT_COLORS["Latest"],
            seed=args.seed,
        ),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_video(
        trajectories,
        active_label="Latest",
        output_path=args.output_dir / "training_ghosts_latest_active.mp4",
        hero_frame_path=args.output_dir / "training_ghosts_latest_active.png",
        width=args.width,
        height=args.height,
        fps=args.fps,
        playback_speed=args.playback_speed,
    )

    summary = {
        "early_checkpoint": str(early_row["checkpoint_path"]),
        "best_checkpoint": str(best_row["checkpoint_path"]),
        "latest_checkpoint": str(latest_row["checkpoint_path"]),
        "video": str(args.output_dir / "training_ghosts_latest_active.mp4"),
        "hero_frame": str(args.output_dir / "training_ghosts_latest_active.png"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
