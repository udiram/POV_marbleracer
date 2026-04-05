from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gymnasium import Wrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from marbleracer.bot_controller import default_policy_manifest_path, export_sb3_policy
from marbleracer.rl_env import MarbleBotTrainingEnv, default_training_config, evaluate_policy_actions


DEFAULT_PHASE2_CURRICULUM_SPAWNS = (0.0, 18.0, 36.0, 54.0, 68.0, 82.0, 96.0, 110.0)
CHECKPOINT_PATTERN = re.compile(r"(?P<phase>.+)_gen_(?P<generation>\d+)_(?P<timesteps>\d+)\.zip$")


class LiveStatusCallback(BaseCallback):
    def __init__(
        self,
        *,
        report_dir: Path,
        phase_name: str,
        current_generation: int,
        latest_saved_timesteps: int,
        checkpoint_interval: int,
        start_time: float,
    ) -> None:
        super().__init__(verbose=0)
        self.report_dir = report_dir
        self.phase_name = phase_name
        self.current_generation = current_generation
        self.latest_saved_timesteps = latest_saved_timesteps
        self.checkpoint_interval = checkpoint_interval
        self.start_time = start_time

    def _on_step(self) -> bool:
        elapsed = max(1e-6, time.time() - self.start_time)
        current_timesteps = int(self.model.num_timesteps)
        fps = (current_timesteps - self.latest_saved_timesteps) / elapsed
        payload = {
            "phase": self.phase_name,
            "in_progress_generation": self.current_generation,
            "latest_saved_timesteps": self.latest_saved_timesteps,
            "current_timesteps": current_timesteps,
            "checkpoint_interval": self.checkpoint_interval,
            "current_fps": fps,
            "updated_at": time.time(),
        }
        (self.report_dir / "live_status.json").write_text(json.dumps(payload, indent=2) + "\n")
        return True


class SpawnCurriculumWrapper(Wrapper):
    def __init__(self, env: MarbleBotTrainingEnv, spawn_distances: tuple[float, ...]) -> None:
        super().__init__(env)
        self._spawn_distances = tuple(float(distance) for distance in spawn_distances)
        self._rng = np.random.default_rng()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        options = dict(options or {})
        if "spawn_distance" not in options and self._spawn_distances:
            options["spawn_distance"] = float(self._rng.choice(self._spawn_distances))
        return self.env.reset(seed=seed, options=options)


def build_env(level_name: str, with_obstacles: bool, *, spawn_distances: tuple[float, ...] = ()):
    def _factory():
        env = MarbleBotTrainingEnv(config=default_training_config(with_obstacles=with_obstacles, level_name=level_name))
        if spawn_distances:
            return SpawnCurriculumWrapper(env, spawn_distances)
        return env

    return _factory


def build_vec_env(*, workers: int, level_name: str, with_obstacles: bool, spawn_distances: tuple[float, ...] = ()):
    factories = [build_env(level_name, with_obstacles, spawn_distances=spawn_distances) for _ in range(max(1, workers))]
    if len(factories) == 1:
        return DummyVecEnv(factories)
    return SubprocVecEnv(factories, start_method="spawn")


def training_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def evaluate_checkpoint(model: PPO, *, level_name: str, with_obstacles: bool) -> dict[str, object]:
    result = evaluate_policy_actions(
        model,
        config=default_training_config(with_obstacles=with_obstacles, level_name=level_name),
        max_steps=4000,
        seed=7,
    )
    return {
        "finished": result.finished,
        "finish_time": result.finish_time,
        "max_progress": result.max_progress,
        "reward": result.reward,
        "steps": result.steps,
        "resets": result.resets,
        "termination_reason": result.termination_reason,
    }


def run_phase_learning(
    model: PPO,
    *,
    phase_name: str,
    total_timesteps: int,
    checkpoint_interval: int,
    artifact_root: Path,
    metrics_rows: list[dict[str, object]],
    level_name: str,
    with_obstacles: bool,
    start_generation: int = 0,
    snapshot_dir: Path | None = None,
    snapshot_width: int = 1280,
    snapshot_height: int = 720,
    snapshot_fps: int = 24,
    snapshot_playback_speed: float = 0.65,
) -> None:
    phase_root = artifact_root / phase_name
    checkpoints_dir = phase_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    completed = 0
    generation = start_generation
    while completed < total_timesteps:
        chunk = min(checkpoint_interval, total_timesteps - completed)
        current_generation = generation + 1
        latest_saved_timesteps = int(model.num_timesteps)
        (artifact_root / "live_status.json").write_text(
            json.dumps(
                {
                    "phase": phase_name,
                    "in_progress_generation": current_generation,
                    "latest_saved_timesteps": latest_saved_timesteps,
                    "current_timesteps": latest_saved_timesteps,
                    "checkpoint_interval": checkpoint_interval,
                    "current_fps": None,
                    "updated_at": time.time(),
                },
                indent=2,
            )
            + "\n"
        )
        callback = LiveStatusCallback(
            report_dir=artifact_root,
            phase_name=phase_name,
            current_generation=current_generation,
            latest_saved_timesteps=latest_saved_timesteps,
            checkpoint_interval=checkpoint_interval,
            start_time=time.time(),
        )
        model.learn(total_timesteps=chunk, progress_bar=False, reset_num_timesteps=False, callback=callback)
        completed += chunk
        generation += 1
        timestep_total = int(model.num_timesteps)
        checkpoint_path = checkpoints_dir / f"{phase_name}_gen_{generation:03d}_{timestep_total:07d}.zip"
        model.save(str(checkpoint_path))
        metrics = evaluate_checkpoint(model, level_name=level_name, with_obstacles=with_obstacles)
        row = {
            "phase": phase_name,
            "generation": generation,
            "timesteps": timestep_total,
            "checkpoint_path": str(checkpoint_path),
            **metrics,
        }
        metrics_rows.append(row)
        save_metrics(artifact_root, metrics_rows)
        if with_obstacles:
            update_snapshot_artifacts(
                report_dir=artifact_root,
                snapshot_dir=snapshot_dir,
                snapshot_width=snapshot_width,
                snapshot_height=snapshot_height,
                snapshot_fps=snapshot_fps,
                snapshot_playback_speed=snapshot_playback_speed,
            )
        (artifact_root / "live_status.json").write_text(
            json.dumps(
                {
                    "phase": phase_name,
                    "in_progress_generation": generation + 1 if completed < total_timesteps else None,
                    "latest_saved_timesteps": timestep_total,
                    "current_timesteps": timestep_total,
                    "checkpoint_interval": checkpoint_interval,
                    "current_fps": None,
                    "updated_at": time.time(),
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(row, indent=2))


def load_existing_metrics(report_dir: Path) -> list[dict[str, object]]:
    metrics_csv = report_dir / "checkpoint_metrics.csv"
    if not metrics_csv.exists():
        return []
    frame = pd.read_csv(metrics_csv)
    return frame.to_dict(orient="records")


def checkpoint_metadata(checkpoint_path: Path) -> tuple[str, int, int]:
    match = CHECKPOINT_PATTERN.match(checkpoint_path.name)
    if match is None:
        raise ValueError(f"Unrecognized checkpoint filename: {checkpoint_path.name}")
    return match.group("phase"), int(match.group("generation")), int(match.group("timesteps"))


def save_metrics(report_dir: Path, metrics_rows: list[dict[str, object]]) -> tuple[Path, Path]:
    metrics_df = pd.DataFrame(metrics_rows).sort_values(["timesteps", "phase"]).reset_index(drop=True)
    metrics_csv = report_dir / "checkpoint_metrics.csv"
    metrics_json = report_dir / "checkpoint_metrics.json"
    metrics_df.to_csv(metrics_csv, index=False)
    metrics_df.to_json(metrics_json, orient="records", indent=2)
    return metrics_csv, metrics_json


def update_snapshot_artifacts(
    *,
    report_dir: Path,
    snapshot_dir: Path | None,
    snapshot_width: int,
    snapshot_height: int,
    snapshot_fps: int,
    snapshot_playback_speed: float,
) -> None:
    if snapshot_dir is None:
        return
    subprocess.run(
        [
            sys.executable,
            "scripts/update_training_snapshot.py",
            "--report-dir",
            str(report_dir),
            "--output-dir",
            str(snapshot_dir),
            "--python-bin",
            sys.executable,
            "--width",
            str(snapshot_width),
            "--height",
            str(snapshot_height),
            "--fps",
            str(snapshot_fps),
            "--playback-speed",
            str(snapshot_playback_speed),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the direct-control PPO policy for Apple Silicon-friendly runs.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--phase1-steps", type=int, default=40_000)
    parser.add_argument("--phase2-steps", type=int, default=120_000)
    parser.add_argument("--checkpoint-interval", type=int, default=10_000)
    parser.add_argument("--model-out", type=Path, default=Path("artifacts/default_bot_policy.zip"))
    parser.add_argument("--report-dir", type=Path, default=Path("artifacts/bot_training_report"))
    parser.add_argument("--export-manifest", type=Path, default=default_policy_manifest_path())
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--resume-phase", choices=("phase1", "phase2"), default=None)
    parser.add_argument("--snapshot-dir", type=Path, default=None)
    parser.add_argument("--snapshot-width", type=int, default=1280)
    parser.add_argument("--snapshot-height", type=int, default=720)
    parser.add_argument("--snapshot-fps", type=int, default=24)
    parser.add_argument("--snapshot-playback-speed", type=float, default=0.65)
    args = parser.parse_args()

    device = training_device()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    policy_kwargs = {"net_arch": {"pi": [64, 64], "vf": [64, 64]}}
    metrics_rows: list[dict[str, object]] = load_existing_metrics(args.report_dir)

    if args.resume_from is not None:
        if args.resume_phase is None:
            raise ValueError("--resume-phase is required when --resume-from is provided.")
        if args.resume_phase == "phase1":
            phase1_env = build_vec_env(workers=args.workers, level_name="time-trial-short", with_obstacles=False)
            model = PPO.load(str(args.resume_from), env=phase1_env, device=device)
            _, start_generation, _ = checkpoint_metadata(args.resume_from)
            run_phase_learning(
                model,
                phase_name="phase1_time_trial_short",
                total_timesteps=args.phase1_steps,
                checkpoint_interval=args.checkpoint_interval,
                artifact_root=args.report_dir,
                metrics_rows=metrics_rows,
                level_name="time-trial-short",
                with_obstacles=False,
                start_generation=start_generation,
                snapshot_dir=args.snapshot_dir,
                snapshot_width=args.snapshot_width,
                snapshot_height=args.snapshot_height,
                snapshot_fps=args.snapshot_fps,
                snapshot_playback_speed=args.snapshot_playback_speed,
            )
            phase1_env.close()
            phase2_env = build_vec_env(
                workers=args.workers,
                level_name="default",
                with_obstacles=True,
                spawn_distances=DEFAULT_PHASE2_CURRICULUM_SPAWNS,
            )
            model.set_env(phase2_env)
            run_phase_learning(
                model,
                phase_name="phase2_default_full",
                total_timesteps=args.phase2_steps,
                checkpoint_interval=args.checkpoint_interval,
                artifact_root=args.report_dir,
                metrics_rows=metrics_rows,
                level_name="default",
                with_obstacles=True,
                snapshot_dir=args.snapshot_dir,
                snapshot_width=args.snapshot_width,
                snapshot_height=args.snapshot_height,
                snapshot_fps=args.snapshot_fps,
                snapshot_playback_speed=args.snapshot_playback_speed,
            )
            phase2_env.close()
        else:
            phase2_env = build_vec_env(
                workers=args.workers,
                level_name="default",
                with_obstacles=True,
                spawn_distances=DEFAULT_PHASE2_CURRICULUM_SPAWNS,
            )
            model = PPO.load(str(args.resume_from), env=phase2_env, device=device)
            _, start_generation, _ = checkpoint_metadata(args.resume_from)
            run_phase_learning(
                model,
                phase_name="phase2_default_full",
                total_timesteps=args.phase2_steps,
                checkpoint_interval=args.checkpoint_interval,
                artifact_root=args.report_dir,
                metrics_rows=metrics_rows,
                level_name="default",
                with_obstacles=True,
                start_generation=start_generation,
                snapshot_dir=args.snapshot_dir,
                snapshot_width=args.snapshot_width,
                snapshot_height=args.snapshot_height,
                snapshot_fps=args.snapshot_fps,
                snapshot_playback_speed=args.snapshot_playback_speed,
            )
            phase2_env.close()
    else:
        phase1_env = build_vec_env(workers=args.workers, level_name="time-trial-short", with_obstacles=False)
        model = PPO(
            "MlpPolicy",
            phase1_env,
            seed=args.seed,
            verbose=1,
            policy_kwargs=policy_kwargs,
            n_steps=1024,
            batch_size=256,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.01,
            learning_rate=3e-4,
            clip_range=0.2,
            device=device,
        )

        run_phase_learning(
            model,
            phase_name="phase1_time_trial_short",
            total_timesteps=args.phase1_steps,
            checkpoint_interval=args.checkpoint_interval,
            artifact_root=args.report_dir,
            metrics_rows=metrics_rows,
            level_name="time-trial-short",
            with_obstacles=False,
            snapshot_dir=args.snapshot_dir,
            snapshot_width=args.snapshot_width,
            snapshot_height=args.snapshot_height,
            snapshot_fps=args.snapshot_fps,
            snapshot_playback_speed=args.snapshot_playback_speed,
        )
        phase1_env.close()

        phase2_env = build_vec_env(
            workers=args.workers,
            level_name="default",
            with_obstacles=True,
            spawn_distances=DEFAULT_PHASE2_CURRICULUM_SPAWNS,
        )
        model.set_env(phase2_env)
        run_phase_learning(
            model,
            phase_name="phase2_default_full",
            total_timesteps=args.phase2_steps,
            checkpoint_interval=args.checkpoint_interval,
            artifact_root=args.report_dir,
            metrics_rows=metrics_rows,
            level_name="default",
            with_obstacles=True,
            snapshot_dir=args.snapshot_dir,
            snapshot_width=args.snapshot_width,
            snapshot_height=args.snapshot_height,
            snapshot_fps=args.snapshot_fps,
            snapshot_playback_speed=args.snapshot_playback_speed,
        )
        phase2_env.close()

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.model_out))
    export_path = export_sb3_policy(model.policy, args.export_manifest)

    metrics_csv, metrics_json = save_metrics(args.report_dir, metrics_rows)

    summary = {
        "seed": args.seed,
        "workers": args.workers,
        "device": device,
        "phase1_steps": args.phase1_steps,
        "phase2_steps": args.phase2_steps,
        "checkpoint_interval": args.checkpoint_interval,
        "final_model": str(args.model_out),
        "export_manifest": str(export_path),
        "metrics_csv": str(metrics_csv),
        "metrics_json": str(metrics_json),
        "snapshot_dir": None if args.snapshot_dir is None else str(args.snapshot_dir),
    }
    (args.report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
