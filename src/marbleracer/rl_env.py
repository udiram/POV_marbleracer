from __future__ import annotations

from dataclasses import dataclass, replace
from math import exp

import gymnasium as gym
import numpy as np
from panda3d.core import Vec3

from .bot_controller import BotObservation, BotSensorFrame, build_bot_observation, monotonic_progress
from .levels import level_to_config, load_level
from .physics import (
    MarbleRampSimulation,
    SimulationConfig,
    recommended_evaluation_time,
    ramp_side,
    ramp_surface_point,
)
from .race_rules import finish_line_distance, should_finish_run, track_relative_offsets

DEFAULT_START_INPUT_LOCK = 0.22
DEFAULT_LAUNCH_ASSIST_DURATION = 0.42
DEFAULT_LAUNCH_SPEED = 3.20
DEFAULT_LAUNCH_STRENGTH = 5.2
DEFAULT_LAUNCH_BIAS = 0.80
DEFAULT_OFF_TRACK_GRACE = 0.18
DEFAULT_BOT_LANE_OFFSET = 0.0
DEFAULT_BOT_SPEED_BIAS = 0.0


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    finished: bool
    finish_time: float | None
    max_progress: float
    resets: int
    reward: float
    steps: int
    termination_reason: str


def default_training_config(*, with_obstacles: bool = True, level_name: str = "default") -> SimulationConfig:
    config = level_to_config(load_level(level_name))
    if with_obstacles:
        return config
    return replace(config, obstacles=(), obstacle_count=0)


def _signed_lateral_offset(config: SimulationConfig, position: Vec3, progress: float) -> float:
    surface = ramp_surface_point(config, progress)
    relative = position - surface
    return float(relative.dot(ramp_side(config, progress)))


def _stable_progress(config: SimulationConfig, snapshot, previous_progress: float, dt: float, action_repeat: int) -> float:
    forward_progress = max(0.0, min(config.length, float(snapshot.position.x)))
    raw_path_distance = float(snapshot.path_distance)
    raw_progress = max(previous_progress, forward_progress)
    max_reasonable_advance = max(0.8, float(snapshot.speed) * dt * action_repeat * 3.0)
    if (
        raw_path_distance >= previous_progress
        and raw_path_distance <= previous_progress + max_reasonable_advance
        and abs(raw_path_distance - forward_progress) <= 0.15
    ):
        raw_progress = max(raw_progress, raw_path_distance)
    return max(previous_progress, min(config.length, raw_progress))


def _reward_terms_template() -> dict[str, float]:
    return {
        "progress": 0.0,
        "time": 0.0,
        "rail_contact": 0.0,
        "obstacle_contact": 0.0,
        "impact": 0.0,
        "airborne": 0.0,
        "vertical_instability": 0.0,
        "dynamic_obstacle_hazard": 0.0,
        "stall": 0.0,
        "off_track": 0.0,
        "finish": 0.0,
    }


def _is_off_track_candidate(config: SimulationConfig, snapshot, progress: float) -> bool:
    lateral_offset, normal_offset, z_drop = track_relative_offsets(
        config,
        snapshot.position,
        progress,
    )
    surface = ramp_surface_point(config, progress)
    below_surface = snapshot.position.z < surface.z - 0.85 or normal_offset < -0.70
    far_from_lane = lateral_offset > config.width * 0.5 + 0.25
    falling_fast = snapshot.linear_velocity.z < -7.0
    unsupported = snapshot.total_contacts == 0
    if snapshot.position.z < -2.0:
        return True
    return unsupported and below_surface and falling_fast and (
        far_from_lane or snapshot.position.z < surface.z - 1.05
    )


class MarbleBotTrainingEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: SimulationConfig | None = None,
        *,
        dt: float = 1.0 / 240.0,
        action_repeat: int = 4,
        max_time: float | None = None,
        preferred_lane_offset: float = DEFAULT_BOT_LANE_OFFSET,
        speed_bias: float = DEFAULT_BOT_SPEED_BIAS,
    ) -> None:
        self.config = config or default_training_config(with_obstacles=True)
        self.dt = dt
        self.action_repeat = max(1, action_repeat)
        self.default_max_time = recommended_evaluation_time(self.config) + 6.0 if max_time is None else max_time
        self.max_time = self.default_max_time
        self.preferred_lane_offset = preferred_lane_offset
        self.speed_bias = speed_bias
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=np.full((23,), -5.0, dtype=np.float32),
            high=np.full((23,), 5.0, dtype=np.float32),
            dtype=np.float32,
        )
        self.simulation = MarbleRampSimulation(self.config)
        self.finish_distance = finish_line_distance(self.config)
        self._launch_assist_timer = 0.0
        self._start_input_lock_timer = 0.0
        self._stall_timer = 0.0
        self._off_track_timer = 0.0
        self._resets = 0
        self._episode_reward = 0.0
        self._steps = 0
        self._progress = 0.0
        self._termination_reason = "running"

    def _current_observation(self, snapshot) -> BotObservation:
        frame = BotSensorFrame(
            position=snapshot.position,
            linear_velocity=snapshot.linear_velocity,
            progress=self._progress,
            time_s=snapshot.time,
            airborne=snapshot.airborne,
            total_contacts=snapshot.total_contacts,
            active_boost_pad_index=snapshot.boost_pad_index,
            preferred_lane_offset=self.preferred_lane_offset,
            speed_bias=self.speed_bias,
        )
        return build_bot_observation(self.config, frame)

    def _prepare_start(self, *, disable_launch_assist: bool) -> None:
        self.simulation.settle_start_contact()
        self.simulation.marble_body.setLinearVelocity(Vec3(0.0, 0.0, 0.0))
        self.simulation.marble_body.setAngularVelocity(Vec3(0.0, 0.0, 0.0))
        self.simulation.marble_body.clearForces()
        self.simulation.set_steering_input(0.0)
        self.simulation.set_brake_input(0.0)
        if disable_launch_assist:
            self._launch_assist_timer = 0.0
            self._start_input_lock_timer = 0.0
        else:
            self.simulation.add_forward_speed(DEFAULT_LAUNCH_SPEED)
            self._launch_assist_timer = DEFAULT_LAUNCH_ASSIST_DURATION
            self._start_input_lock_timer = DEFAULT_START_INPUT_LOCK

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        spawn_distance = float(options.get("spawn_distance", 0.0))
        disable_launch_assist = bool(options.get("disable_launch_assist", False))
        self.max_time = float(options.get("max_time", self.default_max_time))
        self.simulation = MarbleRampSimulation(self.config)
        spawn_distance = max(0.0, min(self.config.length - 1.0, spawn_distance))
        self.simulation.respawn_marble(distance=spawn_distance)
        self._prepare_start(disable_launch_assist=disable_launch_assist)
        self._stall_timer = 0.0
        self._off_track_timer = 0.0
        self._resets = 0
        self._episode_reward = 0.0
        self._steps = 0
        self._termination_reason = "running"
        snapshot = self.simulation.snapshot()
        self._progress = max(0.0, min(self.config.length, spawn_distance))
        observation = self._current_observation(snapshot).as_array()
        info = {
            "termination_reason": self._termination_reason,
            "reward_terms": _reward_terms_template(),
            "speed": float(snapshot.speed),
            "path_distance": float(self._progress),
            "signed_lateral_offset": _signed_lateral_offset(self.config, snapshot.position, self._progress),
            "rail_contacts": int(snapshot.rail_contacts),
            "obstacle_contacts": int(sum(snapshot.obstacle_contacts)),
            "boost_active": bool(snapshot.boost_pad_index is not None),
            "finished": False,
            "resets": self._resets,
            "episode_reward": self._episode_reward,
        }
        return observation, info

    def _apply_action(self, action: np.ndarray) -> tuple[float, float]:
        steer = float(np.clip(action[0], -1.0, 1.0))
        brake = float(np.clip(action[1], 0.0, 1.0))
        if self._start_input_lock_timer > 0.0:
            steer = 0.0
            brake = 0.0
        self.simulation.set_steering_input(steer)
        self.simulation.set_brake_input(brake)
        return steer, brake

    def _update_launch_assist(self) -> None:
        if self._launch_assist_timer <= 0.0:
            return
        normalized = self._launch_assist_timer / max(DEFAULT_LAUNCH_ASSIST_DURATION, 1e-6)
        assist_strength = DEFAULT_LAUNCH_STRENGTH * normalized * normalized + DEFAULT_LAUNCH_BIAS
        self.simulation.add_forward_speed(assist_strength * self.dt)
        self._launch_assist_timer = max(0.0, self._launch_assist_timer - self.dt)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        total_reward = 0.0
        aggregated_terms = _reward_terms_template()
        terminated = False
        truncated = False

        for _ in range(self.action_repeat):
            previous = self.simulation.snapshot()
            previous_progress = self._progress
            self._apply_action(action)
            self._update_launch_assist()
            current = self.simulation.step(self.dt)
            self._steps += 1
            self._progress = _stable_progress(self.config, current, self._progress, self.dt, self.action_repeat)
            self._start_input_lock_timer = max(0.0, self._start_input_lock_timer - self.dt)
            current_observation = self._current_observation(current)

            progress_delta = max(0.0, self._progress - previous_progress)
            reward_terms = _reward_terms_template()
            reward_terms["progress"] = 2.0 * progress_delta
            reward_terms["time"] = -0.01
            reward_terms["rail_contact"] = -0.02 * float(current.rail_contacts)
            reward_terms["obstacle_contact"] = -0.08 * float(sum(current.obstacle_contacts))
            reward_terms["impact"] = -0.08 * float(current.impact_severity)
            reward_terms["airborne"] = -0.03 if current.airborne else 0.0
            reward_terms["vertical_instability"] = -0.02 * max(0.0, abs(current_observation.vertical_speed) - 0.75)

            if (
                current_observation.next_obstacle_dynamic > 0.5
                and 0.0 <= current_observation.next_obstacle_distance <= 6.0
                and abs(current_observation.next_obstacle_current_lateral - _signed_lateral_offset(self.config, current.position, self._progress))
                <= current_observation.next_obstacle_width * 0.75 + self.config.marble_radius * 1.4
            ):
                hazard_scale = (6.0 - current_observation.next_obstacle_distance) / 6.0
                reward_terms["dynamic_obstacle_hazard"] = -0.03 * current.speed * hazard_scale

            if current.speed < 0.15 and self._progress < self.finish_distance:
                self._stall_timer += self.dt
            else:
                self._stall_timer = 0.0

            off_track_candidate = _is_off_track_candidate(self.config, current, self._progress)
            if off_track_candidate:
                self._off_track_timer += self.dt
            else:
                self._off_track_timer = max(0.0, self._off_track_timer - self.dt * 2.0)

            if should_finish_run(
                self.config,
                previous.position,
                previous_progress,
                current.position,
                self._progress,
                target_distance=self.finish_distance,
            ):
                reward_terms["finish"] = 100.0
                self._termination_reason = "finish"
                terminated = True
            elif self._off_track_timer >= DEFAULT_OFF_TRACK_GRACE:
                reward_terms["off_track"] = -40.0
                self._termination_reason = "off_track"
                self._resets += 1
                terminated = True
            elif self._stall_timer >= 1.0:
                reward_terms["stall"] = -20.0
                self._termination_reason = "stall"
                terminated = True
            elif current.time >= self.max_time:
                self._termination_reason = "timeout"
                truncated = True

            reward = sum(reward_terms.values())
            total_reward += reward
            for key, value in reward_terms.items():
                aggregated_terms[key] += value

            if terminated or truncated:
                break

        self._episode_reward += total_reward
        snapshot = self.simulation.snapshot()
        observation = self._current_observation(snapshot).as_array()
        info = {
            "termination_reason": self._termination_reason,
            "reward_terms": aggregated_terms,
            "speed": float(snapshot.speed),
            "path_distance": float(self._progress),
            "progress": float(self._progress),
            "signed_lateral_offset": _signed_lateral_offset(self.config, snapshot.position, self._progress),
            "rail_contacts": int(snapshot.rail_contacts),
            "obstacle_contacts": int(sum(snapshot.obstacle_contacts)),
            "boost_active": bool(snapshot.boost_pad_index is not None),
            "finished": self._termination_reason == "finish",
            "off_track": self._termination_reason == "off_track",
            "stalled": self._termination_reason == "stall",
            "timeout": self._termination_reason == "timeout",
            "resets": self._resets,
            "episode_reward": float(self._episode_reward),
            "finish_time": float(snapshot.time) if self._termination_reason == "finish" else None,
        }
        return observation, float(total_reward), terminated, truncated, info


def rollout_actions(
    env: MarbleBotTrainingEnv,
    action: np.ndarray,
    *,
    max_steps: int = 400,
    seed: int = 7,
    reset_options: dict | None = None,
) -> EpisodeResult:
    env.reset(seed=seed, options=reset_options)
    info: dict[str, object] = {"termination_reason": "running", "episode_reward": 0.0}
    for _ in range(max_steps):
        _, _, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
        if terminated or truncated:
            break
    return EpisodeResult(
        finished=bool(info.get("finished", False)),
        finish_time=info.get("finish_time"),
        max_progress=float(info.get("progress", 0.0)),
        resets=int(info.get("resets", 0)),
        reward=float(info.get("episode_reward", 0.0)),
        steps=env._steps,
        termination_reason=str(info.get("termination_reason", "running")),
    )


def evaluate_policy_actions(
    policy,
    *,
    config: SimulationConfig | None = None,
    deterministic: bool = True,
    max_steps: int = 4000,
    seed: int = 7,
    reset_options: dict | None = None,
) -> EpisodeResult:
    env = MarbleBotTrainingEnv(config=config or default_training_config(with_obstacles=True))
    observation, _ = env.reset(seed=seed, options=reset_options)
    info: dict[str, object] = {"termination_reason": "running", "episode_reward": 0.0}
    for _ in range(max_steps):
        action, _ = policy.predict(observation, deterministic=deterministic)
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return EpisodeResult(
        finished=bool(info.get("finished", False)),
        finish_time=info.get("finish_time"),
        max_progress=float(info.get("progress", 0.0)),
        resets=int(info.get("resets", 0)),
        reward=float(info.get("episode_reward", 0.0)),
        steps=env._steps,
        termination_reason=str(info.get("termination_reason", "running")),
    )


def brake_velocity_scale(brake: float, dt: float, config: SimulationConfig) -> float:
    brake_amount = max(0.0, min(1.0, brake)) ** 2
    return exp(-config.brake_drag * brake_amount * dt)
