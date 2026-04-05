from __future__ import annotations

import json
from dataclasses import dataclass
from math import exp
from pathlib import Path
from typing import Protocol

import numpy as np
from panda3d.bullet import BulletRigidBodyNode
from panda3d.core import Vec3

from .physics import (
    BoostPad,
    SimulationConfig,
    boost_pad_center_position,
    lane_center_offsets,
    obstacle_center_position,
    obstacle_pose,
    path_distance_for_position,
    path_distance_for_position_near,
    ramp_normal,
    ramp_segment_at_distance,
    ramp_side,
    ramp_surface_point,
    ramp_tangent,
)

DEFAULT_POLICY_MANIFEST_NAME = "default_bot_policy_manifest.json"
DEFAULT_POLICY_WEIGHTS_NAME = "default_bot_policy_weights.npz"
DEFAULT_POLICY_LEVEL_KEY = "default-showcase"
DEFAULT_POLICY_BOT_INDEX = 1
ACTION_LOW = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
ACTION_HIGH = np.array([1.0, 1.0, 1.0], dtype=np.float32)
RESIDUAL_POLICY_ACTION_SIZE = 3
LANE_DELTA_SCALE = 0.35
SPEED_DELTA_SCALE = 2.0
HEURISTIC_LOOKAHEAD = 1.4
MAX_CONTEXT_DISTANCE = 18.0


def data_directory() -> Path:
    return Path(__file__).resolve().parent / "data"


def default_policy_manifest_path() -> Path:
    return data_directory() / DEFAULT_POLICY_MANIFEST_NAME


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def bot_lane_target_offset(
    config: SimulationConfig,
    distance: float,
    preferred_offset: float,
) -> float:
    lane_offsets = lane_center_offsets(config, distance)
    if len(lane_offsets) == 1:
        return preferred_offset
    preferred_sign = -1.0 if preferred_offset < 0.0 else 1.0
    matching_lanes = [lane_offset for lane_offset in lane_offsets if lane_offset * preferred_sign > 0.0]
    if matching_lanes:
        return min(matching_lanes, key=lambda lane_offset: abs(lane_offset - preferred_offset))
    return min(lane_offsets, key=lambda lane_offset: abs(lane_offset - preferred_offset))


def heuristic_target_speed(
    config: SimulationConfig,
    progress: float,
    speed_bias: float,
) -> float:
    return 6.2 + speed_bias + min(2.6, progress / max(config.length, 1e-6) * 2.8)


def boost_pad_at_position(
    config: SimulationConfig,
    position: Vec3,
    *,
    airborne: bool = False,
) -> tuple[int, BoostPad] | None:
    activation_margin = config.marble_radius
    for index, boost_pad in enumerate(config.boost_pads):
        center = boost_pad_center_position(config, boost_pad)
        tangent = ramp_tangent(config, boost_pad.distance_along_ramp)
        if tangent.length_squared() <= 1e-9:
            continue
        tangent.normalize()
        side = ramp_side(config, boost_pad.distance_along_ramp)
        normal = ramp_normal(config, boost_pad.distance_along_ramp)
        relative = position - center
        longitudinal_offset = abs(relative.dot(tangent))
        lateral_offset = abs(relative.dot(side))
        normal_offset = relative.dot(normal)
        if longitudinal_offset > boost_pad.length * 0.5 + activation_margin:
            continue
        if lateral_offset > boost_pad.width * 0.5 + activation_margin:
            continue
        if normal_offset < -0.12 or normal_offset > activation_margin * 1.8:
            continue
        if airborne and normal_offset > activation_margin * 0.8:
            continue
        return index, boost_pad
    return None


@dataclass(frozen=True, slots=True)
class BotSensorFrame:
    position: Vec3
    linear_velocity: Vec3
    progress: float
    time_s: float
    airborne: bool
    total_contacts: int
    active_boost_pad_index: int | None
    preferred_lane_offset: float
    speed_bias: float


@dataclass(frozen=True, slots=True)
class BotObservation:
    progress_fraction: float
    speed: float
    forward_speed: float
    lateral_speed: float
    vertical_speed: float
    signed_lane_error: float
    airborne: float
    total_contacts: float
    boost_active: float
    next_boost_distance: float
    next_boost_lateral: float
    next_obstacle_distance: float
    next_obstacle_lateral: float
    next_obstacle_width: float
    next_obstacle_dynamic: float
    next_obstacle_current_lateral: float
    next_obstacle_lateral_velocity: float
    heading_now: float
    heading_lookahead: float
    bank_now: float
    bank_lookahead: float
    base_target_lane_offset: float
    base_target_speed: float

    def as_array(self) -> np.ndarray:
        values = (
            self.progress_fraction,
            _clip(self.speed / 12.0, 0.0, 2.0),
            _clip(self.forward_speed / 12.0, -2.0, 2.0),
            _clip(self.lateral_speed / 6.0, -2.0, 2.0),
            _clip(self.vertical_speed / 6.0, -2.0, 2.0),
            _clip(self.signed_lane_error / max(0.25, 0.5), -2.0, 2.0),
            self.airborne,
            _clip(self.total_contacts / 4.0, 0.0, 2.0),
            self.boost_active,
            _clip(self.next_boost_distance / MAX_CONTEXT_DISTANCE, -1.0, 1.0),
            _clip(self.next_boost_lateral / max(0.4, 0.5), -2.0, 2.0),
            _clip(self.next_obstacle_distance / MAX_CONTEXT_DISTANCE, -1.0, 1.0),
            _clip(self.next_obstacle_lateral / max(0.4, 0.5), -2.0, 2.0),
            _clip(self.next_obstacle_width / max(0.4, 0.5), 0.0, 2.0),
            self.next_obstacle_dynamic,
            _clip(self.next_obstacle_current_lateral / max(0.4, 0.5), -2.0, 2.0),
            _clip(self.next_obstacle_lateral_velocity / 2.5, -2.0, 2.0),
            _clip(self.heading_now / 45.0, -2.0, 2.0),
            _clip(self.heading_lookahead / 45.0, -2.0, 2.0),
            _clip(self.bank_now / 20.0, -2.0, 2.0),
            _clip(self.bank_lookahead / 20.0, -2.0, 2.0),
            _clip(self.base_target_lane_offset / max(0.4, 0.5), -2.0, 2.0),
            _clip(self.base_target_speed / 12.0, 0.0, 2.0),
        )
        return np.asarray(values, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class BotControlPlan:
    target_lane_offset: float
    target_speed: float
    brake: float = 0.0


@dataclass(frozen=True, slots=True)
class BotAppliedControl:
    active_boost_pad_index: int | None
    target_lane_offset: float
    target_speed: float


class BotController(Protocol):
    def reset(self) -> None: ...

    def plan_control(self, observation: BotObservation) -> BotControlPlan: ...


@dataclass(slots=True)
class HeuristicBotController:
    def reset(self) -> None:
        return None

    def plan_control(self, observation: BotObservation) -> BotControlPlan:
        return BotControlPlan(
            target_lane_offset=observation.base_target_lane_offset,
            target_speed=observation.base_target_speed,
            brake=0.0,
        )


@dataclass(frozen=True, slots=True)
class ExportedBotPolicy:
    hidden_weights: tuple[np.ndarray, ...]
    hidden_biases: tuple[np.ndarray, ...]
    action_weight: np.ndarray
    action_bias: np.ndarray
    level_key: str
    action_low: np.ndarray
    action_high: np.ndarray

    @property
    def observation_size(self) -> int:
        return int(self.hidden_weights[0].shape[1])

    @property
    def action_size(self) -> int:
        return int(self.action_weight.shape[0])

    @property
    def is_runtime_compatible(self) -> bool:
        return self.action_size == RESIDUAL_POLICY_ACTION_SIZE

    def predict(self, observation: np.ndarray) -> np.ndarray:
        x = np.asarray(observation, dtype=np.float32)
        if x.shape != (self.observation_size,):
            raise ValueError(f"Expected observation shape {(self.observation_size,)}, got {x.shape}.")
        for weight, bias in zip(self.hidden_weights, self.hidden_biases):
            x = np.tanh(weight @ x + bias)
        action = self.action_weight @ x + self.action_bias
        return np.clip(action, self.action_low, self.action_high).astype(np.float32)

    @classmethod
    def load(cls, manifest_path: str | Path | None = None) -> ExportedBotPolicy | None:
        path = Path(manifest_path) if manifest_path is not None else default_policy_manifest_path()
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        weights_path = path.with_name(str(payload["weights_file"]))
        if not weights_path.exists():
            return None
        with np.load(weights_path) as data:
            hidden_count = int(payload["hidden_layer_count"])
            hidden_weights = tuple(data[f"hidden_weight_{index}"].astype(np.float32) for index in range(hidden_count))
            hidden_biases = tuple(data[f"hidden_bias_{index}"].astype(np.float32) for index in range(hidden_count))
            action_weight = data["action_weight"].astype(np.float32)
            action_bias = data["action_bias"].astype(np.float32)
            action_size = int(action_weight.shape[0])
            action_low = data["action_low"].astype(np.float32)
            action_high = data["action_high"].astype(np.float32)
            if action_low.shape != (action_size,) or action_high.shape != (action_size,):
                if action_size == RESIDUAL_POLICY_ACTION_SIZE:
                    action_low = ACTION_LOW.copy()
                    action_high = ACTION_HIGH.copy()
                else:
                    action_low = np.full((action_size,), -1.0, dtype=np.float32)
                    action_high = np.full((action_size,), 1.0, dtype=np.float32)
            return cls(
                hidden_weights=hidden_weights,
                hidden_biases=hidden_biases,
                action_weight=action_weight,
                action_bias=action_bias,
                level_key=str(payload["level_key"]),
                action_low=action_low,
                action_high=action_high,
            )


@dataclass(slots=True)
class ResidualLearnedBotController:
    policy: ExportedBotPolicy
    base_controller: HeuristicBotController
    config: SimulationConfig

    def reset(self) -> None:
        self.base_controller.reset()

    def plan_control(self, observation: BotObservation) -> BotControlPlan:
        base_plan = self.base_controller.plan_control(observation)
        action = self.policy.predict(observation.as_array())
        lane_delta = float(action[0]) * LANE_DELTA_SCALE
        speed_delta = float(action[1]) * SPEED_DELTA_SCALE
        brake = _clip((float(action[2]) + 1.0) * 0.5, 0.0, 1.0)
        lane_limit = self.config.width * 0.5 - self.config.marble_radius * 1.1
        return BotControlPlan(
            target_lane_offset=_clip(base_plan.target_lane_offset + lane_delta, -lane_limit, lane_limit),
            target_speed=max(0.0, base_plan.target_speed + speed_delta),
            brake=brake,
        )


def build_bot_observation(
    config: SimulationConfig,
    frame: BotSensorFrame,
) -> BotObservation:
    progress = max(0.0, min(config.length, frame.progress))
    lookahead_distance = min(config.length, progress + HEURISTIC_LOOKAHEAD)
    tangent = ramp_tangent(config, lookahead_distance)
    if tangent.length_squared() <= 1e-9:
        tangent = Vec3(1.0, 0.0, 0.0)
    tangent.normalize()
    side = ramp_side(config, lookahead_distance)
    normal = ramp_normal(config, lookahead_distance)
    velocity = frame.linear_velocity
    forward_speed = velocity.dot(tangent)
    lateral_speed = velocity.dot(side)
    vertical_speed = velocity.dot(normal)
    base_target_lane_offset = bot_lane_target_offset(config, lookahead_distance, frame.preferred_lane_offset)
    base_target_speed = heuristic_target_speed(config, progress, frame.speed_bias)
    target_point = (
        ramp_surface_point(config, lookahead_distance)
        + side * base_target_lane_offset
        + normal * (config.marble_radius + 0.01)
    )
    signed_lane_error = (target_point - frame.position).dot(side)

    next_boost_distance = MAX_CONTEXT_DISTANCE
    next_boost_lateral = 0.0
    for boost_pad in config.boost_pads:
        delta = boost_pad.distance_along_ramp - progress
        if delta < -0.5:
            continue
        next_boost_distance = min(next_boost_distance, delta)
        next_boost_lateral = boost_pad.lateral_offset - base_target_lane_offset
        break

    next_obstacle_distance = MAX_CONTEXT_DISTANCE
    next_obstacle_lateral = 0.0
    next_obstacle_width = 0.0
    next_obstacle_dynamic = 0.0
    next_obstacle_current_lateral = 0.0
    next_obstacle_lateral_velocity = 0.0
    for obstacle in config.obstacles:
        delta = obstacle.distance_along_ramp - progress
        if delta < -0.5:
            continue
        next_obstacle_distance = min(next_obstacle_distance, delta)
        next_obstacle_lateral = obstacle.lateral_offset - base_target_lane_offset
        next_obstacle_width = obstacle.width
        next_obstacle_dynamic = 0.0 if obstacle.motion_kind == "static" else 1.0
        current_position, _ = obstacle_pose(config, obstacle, frame.time_s)
        next_position, _ = obstacle_pose(config, obstacle, frame.time_s + 1.0 / 30.0)
        side_now = ramp_side(config, obstacle.distance_along_ramp)
        next_obstacle_current_lateral = (current_position - ramp_surface_point(config, obstacle.distance_along_ramp)).dot(side_now)
        next_lateral = (next_position - ramp_surface_point(config, obstacle.distance_along_ramp)).dot(side_now)
        next_obstacle_lateral_velocity = (next_lateral - next_obstacle_current_lateral) * 30.0
        break

    segment_now = ramp_segment_at_distance(config, progress)
    segment_lookahead = ramp_segment_at_distance(config, min(config.length, progress + 5.0))
    return BotObservation(
        progress_fraction=progress / max(config.length, 1e-6),
        speed=velocity.length(),
        forward_speed=forward_speed,
        lateral_speed=lateral_speed,
        vertical_speed=vertical_speed,
        signed_lane_error=signed_lane_error,
        airborne=1.0 if frame.airborne else 0.0,
        total_contacts=float(frame.total_contacts),
        boost_active=1.0 if frame.active_boost_pad_index is not None else 0.0,
        next_boost_distance=next_boost_distance,
        next_boost_lateral=next_boost_lateral,
        next_obstacle_distance=next_obstacle_distance,
        next_obstacle_lateral=next_obstacle_lateral,
        next_obstacle_width=next_obstacle_width,
        next_obstacle_dynamic=next_obstacle_dynamic,
        next_obstacle_current_lateral=next_obstacle_current_lateral,
        next_obstacle_lateral_velocity=next_obstacle_lateral_velocity,
        heading_now=segment_now.heading_deg,
        heading_lookahead=segment_lookahead.heading_deg,
        bank_now=segment_now.bank_deg,
        bank_lookahead=segment_lookahead.bank_deg,
        base_target_lane_offset=base_target_lane_offset,
        base_target_speed=base_target_speed,
    )


def monotonic_progress(
    config: SimulationConfig,
    position: Vec3,
    current_progress: float,
) -> float:
    near_progress = path_distance_for_position_near(config, position, current_progress)
    absolute_progress = path_distance_for_position(config, position)
    candidate = near_progress
    if abs(absolute_progress - near_progress) <= 4.0:
        candidate = max(candidate, absolute_progress)
    return max(current_progress, candidate)


def apply_bot_control(
    config: SimulationConfig,
    body: BulletRigidBodyNode,
    position: Vec3,
    velocity: Vec3,
    progress: float,
    plan: BotControlPlan,
    dt: float,
    *,
    airborne: bool = False,
) -> BotAppliedControl:
    lookahead_distance = min(config.length, progress + HEURISTIC_LOOKAHEAD)
    target_tangent = ramp_tangent(config, lookahead_distance)
    if target_tangent.length_squared() <= 1e-9:
        return BotAppliedControl(active_boost_pad_index=None, target_lane_offset=plan.target_lane_offset, target_speed=plan.target_speed)
    target_tangent.normalize()
    target_side = ramp_side(config, lookahead_distance)
    target_normal = ramp_normal(config, lookahead_distance)
    target_point = (
        ramp_surface_point(config, lookahead_distance)
        + target_side * plan.target_lane_offset
        + target_normal * (config.marble_radius + 0.01)
    )
    relative = target_point - position
    forward_speed = velocity.dot(target_tangent)
    lateral_speed = velocity.dot(target_side)
    vertical_speed = velocity.dot(target_normal)
    mass = config.marble_mass
    forward_force = max(0.0, plan.target_speed - forward_speed) * mass * 4.0
    lateral_force = (relative.dot(target_side) * 7.4 - lateral_speed * 2.2) * mass
    body.applyCentralForce(target_tangent * forward_force + target_side * lateral_force)

    if plan.brake > 1e-6 and config.brake_drag > 0.0:
        brake_amount = plan.brake * plan.brake
        linear_damping = exp(-config.brake_drag * brake_amount * dt)
        angular_damping = exp(-config.brake_drag * brake_amount * dt * 0.45)
        body.setLinearVelocity(body.getLinearVelocity() * linear_damping)
        body.setAngularVelocity(body.getAngularVelocity() * angular_damping)

    active_boost = boost_pad_at_position(config, position, airborne=airborne)
    if active_boost is not None:
        _, boost_pad = active_boost
        body.setLinearVelocity(body.getLinearVelocity() + target_tangent * (boost_pad.acceleration * dt))

    if relative.dot(target_normal) < -0.02 or vertical_speed > 0.55:
        body.applyCentralForce(-target_normal * mass * 2.2)

    return BotAppliedControl(
        active_boost_pad_index=active_boost[0] if active_boost is not None else None,
        target_lane_offset=plan.target_lane_offset,
        target_speed=plan.target_speed,
    )


def make_bot_controller(
    mode: str,
    *,
    level_key: str,
    bot_index: int,
    config: SimulationConfig,
    policy: ExportedBotPolicy | None,
) -> BotController:
    heuristic = HeuristicBotController()
    if mode == "heuristic":
        return heuristic
    if (
        policy is None
        or not policy.is_runtime_compatible
        or level_key != policy.level_key
        or bot_index != DEFAULT_POLICY_BOT_INDEX
    ):
        return heuristic
    return ResidualLearnedBotController(policy=policy, base_controller=heuristic, config=config)


def export_sb3_policy(policy, export_manifest_path: str | Path) -> Path:
    from torch import nn

    path = Path(export_manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    policy_net = policy.mlp_extractor.policy_net
    hidden_weights: list[np.ndarray] = []
    hidden_biases: list[np.ndarray] = []
    for module in policy_net:
        if isinstance(module, nn.Linear):
            hidden_weights.append(module.weight.detach().cpu().numpy().astype(np.float32))
            hidden_biases.append(module.bias.detach().cpu().numpy().astype(np.float32))
    action_layer = policy.action_net
    weights_path = path.with_name(DEFAULT_POLICY_WEIGHTS_NAME)
    arrays: dict[str, np.ndarray] = {
        "action_weight": action_layer.weight.detach().cpu().numpy().astype(np.float32),
        "action_bias": action_layer.bias.detach().cpu().numpy().astype(np.float32),
        "action_low": ACTION_LOW,
        "action_high": ACTION_HIGH,
    }
    for index, (weight, bias) in enumerate(zip(hidden_weights, hidden_biases)):
        arrays[f"hidden_weight_{index}"] = weight
        arrays[f"hidden_bias_{index}"] = bias
    np.savez(weights_path, **arrays)
    payload = {
        "level_key": DEFAULT_POLICY_LEVEL_KEY,
        "weights_file": weights_path.name,
        "hidden_layer_count": len(hidden_weights),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
