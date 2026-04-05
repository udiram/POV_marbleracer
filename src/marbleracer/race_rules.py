from __future__ import annotations

from dataclasses import dataclass

from panda3d.core import Vec3

from .physics import (
    SimulationConfig,
    is_lateral_within_gap,
    lane_center_offsets,
    lane_surface_width,
    path_distance_for_position,
    path_distance_for_position_near,
    ramp_normal,
    ramp_side,
    ramp_surface_point,
    ramp_tangent,
)


@dataclass(frozen=True, slots=True)
class FinishMetrics:
    longitudinal_offset: float
    lateral_offset: float
    normal_offset: float


@dataclass(frozen=True, slots=True)
class TrackSample:
    path_distance: float
    signed_lateral_offset: float
    lateral_offset: float
    normal_offset: float
    z_drop: float
    lane_half_width: float
    within_gap: bool


def finish_line_distance(config: SimulationConfig) -> float:
    return max(0.0, config.length - max(0.18, config.marble_radius * 0.8))


def finish_grace_distance(config: SimulationConfig) -> float:
    return max(3.6, config.marble_radius * 16.0)


def _sample_track(
    config: SimulationConfig,
    position: Vec3,
    reference_distance: float,
) -> TrackSample:
    clamped_reference = max(0.0, min(config.length, reference_distance))
    path_distance = path_distance_for_position_near(config, position, clamped_reference)
    surface_point = ramp_surface_point(config, path_distance)
    relative = position - surface_point
    signed_lateral_offset = relative.dot(ramp_side(config, path_distance))
    within_gap = is_lateral_within_gap(config, path_distance, signed_lateral_offset)
    if within_gap:
        lateral_offset = abs(signed_lateral_offset)
    else:
        lane_offsets = lane_center_offsets(config, path_distance)
        nearest_lane_center = min(lane_offsets, key=lambda lane_offset: abs(signed_lateral_offset - lane_offset))
        lateral_offset = abs(signed_lateral_offset - nearest_lane_center)
    normal_offset = relative.dot(ramp_normal(config, path_distance))
    return TrackSample(
        path_distance=path_distance,
        signed_lateral_offset=signed_lateral_offset,
        lateral_offset=lateral_offset,
        normal_offset=normal_offset,
        z_drop=position.z - surface_point.z,
        lane_half_width=lane_surface_width(config, path_distance) * 0.5,
        within_gap=within_gap,
    )


def track_relative_offsets(
    config: SimulationConfig,
    position: Vec3,
    reference_distance: float,
) -> tuple[float, float, float]:
    sample = _sample_track(config, position, reference_distance)
    return sample.lateral_offset, sample.normal_offset, sample.z_drop


def effective_track_distance(
    config: SimulationConfig,
    position: Vec3,
    reference_distance: float,
) -> float:
    clamped_reference = max(0.0, min(config.length, reference_distance))
    near_distance = path_distance_for_position_near(config, position, clamped_reference)
    absolute_distance = path_distance_for_position(config, position)
    if abs(absolute_distance - near_distance) <= 4.0:
        return max(clamped_reference, near_distance, absolute_distance)
    return max(clamped_reference, near_distance)


def is_definitely_off_track(
    config: SimulationConfig,
    position: Vec3,
    reference_distance: float,
) -> bool:
    sample = _sample_track(config, position, reference_distance)
    if sample.within_gap and sample.normal_offset < config.marble_radius * 1.4:
        return True
    lateral_limit = sample.lane_half_width + max(0.58, config.marble_radius * 2.8)
    if sample.lateral_offset > lateral_limit:
        return True
    if sample.normal_offset < -0.82:
        return True
    if sample.z_drop < -1.20:
        return True
    if position.z < -2.0:
        return True
    return False


def finish_metrics(
    config: SimulationConfig,
    position: Vec3,
    *,
    target_distance: float | None = None,
) -> FinishMetrics:
    final_distance = finish_line_distance(config) if target_distance is None else target_distance
    finish_point = ramp_surface_point(config, final_distance)
    finish_tangent = ramp_tangent(config, final_distance)
    finish_side = ramp_side(config, final_distance)
    finish_normal = ramp_normal(config, final_distance)
    if finish_tangent.length_squared() <= 1e-9:
        return FinishMetrics(float("-inf"), float("inf"), float("inf"))
    finish_tangent.normalize()
    relative = position - finish_point
    return FinishMetrics(
        longitudinal_offset=relative.dot(finish_tangent),
        lateral_offset=abs(relative.dot(finish_side)),
        normal_offset=relative.dot(finish_normal),
    )


def has_crossed_finish(
    config: SimulationConfig,
    position: Vec3,
    *,
    target_distance: float | None = None,
) -> bool:
    metrics = finish_metrics(config, position, target_distance=target_distance)
    if metrics.longitudinal_offset < -config.marble_radius * 0.25:
        return False
    if metrics.lateral_offset > config.width * 0.5 + config.marble_radius * 0.15:
        return False
    if metrics.normal_offset < -0.50 or metrics.normal_offset > config.marble_radius * 2.0:
        return False
    return True


def should_finish_run(
    config: SimulationConfig,
    previous_position: Vec3,
    previous_distance: float,
    current_position: Vec3,
    current_distance: float,
    *,
    target_distance: float | None = None,
) -> bool:
    final_distance = finish_line_distance(config) if target_distance is None else target_distance
    previous_metrics = finish_metrics(config, previous_position, target_distance=final_distance)
    current_metrics = finish_metrics(config, current_position, target_distance=final_distance)
    lateral_limit = config.width * 0.5 + config.marble_radius * 0.28
    normal_floor = -0.50
    normal_ceiling = config.marble_radius * 2.1
    previous_inside_lane = (
        previous_metrics.lateral_offset <= lateral_limit
        and normal_floor <= previous_metrics.normal_offset <= normal_ceiling
    )
    current_inside_lane = (
        current_metrics.lateral_offset <= lateral_limit
        and normal_floor <= current_metrics.normal_offset <= normal_ceiling
    )
    crossed_plane = previous_metrics.longitudinal_offset < 0.0 <= current_metrics.longitudinal_offset
    on_finish_surface = (
        current_metrics.longitudinal_offset >= -max(0.46, config.marble_radius * 2.1)
        and current_inside_lane
    )
    current_effective_distance = effective_track_distance(config, current_position, current_distance)
    finished_by_progress = current_effective_distance >= final_distance - max(0.34, config.marble_radius * 2.2)
    if crossed_plane and (previous_inside_lane or current_inside_lane):
        return True
    if on_finish_surface and finished_by_progress:
        return True
    return False
