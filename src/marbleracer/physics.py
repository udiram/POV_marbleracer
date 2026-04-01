from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from math import cos, exp, radians, sin, tan
from random import Random

from panda3d.bullet import BulletBoxShape, BulletRigidBodyNode, BulletSphereShape, BulletWorld
from panda3d.core import NodePath, Vec3


GRAVITY = 9.81
SOLID_SPHERE_INERTIA_RATIO = 2.0 / 5.0
GENERATION_CLEARANCE = 0.18
MIN_PASSAGE_WIDTH = 0.56
BOOST_CLEARANCE_DISTANCE = 0.55
MARBLE_START_SURFACE_CLEARANCE = 0.005
TRACK_SEGMENT_SUBDIVISIONS = 4


@dataclass(frozen=True, slots=True)
class SectionBlueprint:
    name: str
    start_ratio: float
    end_ratio: float
    patterns: tuple[str, ...]
    boost_ratio: float


SECTION_BLUEPRINTS: tuple[SectionBlueprint, ...] = (
    SectionBlueprint("launch-drop", 0.08, 0.20, ("staggered", "lane-gate"), 0.10),
    SectionBlueprint("banked-sweep", 0.20, 0.36, ("rail-guard", "sweeper", "lane-gate"), 0.27),
    SectionBlueprint("switchback-lab", 0.36, 0.54, ("staggered", "slalom", "offset-block", "lane-gate"), 0.42),
    SectionBlueprint("compression-run", 0.54, 0.72, ("desk-bumper", "sweeper", "offset-block"), 0.60),
    SectionBlueprint("boost-corridor", 0.72, 0.88, ("rail-guard", "offset-block", "lane-gate"), 0.76),
    SectionBlueprint("final-vector", 0.88, 0.97, ("slalom", "sweeper", "offset-block", "lane-gate"), 0.88),
)


@dataclass(frozen=True, slots=True)
class GuideObstacle:
    distance_along_ramp: float
    lateral_offset: float
    length: float
    width: float
    height: float
    heading_deg: float
    kind: str = "block"


@dataclass(frozen=True, slots=True)
class BoostPad:
    distance_along_ramp: float
    lateral_offset: float
    length: float
    width: float
    acceleration: float


@dataclass(frozen=True, slots=True)
class RampSegment:
    start_distance: float
    length: float
    heading_deg: float
    bank_deg: float
    start_point: Vec3


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    angle_deg: float = 16.5
    length: float = 108.0
    height: float = 39.5
    width: float = 2.32
    ramp_thickness: float = 0.22
    marble_radius: float = 0.22
    marble_mass: float = 0.18
    gravity: float = GRAVITY
    static_friction_coeff: float = 0.32
    ramp_friction: float = 1.18
    marble_friction: float = 1.05
    obstacle_friction: float = 0.85
    rail_friction: float = 0.9
    restitution: float = 0.02
    rail_height: float = 0.46
    rail_width: float = 0.16
    steering_acceleration: float = 4.0
    brake_drag: float = 1.2
    obstacle_count: int = 10
    course_seed: int = 7
    auto_boost_pads: bool = True
    segment_headings_deg: tuple[float, ...] = (
        0.0,
        2.0,
        5.0,
        8.0,
        11.0,
        14.0,
        16.0,
        16.0,
        14.0,
        10.0,
        6.0,
        2.0,
        -2.0,
        -6.0,
        -10.0,
        -14.0,
        -16.0,
        -16.0,
        -13.0,
        -9.0,
        -5.0,
        -1.0,
        3.0,
        7.0,
    )
    segment_bank_deg: tuple[float, ...] = (
        0.0,
        1.0,
        2.0,
        4.0,
        6.0,
        9.0,
        12.0,
        12.0,
        10.0,
        7.0,
        4.0,
        1.0,
        -1.0,
        -4.0,
        -7.0,
        -10.0,
        -13.0,
        -13.0,
        -10.0,
        -7.0,
        -4.0,
        -1.0,
        2.0,
        4.0,
    )
    boost_pads: tuple[BoostPad, ...] = field(default_factory=tuple)
    obstacles: tuple[GuideObstacle, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.length <= 0.0 or self.height <= 0.0 or self.width <= 0.0:
            raise ValueError("Ramp dimensions must be positive.")
        if self.ramp_thickness <= 0.0:
            raise ValueError("Ramp thickness must be positive.")
        if self.marble_radius <= 0.0 or self.marble_mass <= 0.0:
            raise ValueError("Marble parameters must be positive.")
        if self.gravity <= 0.0:
            raise ValueError("Gravity must be positive.")
        if self.steering_acceleration < 0.0:
            raise ValueError("Steering acceleration must be non-negative.")
        if self.brake_drag < 0.0:
            raise ValueError("Brake drag must be non-negative.")
        if self.end_height <= 0.0:
            raise ValueError("Ramp exit must stay above ground.")
        if not supports_pure_rolling(self):
            raise ValueError("Static friction is too low to sustain rolling without slipping.")
        if self.obstacle_count < 0:
            raise ValueError("Obstacle count must be non-negative.")
        if not self.segment_headings_deg:
            raise ValueError("At least one ramp segment heading is required.")
        if not self.segment_bank_deg:
            object.__setattr__(self, "segment_bank_deg", tuple(0.0 for _ in self.segment_headings_deg))
        if len(self.segment_bank_deg) != len(self.segment_headings_deg):
            raise ValueError("Segment bank angles must align with the ramp heading segments.")
        if self.auto_boost_pads and not self.boost_pads:
            object.__setattr__(self, "boost_pads", default_boost_pads(self))
        if not self.obstacles and self.obstacle_count > 0:
            default_obstacle_count = type(self).__dataclass_fields__["obstacle_count"].default
            if self.obstacle_count == default_obstacle_count:
                obstacles = fixed_showcase_obstacles(self)
            else:
                requested = min(self.obstacle_count, max_generated_obstacles(self))
                obstacles = generate_obstacle_course(self, obstacle_count=requested, seed=self.course_seed)
            object.__setattr__(self, "obstacles", obstacles)
            object.__setattr__(self, "obstacle_count", len(obstacles))
        for obstacle in self.obstacles:
            if not (0.0 < obstacle.distance_along_ramp < self.length):
                raise ValueError("Obstacle must lie on the ramp.")
            if obstacle.length <= 0.0 or obstacle.width <= 0.0 or obstacle.height <= 0.0:
                raise ValueError("Obstacle dimensions must be positive.")
            if abs(obstacle.lateral_offset) + obstacle.width * 0.5 >= self.width * 0.5:
                raise ValueError("Obstacle extends off the ramp.")
        for boost_pad in self.boost_pads:
            if not (0.0 < boost_pad.distance_along_ramp < self.length):
                raise ValueError("Boost pad must lie on the ramp.")
            if boost_pad.length <= 0.0 or boost_pad.width <= 0.0 or boost_pad.acceleration <= 0.0:
                raise ValueError("Boost pad dimensions and acceleration must be positive.")
            if abs(boost_pad.lateral_offset) + boost_pad.width * 0.5 >= self.width * 0.5:
                raise ValueError("Boost pad extends off the ramp.")
        for index, obstacle in enumerate(self.obstacles):
            for other in self.obstacles[index + 1 :]:
                if obstacles_overlap(obstacle, other):
                    raise ValueError("Obstacle layout overlaps in an unplayable way.")
        for obstacle in self.obstacles:
            if not obstacle_has_clear_path(self, obstacle, self.obstacles):
                raise ValueError("Obstacle layout pinches the available lane below a safe passage width.")
            if obstacle_conflicts_with_boosts(self, obstacle, self.boost_pads):
                raise ValueError("Obstacle conflicts with a boost pad activation zone.")

    @property
    def angle_rad(self) -> float:
        return radians(self.angle_deg)

    @property
    def end_height(self) -> float:
        return self.height - self.length * sin(self.angle_rad)


@dataclass(frozen=True, slots=True)
class SimulationSnapshot:
    time: float
    position: Vec3
    path_distance: float
    linear_velocity: Vec3
    angular_velocity: Vec3
    boost_active: bool
    boost_pad_index: int | None
    obstacle_contacts: tuple[int, ...]
    rail_contacts: int
    impact_severity: float
    airborne: bool
    lateral_speed: float

    @property
    def speed(self) -> float:
        return self.linear_velocity.length()

    @property
    def total_contacts(self) -> int:
        return self.rail_contacts + sum(self.obstacle_contacts)


@dataclass(frozen=True, slots=True)
class CourseEvaluation:
    max_distance: float
    max_abs_lateral: float
    obstacle_hits: tuple[bool, ...]
    boost_activations: int
    max_impact_severity: float
    airborne_fraction: float
    stalled: bool
    exited_ramp: bool


def required_static_friction(config: SimulationConfig) -> float:
    return (SOLID_SPHERE_INERTIA_RATIO / (1.0 + SOLID_SPHERE_INERTIA_RATIO)) * tan(
        config.angle_rad
    )


def supports_pure_rolling(config: SimulationConfig) -> bool:
    return config.static_friction_coeff + 1e-12 >= required_static_friction(config)


def heading_side_vector(heading_deg: float) -> Vec3:
    heading_rad = radians(heading_deg)
    return Vec3(-sin(heading_rad), cos(heading_rad), 0.0)


def rotate_about_axis(vector: Vec3, axis: Vec3, angle_deg: float) -> Vec3:
    if abs(angle_deg) <= 1e-9:
        return Vec3(vector)
    axis = Vec3(axis)
    if axis.length_squared() <= 1e-12:
        return Vec3(vector)
    axis.normalize()
    angle_rad = radians(angle_deg)
    return (
        vector * cos(angle_rad)
        + axis.cross(vector) * sin(angle_rad)
        + axis * axis.dot(vector) * (1.0 - cos(angle_rad))
    )


def heading_tangent_vector(config: SimulationConfig, heading_deg: float) -> Vec3:
    heading_rad = radians(heading_deg)
    theta = config.angle_rad
    return Vec3(cos(heading_rad) * cos(theta), sin(heading_rad) * cos(theta), -sin(theta))


def ramp_tangent(config: SimulationConfig, distance_along_ramp: float = 0.0) -> Vec3:
    segment = ramp_segment_at_distance(config, distance_along_ramp)
    return heading_tangent_vector(config, segment.heading_deg)


def ramp_side(config: SimulationConfig, distance_along_ramp: float = 0.0) -> Vec3:
    segment = ramp_segment_at_distance(config, distance_along_ramp)
    side = rotate_about_axis(
        heading_side_vector(segment.heading_deg),
        heading_tangent_vector(config, segment.heading_deg),
        segment.bank_deg,
    )
    side.normalize()
    return side


def ramp_normal(config: SimulationConfig, distance_along_ramp: float = 0.0) -> Vec3:
    tangent = ramp_tangent(config, distance_along_ramp)
    side = ramp_side(config, distance_along_ramp)
    normal = tangent.cross(side)
    normal.normalize()
    return normal


def _interpolated_track_angles(
    headings: tuple[float, ...],
    banks: tuple[float, ...],
    sample_index: int,
    total_samples: int,
) -> tuple[float, float]:
    if len(headings) == 1:
        return headings[0], banks[0]
    coarse_position = (sample_index + 0.5) / max(1, total_samples) * len(headings) - 0.5
    base_index = max(0, min(len(headings) - 1, int(coarse_position)))
    next_index = min(base_index + 1, len(headings) - 1)
    local_t = max(0.0, min(1.0, coarse_position - base_index))
    blend = local_t * local_t * (3.0 - 2.0 * local_t)
    heading = headings[base_index] * (1.0 - blend) + headings[next_index] * blend
    bank = banks[base_index] * (1.0 - blend) + banks[next_index] * blend
    return heading, bank


@lru_cache(maxsize=128)
def _cached_ramp_segments(config: SimulationConfig) -> tuple[RampSegment, ...]:
    segment_count = len(config.segment_headings_deg) * TRACK_SEGMENT_SUBDIVISIONS
    segment_length = config.length / segment_count
    current_point = Vec3(0.0, 0.0, config.height)
    segments: list[RampSegment] = []
    for index in range(segment_count):
        heading_deg, bank_deg = _interpolated_track_angles(
            config.segment_headings_deg,
            config.segment_bank_deg,
            index,
            segment_count,
        )
        start_distance = segment_length * index
        segments.append(
            RampSegment(
                start_distance=start_distance,
                length=segment_length,
                heading_deg=heading_deg,
                bank_deg=bank_deg,
                start_point=Vec3(current_point),
            )
        )
        current_point = current_point + heading_tangent_vector(config, heading_deg) * segment_length
    return tuple(segments)


def build_ramp_segments(config: SimulationConfig) -> tuple[RampSegment, ...]:
    return _cached_ramp_segments(config)


def ramp_segment_at_distance(config: SimulationConfig, distance_along_ramp: float) -> RampSegment:
    segments = build_ramp_segments(config)
    clamped_distance = max(0.0, min(config.length - 1e-6, distance_along_ramp))
    segment_index = min(int(clamped_distance / (config.length / len(segments))), len(segments) - 1)
    return segments[segment_index]


def path_distance_for_position(config: SimulationConfig, position: Vec3) -> float:
    return _path_distance_for_position(config, position)


def _path_distance_for_position(
    config: SimulationConfig,
    position: Vec3,
    *,
    reference_distance: float | None = None,
    search_radius_segments: int | None = None,
) -> float:
    segments = build_ramp_segments(config)
    if reference_distance is None or search_radius_segments is None:
        candidate_segments = segments
    else:
        segment_length = config.length / max(1, len(segments))
        center_index = min(
            len(segments) - 1,
            max(0, int(max(0.0, min(config.length - 1e-6, reference_distance)) / segment_length)),
        )
        start_index = max(0, center_index - search_radius_segments)
        end_index = min(len(segments), center_index + search_radius_segments + 1)
        candidate_segments = segments[start_index:end_index]
    best_distance = 0.0
    best_score = float("inf")
    for segment in candidate_segments:
        tangent = heading_tangent_vector(config, segment.heading_deg)
        side = ramp_side(config, segment.start_distance)
        normal = ramp_normal(config, segment.start_distance)
        relative = position - segment.start_point
        longitudinal = max(0.0, min(segment.length, relative.dot(tangent)))
        surface_point = segment.start_point + tangent * longitudinal
        local_relative = position - surface_point
        lateral = local_relative.dot(side)
        vertical = local_relative.dot(normal)
        score = lateral * lateral + vertical * vertical
        if score < best_score:
            best_score = score
            best_distance = segment.start_distance + longitudinal
    return best_distance


def path_distance_for_position_near(
    config: SimulationConfig,
    position: Vec3,
    reference_distance: float,
    *,
    search_radius_segments: int = 24,
) -> float:
    return _path_distance_for_position(
        config,
        position,
        reference_distance=reference_distance,
        search_radius_segments=search_radius_segments,
    )


def lateral_offset_for_position(config: SimulationConfig, position: Vec3) -> float:
    path_distance = path_distance_for_position(config, position)
    center = ramp_surface_point(config, path_distance)
    return (position - center).dot(ramp_side(config, path_distance))


def is_position_off_track(
    config: SimulationConfig,
    position: Vec3,
    *,
    reference_distance: float | None = None,
) -> bool:
    path_distance = (
        max(0.0, min(config.length, reference_distance))
        if reference_distance is not None
        else path_distance_for_position(config, position)
    )
    surface_point = ramp_surface_point(config, path_distance)
    relative = position - surface_point
    lateral_offset = abs(relative.dot(ramp_side(config, path_distance)))
    normal_offset = relative.dot(ramp_normal(config, path_distance))
    if lateral_offset > config.width * 0.5 + 0.22:
        return True
    if normal_offset < -0.28:
        return True
    if position.z < surface_point.z - 1.0:
        return True
    if position.z < -2.0:
        return True
    return False


def ramp_surface_point(config: SimulationConfig, distance_along_ramp: float) -> Vec3:
    segment = ramp_segment_at_distance(config, distance_along_ramp)
    local_distance = max(0.0, min(segment.length, distance_along_ramp - segment.start_distance))
    return segment.start_point + ramp_tangent(config, distance_along_ramp) * local_distance


def ramp_center_position(config: SimulationConfig, segment: RampSegment) -> Vec3:
    center_distance = segment.start_distance + segment.length * 0.5
    return (
        ramp_surface_point(config, center_distance)
        - ramp_normal(config, center_distance) * (config.ramp_thickness * 0.5)
    )


def obstacle_center_position(config: SimulationConfig, obstacle: GuideObstacle) -> Vec3:
    return (
        ramp_surface_point(config, obstacle.distance_along_ramp)
        + ramp_side(config, obstacle.distance_along_ramp) * obstacle.lateral_offset
        + ramp_normal(config, obstacle.distance_along_ramp) * (obstacle.height * 0.5)
    )


def boost_pad_center_position(config: SimulationConfig, boost_pad: BoostPad) -> Vec3:
    return (
        ramp_surface_point(config, boost_pad.distance_along_ramp)
        + ramp_side(config, boost_pad.distance_along_ramp) * boost_pad.lateral_offset
        + ramp_normal(config, boost_pad.distance_along_ramp) * 0.014
    )


def rail_center_position(config: SimulationConfig, segment: RampSegment, lateral_offset: float) -> Vec3:
    center_distance = segment.start_distance + segment.length * 0.5
    return (
        ramp_surface_point(config, center_distance)
        + ramp_side(config, center_distance) * lateral_offset
        + ramp_normal(config, center_distance) * (config.rail_height * 0.5)
    )


def max_generated_obstacles(config: SimulationConfig) -> int:
    start_distance = max(3.6, config.length * 0.10)
    end_distance = config.length * 0.95
    if end_distance <= start_distance:
        return 0
    return max(0, int((end_distance - start_distance) / 1.05) + 1)


def usable_half_width(config: SimulationConfig) -> float:
    return config.width * 0.5 - config.rail_width - 0.06


def section_distance_ranges(config: SimulationConfig) -> tuple[tuple[SectionBlueprint, float, float], ...]:
    return tuple(
        (
            blueprint,
            max(2.0, config.length * blueprint.start_ratio),
            min(config.length - 1.4, config.length * blueprint.end_ratio),
        )
        for blueprint in SECTION_BLUEPRINTS
    )


def default_boost_pads(config: SimulationConfig) -> tuple[BoostPad, ...]:
    boost_width = min(config.width - 0.26, 1.52)
    return (
        BoostPad(config.length * 0.12, 0.0, 1.60, boost_width, 5.7),
        BoostPad(config.length * 0.33, 0.0, 1.74, boost_width, 6.1),
        BoostPad(config.length * 0.55, 0.0, 1.86, boost_width, 6.6),
        BoostPad(config.length * 0.77, 0.0, 1.94, boost_width, 7.0),
        BoostPad(config.length * 0.91, 0.0, 2.00, boost_width, 7.4),
    )


def fixed_showcase_obstacles(config: SimulationConfig) -> tuple[GuideObstacle, ...]:
    return (
        GuideObstacle(config.length * 0.18, -0.34, 0.96, 0.30, 0.24, 0.0, "block"),
        GuideObstacle(config.length * 0.24, 0.32, 0.92, 0.30, 0.24, 0.0, "block"),
        GuideObstacle(config.length * 0.39, -0.30, 0.84, 0.30, 0.24, 4.0, "block"),
        GuideObstacle(config.length * 0.46, 0.34, 0.84, 0.30, 0.24, -4.0, "block"),
        GuideObstacle(config.length * 0.58, 0.0, 1.08, 0.30, 0.26, 8.0, "block"),
        GuideObstacle(config.length * 0.66, -0.30, 0.86, 0.30, 0.24, 0.0, "block"),
        GuideObstacle(config.length * 0.72, 0.32, 0.86, 0.30, 0.24, 0.0, "block"),
        GuideObstacle(config.length * 0.82, -0.28, 0.82, 0.28, 0.24, 0.0, "block"),
        GuideObstacle(config.length * 0.87, 0.28, 0.82, 0.28, 0.24, 0.0, "block"),
        GuideObstacle(config.length * 0.94, 0.0, 0.92, 0.28, 0.24, 0.0, "block"),
    )


def obstacles_overlap(first: GuideObstacle, second: GuideObstacle) -> bool:
    longitudinal_gap = abs(first.distance_along_ramp - second.distance_along_ramp)
    lateral_gap = abs(first.lateral_offset - second.lateral_offset)
    longitudinal_clearance = (first.length + second.length) * 0.48
    lateral_clearance = (first.width + second.width) * 0.55
    return longitudinal_gap < longitudinal_clearance and lateral_gap < lateral_clearance


def obstacle_conflicts_with_boosts(
    config: SimulationConfig,
    obstacle: GuideObstacle,
    boost_pads: tuple[BoostPad, ...],
) -> bool:
    for boost_pad in boost_pads:
        longitudinal_clearance = BOOST_CLEARANCE_DISTANCE + boost_pad.length * 0.5 + obstacle.length * 0.55
        if abs(obstacle.distance_along_ramp - boost_pad.distance_along_ramp) > longitudinal_clearance:
            continue
        lateral_clearance = (obstacle.width + boost_pad.width) * 0.5 + GENERATION_CLEARANCE * 0.4
        if abs(obstacle.lateral_offset - boost_pad.lateral_offset) < lateral_clearance:
            return True
    return False


def largest_open_gap(
    config: SimulationConfig,
    obstacles: tuple[GuideObstacle, ...],
    distance: float,
) -> float:
    half_width = usable_half_width(config)
    blocked: list[tuple[float, float]] = []
    for obstacle in obstacles:
        longitudinal_influence = obstacle.length * 0.65 + 0.30
        if abs(obstacle.distance_along_ramp - distance) > longitudinal_influence:
            continue
        blocked.append(
            (
                max(-half_width, obstacle.lateral_offset - obstacle.width * 0.5 - GENERATION_CLEARANCE),
                min(half_width, obstacle.lateral_offset + obstacle.width * 0.5 + GENERATION_CLEARANCE),
            )
        )
    if not blocked:
        return half_width * 2.0

    blocked.sort()
    cursor = -half_width
    best_gap = 0.0
    for start, end in blocked:
        if start > cursor:
            best_gap = max(best_gap, start - cursor)
        cursor = max(cursor, end)
    best_gap = max(best_gap, half_width - cursor)
    return best_gap


def obstacle_has_clear_path(
    config: SimulationConfig,
    obstacle: GuideObstacle,
    obstacles: tuple[GuideObstacle, ...],
) -> bool:
    probe_offsets = (0.0, -obstacle.length * 0.22, obstacle.length * 0.22)
    return all(
        largest_open_gap(
            config,
            tuple(obstacles),
            max(0.0, min(config.length, obstacle.distance_along_ramp + offset)),
        )
        >= MIN_PASSAGE_WIDTH
        for offset in probe_offsets
    )


def layout_is_valid(config: SimulationConfig, obstacles: tuple[GuideObstacle, ...]) -> bool:
    for index, obstacle in enumerate(obstacles):
        if abs(obstacle.lateral_offset) + obstacle.width * 0.5 >= config.width * 0.5:
            return False
        if obstacle_conflicts_with_boosts(config, obstacle, config.boost_pads):
            return False
        if not obstacle_has_clear_path(config, obstacle, obstacles):
            return False
        for other in obstacles[index + 1 :]:
            if obstacles_overlap(obstacle, other):
                return False
    return True


def shift_distance_off_boosts(
    config: SimulationConfig,
    distance: float,
    *,
    min_distance: float,
    max_distance: float,
) -> float | None:
    adjusted = distance
    for boost_pad in config.boost_pads:
        clearance = BOOST_CLEARANCE_DISTANCE + boost_pad.length * 0.5 + 0.35
        delta = adjusted - boost_pad.distance_along_ramp
        if abs(delta) >= clearance:
            continue
        direction = -1.0 if delta <= 0.0 else 1.0
        adjusted = boost_pad.distance_along_ramp + direction * clearance
    if adjusted < min_distance or adjusted > max_distance:
        return None
    return adjusted


def recommended_evaluation_time(
    config: SimulationConfig,
    *,
    distance: float | None = None,
    obstacle_count: int | None = None,
) -> float:
    target_distance = config.length if distance is None else max(0.0, min(config.length, distance))
    total_obstacles = config.obstacle_count if obstacle_count is None else max(0, obstacle_count)
    return max(16.0, 6.8 + target_distance * 0.24 + total_obstacles * 0.50 + len(config.boost_pads) * 0.22)


def ramp_exit_distance(config: SimulationConfig) -> float:
    segment_length = config.length / max(1, len(build_ramp_segments(config)))
    return config.length - max(config.marble_radius, segment_length * 1.40)


def evaluate_course(
    config: SimulationConfig,
    *,
    max_time: float | None = None,
    dt: float = 1.0 / 240.0,
    stall_speed: float = 0.05,
    stall_window: float = 0.75,
) -> CourseEvaluation:
    max_time = recommended_evaluation_time(config) if max_time is None else max_time
    simulation = MarbleRampSimulation(config)
    snapshot = simulation.snapshot()
    obstacle_hits = [False] * len(config.obstacles)
    max_distance = snapshot.path_distance
    max_abs_lateral = abs(lateral_offset_for_position(config, snapshot.position))
    stall_steps = max(1, int(stall_window / dt))
    stalled = False
    activated_boosts: set[int] = set()
    max_impact_severity = snapshot.impact_severity
    airborne_steps = 0
    total_steps = 0

    for step_index in range(max(1, int(max_time / dt))):
        snapshot = simulation.step(dt)
        total_steps += 1
        max_distance = max(max_distance, snapshot.path_distance)
        max_abs_lateral = max(max_abs_lateral, abs(lateral_offset_for_position(config, snapshot.position)))
        obstacle_hits = [
            hit or contact_count > 0
            for hit, contact_count in zip(obstacle_hits, snapshot.obstacle_contacts)
        ]
        max_impact_severity = max(max_impact_severity, snapshot.impact_severity)
        if snapshot.airborne:
            airborne_steps += 1
        if snapshot.boost_pad_index is not None:
            activated_boosts.add(snapshot.boost_pad_index)
        exit_distance = ramp_exit_distance(config)
        if step_index > stall_steps and snapshot.path_distance < exit_distance and snapshot.speed < stall_speed:
            stalled = True
            break
        if snapshot.path_distance >= exit_distance:
            break

    return CourseEvaluation(
        max_distance=max_distance,
        max_abs_lateral=max_abs_lateral,
        obstacle_hits=tuple(obstacle_hits),
        boost_activations=len(activated_boosts),
        max_impact_severity=max_impact_severity,
        airborne_fraction=(airborne_steps / total_steps) if total_steps else 0.0,
        stalled=stalled,
        exited_ramp=max_distance >= ramp_exit_distance(config),
    )


def _generate_obstacle_course_uncached(
    config: SimulationConfig,
    *,
    obstacle_count: int,
    seed: int,
    max_layout_attempts: int = 12,
    max_slot_attempts: int = 20,
) -> tuple[GuideObstacle, ...]:
    if obstacle_count == 0:
        return ()

    for layout_attempt in range(max_layout_attempts):
        rng = Random(seed + layout_attempt * 17)
        placed: list[GuideObstacle] = []
        section_budgets = _allocate_section_budgets(obstacle_count)
        success = True
        for section_index, ((blueprint, start_distance, end_distance), budget) in enumerate(
            zip(section_distance_ranges(config), section_budgets)
        ):
            if budget <= 0:
                continue
            slot_count = budget
            section_progress = 0
            for slot_index in range(slot_count):
                remaining = budget - section_progress
                if remaining <= 0:
                    break
                target_distance = start_distance + (slot_index + 1) * (end_distance - start_distance) / (slot_count + 1)
                preferred_side = -1 if (slot_index + section_index) % 2 == 0 else 1
                pattern = _pick_obstacle_pattern(
                    config=config,
                    placed=tuple(placed),
                    blueprint=blueprint,
                    target_distance=target_distance,
                    preferred_side=preferred_side,
                    rng=rng,
                    seed=seed,
                    remaining=remaining,
                    max_slot_attempts=max_slot_attempts,
                    section_start=start_distance,
                    section_end=end_distance,
                )
                if pattern is None:
                    success = False
                    break
                placed.extend(pattern)
                section_progress += len(pattern)
            if not success or section_progress != budget:
                success = False
                break
        if success and len(placed) == obstacle_count:
            final_layout = tuple(sorted(placed, key=lambda obstacle: obstacle.distance_along_ramp))
            if not layout_is_valid(config, final_layout):
                continue
            final_config = replace(
                config,
                obstacles=final_layout,
                obstacle_count=len(final_layout),
            )
            evaluation = evaluate_course(
                final_config,
                max_time=recommended_evaluation_time(final_config),
                dt=1.0 / 180.0,
            )
            if (
                evaluation.exited_ramp
                and not evaluation.stalled
                and evaluation.boost_activations >= max(3, len(config.boost_pads) - 2)
                and evaluation.max_impact_severity < 0.96
                and evaluation.airborne_fraction < 0.24
            ):
                return final_layout

    raise ValueError("Could not generate a playable obstacle course for this ramp.")


@lru_cache(maxsize=64)
def _cached_generate_obstacle_course(
    config: SimulationConfig,
    obstacle_count: int,
    seed: int,
) -> tuple[GuideObstacle, ...]:
    return _generate_obstacle_course_uncached(
        config,
        obstacle_count=obstacle_count,
        seed=seed,
    )


def generate_obstacle_course(
    config: SimulationConfig,
    *,
    obstacle_count: int,
    seed: int,
    max_layout_attempts: int = 12,
    max_slot_attempts: int = 20,
) -> tuple[GuideObstacle, ...]:
    if max_layout_attempts == 12 and max_slot_attempts == 20:
        return _cached_generate_obstacle_course(config, obstacle_count, seed)
    return _generate_obstacle_course_uncached(
        config,
        obstacle_count=obstacle_count,
        seed=seed,
        max_layout_attempts=max_layout_attempts,
        max_slot_attempts=max_slot_attempts,
    )


def _pick_obstacle_pattern(
    *,
    config: SimulationConfig,
    placed: tuple[GuideObstacle, ...],
    blueprint: SectionBlueprint,
    target_distance: float,
    preferred_side: int,
    rng: Random,
    seed: int,
    remaining: int,
    max_slot_attempts: int,
    section_start: float,
    section_end: float,
) -> tuple[GuideObstacle, ...] | None:
    min_distance = (
        placed[-1].distance_along_ramp + max(0.92, placed[-1].length * 0.58)
        if placed
        else max(3.2, section_start - 0.2)
    )
    max_distance = section_end - 0.35
    if min_distance >= max_distance:
        return None

    for _ in range(max_slot_attempts):
        distance_jitter = max(0.36, (section_end - section_start) * 0.24)
        distance = min(
            max(target_distance + rng.uniform(-distance_jitter, distance_jitter), min_distance),
            max_distance,
        )
        distance = shift_distance_off_boosts(
            config,
            distance,
            min_distance=min_distance,
            max_distance=max_distance,
        )
        if distance is None:
            continue
        patterns = [name for name in blueprint.patterns if _pattern_size(name) <= remaining]
        if not patterns:
            return None
        usable_width = usable_half_width(config)
        patterns.sort(key=lambda name: abs(_pattern_size(name) - min(2, remaining)))
        pattern_name = rng.choice(patterns[: max(1, min(3, len(patterns)))])
        pattern = _build_pattern(
            pattern_name=pattern_name,
            distance=distance,
            preferred_side=preferred_side,
            usable_half_width=usable_width,
            rng=rng,
        )
        if len(pattern) > remaining:
            continue
        candidate_layout = tuple(sorted((*placed, *pattern), key=lambda obstacle: obstacle.distance_along_ramp))
        if not layout_is_valid(config, candidate_layout):
            continue
        candidate_config = replace(
            config,
            obstacles=candidate_layout,
            obstacle_count=len(candidate_layout),
            course_seed=seed,
        )
        furthest_distance = max(obstacle.distance_along_ramp for obstacle in pattern)
        evaluation = evaluate_course(
            candidate_config,
            max_time=recommended_evaluation_time(
                candidate_config,
                distance=furthest_distance + 0.70,
                obstacle_count=len(candidate_layout),
            ),
            dt=1.0 / 180.0,
        )
        if (
            not evaluation.stalled
            and evaluation.max_distance >= furthest_distance + 0.55
            and evaluation.max_impact_severity < 0.98
        ):
            return pattern
    return None


def _allocate_section_budgets(obstacle_count: int) -> tuple[int, ...]:
    weights = (1.0, 1.1, 1.3, 1.1, 1.0, 1.1)
    section_count = len(SECTION_BLUEPRINTS)
    if obstacle_count <= 0:
        return (0,) * section_count

    budgets = [0] * section_count
    if obstacle_count >= section_count:
        budgets = [1] * section_count
        remaining = obstacle_count - section_count
    else:
        remaining = obstacle_count

    for _ in range(remaining):
        best_index = max(
            range(section_count),
            key=lambda index: weights[index] / (budgets[index] + 1.0),
        )
        budgets[best_index] += 1
    return tuple(budgets)


def _pattern_size(pattern_name: str) -> int:
    return 2 if pattern_name in {"lane-gate", "slalom"} else 1


def _build_pattern(
    *,
    pattern_name: str,
    distance: float,
    preferred_side: int,
    usable_half_width: float,
    rng: Random,
) -> tuple[GuideObstacle, ...]:
    lateral_margin = 0.14
    inside_edge = usable_half_width - lateral_margin
    swing_side = preferred_side if rng.random() < 0.8 else -preferred_side

    if pattern_name == "lane-gate":
        gap_center = swing_side * rng.uniform(0.0, usable_half_width * 0.22)
        gap_width = rng.uniform(0.60, 0.74)
        block_width = max(0.22, inside_edge - gap_width * 0.5 - abs(gap_center))
        left_center = (gap_center - gap_width * 0.5 - block_width * 0.5)
        right_center = (gap_center + gap_width * 0.5 + block_width * 0.5)
        return (
            GuideObstacle(distance, left_center, 0.64, block_width, 0.26, rng.uniform(-5.0, 5.0), "eraser"),
            GuideObstacle(distance + rng.uniform(0.0, 0.06), right_center, 0.64, block_width, 0.26, rng.uniform(-5.0, 5.0), "eraser"),
        )

    if pattern_name == "desk-bumper":
        return (
            GuideObstacle(
                distance,
                swing_side * rng.uniform(usable_half_width * 0.42, usable_half_width * 0.72),
                rng.uniform(0.95, 1.25),
                rng.uniform(0.24, 0.34),
                rng.uniform(0.24, 0.34),
                -swing_side * rng.uniform(26.0, 36.0),
                "pen",
            ),
        )

    if pattern_name == "slalom":
        primary_offset = swing_side * rng.uniform(usable_half_width * 0.18, usable_half_width * 0.38)
        secondary_offset = -swing_side * rng.uniform(usable_half_width * 0.12, usable_half_width * 0.30)
        return (
            GuideObstacle(distance, primary_offset, 0.78, 0.24, 0.26, -swing_side * rng.uniform(10.0, 18.0), "block"),
            GuideObstacle(distance + rng.uniform(0.48, 0.62), secondary_offset, 0.86, 0.22, 0.24, swing_side * rng.uniform(8.0, 16.0), "eraser"),
        )

    if pattern_name == "staggered":
        return (
            GuideObstacle(
                distance,
                swing_side * rng.uniform(usable_half_width * 0.18, usable_half_width * 0.42),
                rng.uniform(0.7, 0.92),
                rng.uniform(0.18, 0.26),
                rng.uniform(0.18, 0.26),
                -swing_side * rng.uniform(8.0, 18.0),
                "eraser",
            ),
        )

    if pattern_name == "offset-block":
        return (
            GuideObstacle(
                distance,
                swing_side * rng.uniform(usable_half_width * 0.04, usable_half_width * 0.20),
                rng.uniform(0.88, 1.05),
                rng.uniform(0.24, 0.32),
                rng.uniform(0.22, 0.30),
                -swing_side * rng.uniform(12.0, 20.0),
                "block",
            ),
        )

    if pattern_name == "rail-guard":
        guard_side = swing_side
        return (
            GuideObstacle(
                distance,
                guard_side * rng.uniform(usable_half_width * 0.72, usable_half_width * 0.9),
                rng.uniform(1.5, 2.2),
                rng.uniform(0.12, 0.18),
                rng.uniform(0.1, 0.16),
                -guard_side * rng.uniform(10.0, 20.0),
                "ruler",
            ),
        )

    if pattern_name == "sweeper":
        return (
            GuideObstacle(
                distance,
                swing_side * rng.uniform(usable_half_width * 0.10, usable_half_width * 0.30),
                rng.uniform(1.2, 1.6),
                rng.uniform(0.18, 0.24),
                rng.uniform(0.20, 0.28),
                -swing_side * rng.uniform(20.0, 34.0),
                "pencil",
            ),
        )

    return (
        GuideObstacle(
            distance,
            swing_side * rng.uniform(usable_half_width * 0.18, usable_half_width * 0.42),
            rng.uniform(0.82, 1.05),
            rng.uniform(0.20, 0.28),
            rng.uniform(0.22, 0.30),
            -swing_side * rng.uniform(12.0, 22.0),
            "block",
        ),
    )


class MarbleRampSimulation:
    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        self.ramp_segments = build_ramp_segments(self.config)
        self.root = NodePath("simulation-root")
        self.world = BulletWorld()
        self.world.setGravity(Vec3(0.0, 0.0, -self.config.gravity))
        self.time = 0.0
        self.steering_input = 0.0
        self.brake_input = 0.0
        self.active_boost_pad_index: int | None = None
        self._cached_obstacle_contacts: tuple[int, ...] = ()
        self._cached_rail_contacts = 0
        self._cached_impact_severity = 0.0
        self._cached_airborne = False
        self._cached_lateral_speed = 0.0
        self._cached_path_distance = 0.0

        self.ground_np, self.ground_body = self._create_ground()
        self.ramp_nodes = self._create_ramp()
        self.rail_nodes = self._create_rails()
        self.obstacle_nodes = self._create_obstacles()
        self.marble_np, self.marble_body = self._create_marble()
        self._cached_path_distance = path_distance_for_position(self.config, self.marble_np.getPos())
        self._refresh_motion_signals(1.0 / 120.0, Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0))

    def _create_ground(self) -> tuple[NodePath, BulletRigidBodyNode]:
        body = BulletRigidBodyNode("ground")
        body.addShape(BulletBoxShape(Vec3(self.config.length + 12.0, 12.0, 0.25)))
        body.setFriction(self.config.ramp_friction)
        body.setRestitution(self.config.restitution)
        node_path = self.root.attachNewNode(body)
        node_path.setPos(self.config.length * 0.55, 0.0, -0.25)
        self.world.attachRigidBody(body)
        return node_path, body

    def _create_ramp(self) -> list[tuple[NodePath, BulletRigidBodyNode, RampSegment]]:
        ramp_nodes: list[tuple[NodePath, BulletRigidBodyNode, RampSegment]] = []
        for index, segment in enumerate(self.ramp_segments):
            body = BulletRigidBodyNode(f"ramp-{index}")
            body.addShape(
                BulletBoxShape(
                    Vec3(
                        segment.length * 0.525,
                        self.config.width * 0.54,
                        self.config.ramp_thickness * 0.5,
                    )
                )
            )
            body.setFriction(self.config.ramp_friction)
            body.setRestitution(self.config.restitution)
            node_path = self.root.attachNewNode(body)
            node_path.setPos(ramp_center_position(self.config, segment))
            node_path.setHpr(segment.heading_deg, segment.bank_deg, self.config.angle_deg)
            self.world.attachRigidBody(body)
            ramp_nodes.append((node_path, body, segment))
        return ramp_nodes

    def _create_rails(self) -> list[tuple[NodePath, BulletRigidBodyNode, RampSegment, float]]:
        rails: list[tuple[NodePath, BulletRigidBodyNode]] = []
        lateral = self.config.width * 0.5 - self.config.rail_width * 0.5
        rails: list[tuple[NodePath, BulletRigidBodyNode, RampSegment, float]] = []
        for segment in self.ramp_segments:
            for side, offset in enumerate((-lateral, lateral)):
                body = BulletRigidBodyNode(f"rail-{int(segment.start_distance)}-{side}")
                body.addShape(
                    BulletBoxShape(
                        Vec3(
                            segment.length * 0.5,
                            self.config.rail_width * 0.5,
                            self.config.rail_height * 0.5,
                        )
                    )
                )
                body.setFriction(self.config.rail_friction)
                body.setRestitution(self.config.restitution)
                node_path = self.root.attachNewNode(body)
                node_path.setPos(rail_center_position(self.config, segment, offset))
                node_path.setHpr(segment.heading_deg, segment.bank_deg, self.config.angle_deg)
                self.world.attachRigidBody(body)
                rails.append((node_path, body, segment, offset))
        return rails

    def _create_obstacles(self) -> list[tuple[NodePath, BulletRigidBodyNode, GuideObstacle]]:
        obstacle_nodes: list[tuple[NodePath, BulletRigidBodyNode, GuideObstacle]] = []
        for index, obstacle in enumerate(self.config.obstacles):
            body = BulletRigidBodyNode(f"obstacle-{index}")
            body.addShape(
                BulletBoxShape(Vec3(obstacle.length * 0.5, obstacle.width * 0.5, obstacle.height * 0.5))
            )
            body.setFriction(self.config.obstacle_friction)
            body.setRestitution(self.config.restitution)
            node_path = self.root.attachNewNode(body)
            node_path.setPos(obstacle_center_position(self.config, obstacle))
            segment = ramp_segment_at_distance(self.config, obstacle.distance_along_ramp)
            node_path.setHpr(segment.heading_deg + obstacle.heading_deg, segment.bank_deg, self.config.angle_deg)
            self.world.attachRigidBody(body)
            obstacle_nodes.append((node_path, body, obstacle))
        return obstacle_nodes

    def _create_marble(self) -> tuple[NodePath, BulletRigidBodyNode]:
        body = BulletRigidBodyNode("marble")
        body.addShape(BulletSphereShape(self.config.marble_radius))
        body.setMass(self.config.marble_mass)
        body.setFriction(self.config.marble_friction)
        body.setRestitution(self.config.restitution)
        body.setLinearDamping(0.015)
        body.setAngularDamping(0.015)
        body.setCcdMotionThreshold(1e-7)
        body.setCcdSweptSphereRadius(self.config.marble_radius * 0.98)

        node_path = self.root.attachNewNode(body)
        start_pos = ramp_surface_point(self.config, 0.0) + ramp_normal(self.config) * (
            self.config.marble_radius + MARBLE_START_SURFACE_CLEARANCE
        )
        node_path.setPos(start_pos)
        self.world.attachRigidBody(body)
        return node_path, body

    def reset(self) -> None:
        self.time = 0.0
        self.steering_input = 0.0
        self.brake_input = 0.0
        self.active_boost_pad_index = None
        start_pos = ramp_surface_point(self.config, 0.0) + ramp_normal(self.config) * (
            self.config.marble_radius + MARBLE_START_SURFACE_CLEARANCE
        )
        self.marble_np.setPos(start_pos)
        self.marble_np.setQuat(NodePath("identity").getQuat())
        self.marble_body.setLinearVelocity(Vec3(0.0, 0.0, 0.0))
        self.marble_body.setAngularVelocity(Vec3(0.0, 0.0, 0.0))
        self.marble_body.clearForces()
        self._cached_path_distance = path_distance_for_position(self.config, self.marble_np.getPos())
        self._refresh_motion_signals(1.0 / 120.0, Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0))

    def settle_start_contact(self, *, settle_steps: int = 48, dt: float = 1.0 / 960.0) -> None:
        if settle_steps <= 0 or dt <= 0.0:
            return
        for _ in range(settle_steps):
            self.world.doPhysics(dt, 1, dt)
        self.marble_body.setLinearVelocity(Vec3(0.0, 0.0, 0.0))
        self.marble_body.setAngularVelocity(Vec3(0.0, 0.0, 0.0))
        self.marble_body.clearForces()
        self._refresh_motion_signals(dt, Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0))

    def set_steering_input(self, lateral_input: float) -> None:
        self.steering_input = max(-1.0, min(1.0, lateral_input))

    def set_brake_input(self, brake_input: float) -> None:
        self.brake_input = max(0.0, min(1.0, brake_input))

    def add_forward_speed(self, speed_delta: float) -> None:
        if speed_delta <= 0.0:
            return
        path_distance = self._current_path_distance()
        forward = ramp_tangent(self.config, path_distance)
        if forward.length_squared() <= 1e-9:
            return
        forward.normalize()
        self.marble_body.setLinearVelocity(self.marble_body.getLinearVelocity() + forward * speed_delta)

    def step(self, dt: float) -> SimulationSnapshot:
        dt = max(0.0, dt)
        if dt > 0.0:
            self._apply_steering_force()
            velocity_before_step = Vec3(self.marble_body.getLinearVelocity())
            self.world.doPhysics(dt, 16, 1.0 / 960.0)
            velocity_after_physics = Vec3(self.marble_body.getLinearVelocity())
            self._refresh_motion_signals(dt, velocity_before_step, velocity_after_physics)
            self._apply_brake_drag(dt)
            self._apply_boost_pad(dt)
            self.time += dt
        return self.snapshot()

    def _apply_steering_force(self) -> None:
        if abs(self.steering_input) <= 1e-6 or self.config.steering_acceleration <= 0.0:
            return
        steering_force = Vec3(
            0.0,
            self.steering_input * self.config.marble_mass * self.config.steering_acceleration,
            0.0,
        )
        self.marble_body.applyCentralForce(steering_force)

    def _apply_brake_drag(self, dt: float) -> None:
        if self.brake_input <= 1e-6 or self.config.brake_drag <= 0.0:
            return
        brake_amount = self.brake_input * self.brake_input
        linear_damping = exp(-self.config.brake_drag * brake_amount * dt)
        angular_damping = exp(-self.config.brake_drag * brake_amount * dt * 0.45)
        self.marble_body.setLinearVelocity(self.marble_body.getLinearVelocity() * linear_damping)
        self.marble_body.setAngularVelocity(self.marble_body.getAngularVelocity() * angular_damping)

    def _apply_boost_pad(self, dt: float) -> None:
        active_boost = self.active_boost_pad()
        self.active_boost_pad_index = active_boost[0] if active_boost is not None else None
        if active_boost is None:
            return
        _, boost_pad = active_boost
        path_distance = self._current_path_distance()
        boost_direction = ramp_tangent(self.config, path_distance)
        if boost_direction.length_squared() <= 1e-9:
            return
        boost_direction.normalize()
        self.marble_body.setLinearVelocity(
            self.marble_body.getLinearVelocity() + boost_direction * (boost_pad.acceleration * dt)
        )

    def active_boost_pad(self) -> tuple[int, BoostPad] | None:
        position = self.marble_np.getPos()
        activation_margin = self.config.marble_radius
        for index, boost_pad in enumerate(self.config.boost_pads):
            center = boost_pad_center_position(self.config, boost_pad)
            tangent = ramp_tangent(self.config, boost_pad.distance_along_ramp)
            if tangent.length_squared() <= 1e-9:
                continue
            tangent.normalize()
            side = ramp_side(self.config, boost_pad.distance_along_ramp)
            normal = ramp_normal(self.config, boost_pad.distance_along_ramp)
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
            if self._cached_airborne and normal_offset > activation_margin * 0.8:
                continue
            return index, boost_pad
        return None

    def _refresh_motion_signals(
        self,
        dt: float,
        velocity_before_step: Vec3,
        velocity_after_physics: Vec3,
    ) -> None:
        self._cached_obstacle_contacts = self.obstacle_contact_counts()
        self._cached_rail_contacts = self.rail_contact_count()
        path_distance = self._current_path_distance()
        side = ramp_side(self.config, path_distance)
        normal = ramp_normal(self.config, path_distance)
        surface_point = ramp_surface_point(self.config, path_distance)
        normal_gap = (self.marble_np.getPos() - surface_point).dot(normal) - self.config.marble_radius
        self._cached_airborne = (
            normal_gap > 0.06
            and self._cached_rail_contacts == 0
            and sum(self._cached_obstacle_contacts) == 0
        )
        self._cached_lateral_speed = abs(velocity_after_physics.dot(side))

        speed_loss = max(0.0, velocity_before_step.length() - velocity_after_physics.length())
        contact_load = 0.0
        if self._cached_rail_contacts > 0:
            contact_load += 0.18 + min(0.22, self._cached_rail_contacts * 0.05)
        obstacle_contact_total = sum(self._cached_obstacle_contacts)
        if obstacle_contact_total > 0:
            contact_load += 0.32 + min(0.34, obstacle_contact_total * 0.08)
        if self._cached_airborne:
            contact_load += 0.08
        impact_candidate = min(1.0, contact_load + speed_loss * 0.22 + self._cached_lateral_speed * 0.03)
        if impact_candidate > self._cached_impact_severity:
            response = 14.0
            damping = 1.0 - exp(-response * max(dt, 1e-6))
            self._cached_impact_severity += (impact_candidate - self._cached_impact_severity) * damping
        else:
            release = exp(-5.6 * max(dt, 1e-6))
            self._cached_impact_severity *= release

    def rail_contact_count(self) -> int:
        return sum(
            self.world.contactTestPair(self.marble_body, rail_body).getNumContacts()
            for _, rail_body, _, _ in self.rail_nodes
        )

    def snapshot(self) -> SimulationSnapshot:
        return SimulationSnapshot(
            time=self.time,
            position=self.marble_np.getPos(),
            path_distance=self._current_path_distance(),
            linear_velocity=self.marble_body.getLinearVelocity(),
            angular_velocity=self.marble_body.getAngularVelocity(),
            boost_active=self.active_boost_pad_index is not None,
            boost_pad_index=self.active_boost_pad_index,
            obstacle_contacts=self._cached_obstacle_contacts,
            rail_contacts=self._cached_rail_contacts,
            impact_severity=self._cached_impact_severity,
            airborne=self._cached_airborne,
            lateral_speed=self._cached_lateral_speed,
        )

    def obstacle_contact_counts(self) -> tuple[int, ...]:
        return tuple(
            self.world.contactTestPair(self.marble_body, obstacle_body).getNumContacts()
            for _, obstacle_body, _ in self.obstacle_nodes
        )

    def _current_path_distance(self) -> float:
        candidate_distance = path_distance_for_position_near(
            self.config,
            self.marble_np.getPos(),
            self._cached_path_distance,
        )
        forward_progress = max(0.0, min(self.config.length, self.marble_np.getX()))
        if self.marble_body.getLinearVelocity().length_squared() > 0.09:
            self._cached_path_distance = max(self._cached_path_distance, candidate_distance, forward_progress)
        else:
            self._cached_path_distance = max(candidate_distance, forward_progress)
        return self._cached_path_distance
