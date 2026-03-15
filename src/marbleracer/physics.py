from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import cos, exp, radians, sin, tan
from random import Random

from panda3d.bullet import BulletBoxShape, BulletRigidBodyNode, BulletSphereShape, BulletWorld
from panda3d.core import NodePath, Vec3


GRAVITY = 9.81
SOLID_SPHERE_INERTIA_RATIO = 2.0 / 5.0


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
class RampSegment:
    start_distance: float
    length: float
    heading_deg: float
    start_point: Vec3


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    angle_deg: float = 18.5
    length: float = 38.0
    height: float = 13.6
    width: float = 2.1
    ramp_thickness: float = 0.16
    marble_radius: float = 0.22
    marble_mass: float = 0.18
    gravity: float = GRAVITY
    static_friction_coeff: float = 0.30
    ramp_friction: float = 1.15
    marble_friction: float = 1.05
    obstacle_friction: float = 0.85
    rail_friction: float = 0.9
    restitution: float = 0.04
    rail_height: float = 0.48
    rail_width: float = 0.16
    steering_acceleration: float = 4.25
    brake_drag: float = 3.0
    obstacle_count: int = 8
    course_seed: int = 7
    segment_headings_deg: tuple[float, ...] = (0.0, 8.0, 18.0, 28.0, 16.0, -2.0, -18.0, -28.0, -12.0, 6.0)
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
        if not self.obstacles and self.obstacle_count > max_generated_obstacles(self):
            raise ValueError(
                f"Requested {self.obstacle_count} generated obstacles, but this ramp supports at most "
                f"{max_generated_obstacles(self)}."
            )
        if not self.obstacles and self.obstacle_count > 0:
            object.__setattr__(
                self,
                "obstacles",
                generate_obstacle_course(self, obstacle_count=self.obstacle_count, seed=self.course_seed),
            )
        for obstacle in self.obstacles:
            if not (0.0 < obstacle.distance_along_ramp < self.length):
                raise ValueError("Obstacle must lie on the ramp.")
            if obstacle.length <= 0.0 or obstacle.width <= 0.0 or obstacle.height <= 0.0:
                raise ValueError("Obstacle dimensions must be positive.")
            if abs(obstacle.lateral_offset) + obstacle.width * 0.5 >= self.width * 0.5:
                raise ValueError("Obstacle extends off the ramp.")
        for index, obstacle in enumerate(self.obstacles):
            for other in self.obstacles[index + 1 :]:
                if obstacles_overlap(obstacle, other):
                    raise ValueError("Obstacle layout overlaps in an unplayable way.")

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
    obstacle_contacts: tuple[int, ...]

    @property
    def speed(self) -> float:
        return self.linear_velocity.length()

    @property
    def total_contacts(self) -> int:
        return sum(self.obstacle_contacts)


@dataclass(frozen=True, slots=True)
class CourseEvaluation:
    max_distance: float
    max_abs_lateral: float
    obstacle_hits: tuple[bool, ...]
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


def heading_tangent_vector(config: SimulationConfig, heading_deg: float) -> Vec3:
    heading_rad = radians(heading_deg)
    theta = config.angle_rad
    return Vec3(cos(heading_rad) * cos(theta), sin(heading_rad) * cos(theta), -sin(theta))


def ramp_tangent(config: SimulationConfig, distance_along_ramp: float = 0.0) -> Vec3:
    segment = ramp_segment_at_distance(config, distance_along_ramp)
    return heading_tangent_vector(config, segment.heading_deg)


def ramp_side(config: SimulationConfig, distance_along_ramp: float = 0.0) -> Vec3:
    segment = ramp_segment_at_distance(config, distance_along_ramp)
    return heading_side_vector(segment.heading_deg)


def ramp_normal(config: SimulationConfig, distance_along_ramp: float = 0.0) -> Vec3:
    tangent = ramp_tangent(config, distance_along_ramp)
    side = ramp_side(config, distance_along_ramp)
    normal = tangent.cross(side)
    normal.normalize()
    return normal


def build_ramp_segments(config: SimulationConfig) -> tuple[RampSegment, ...]:
    segment_count = len(config.segment_headings_deg)
    segment_length = config.length / segment_count
    current_point = Vec3(0.0, 0.0, config.height)
    segments: list[RampSegment] = []
    for index, heading_deg in enumerate(config.segment_headings_deg):
        start_distance = segment_length * index
        segments.append(
            RampSegment(
                start_distance=start_distance,
                length=segment_length,
                heading_deg=heading_deg,
                start_point=Vec3(current_point),
            )
        )
        current_point = current_point + heading_tangent_vector(config, heading_deg) * segment_length
    return tuple(segments)


def ramp_segment_at_distance(config: SimulationConfig, distance_along_ramp: float) -> RampSegment:
    segments = build_ramp_segments(config)
    clamped_distance = max(0.0, min(config.length - 1e-6, distance_along_ramp))
    segment_index = min(int(clamped_distance / (config.length / len(segments))), len(segments) - 1)
    return segments[segment_index]


def path_distance_for_position(config: SimulationConfig, position: Vec3) -> float:
    best_distance = 0.0
    best_score = float("inf")
    for segment in build_ramp_segments(config):
        tangent = ramp_tangent(config, segment.start_distance)
        side = ramp_side(config, segment.start_distance)
        relative = position - segment.start_point
        longitudinal = max(0.0, min(segment.length, relative.dot(tangent)))
        lateral = relative.dot(side)
        vertical = relative.z + longitudinal * sin(config.angle_rad)
        score = abs(lateral) + abs(vertical) * 0.35
        if score < best_score:
            best_score = score
            best_distance = segment.start_distance + longitudinal
    return best_distance


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


def rail_center_position(config: SimulationConfig, segment: RampSegment, lateral_offset: float) -> Vec3:
    center_distance = segment.start_distance + segment.length * 0.5
    return (
        ramp_surface_point(config, center_distance)
        + ramp_side(config, center_distance) * lateral_offset
        + ramp_normal(config, center_distance) * (config.rail_height * 0.5)
    )




def max_generated_obstacles(config: SimulationConfig) -> int:
    start_distance = min(10.0, config.length * 0.28)
    end_distance = config.length - 1.4
    if end_distance <= start_distance:
        return 0
    return max(0, int((end_distance - start_distance) / 0.75) + 1)


def obstacles_overlap(first: GuideObstacle, second: GuideObstacle) -> bool:
    longitudinal_gap = abs(first.distance_along_ramp - second.distance_along_ramp)
    lateral_gap = abs(first.lateral_offset - second.lateral_offset)
    longitudinal_clearance = (first.length + second.length) * 0.48
    lateral_clearance = (first.width + second.width) * 0.55
    return longitudinal_gap < longitudinal_clearance and lateral_gap < lateral_clearance


def evaluate_course(
    config: SimulationConfig,
    *,
    max_time: float = 8.0,
    dt: float = 1.0 / 240.0,
    stall_speed: float = 0.05,
    stall_window: float = 0.75,
) -> CourseEvaluation:
    simulation = MarbleRampSimulation(config)
    snapshot = simulation.snapshot()
    obstacle_hits = [False] * len(config.obstacles)
    max_distance = snapshot.path_distance
    max_abs_lateral = abs(snapshot.position.y)
    stall_steps = max(1, int(stall_window / dt))
    stalled = False

    for step_index in range(max(1, int(max_time / dt))):
        snapshot = simulation.step(dt)
        max_distance = max(max_distance, snapshot.path_distance)
        max_abs_lateral = max(max_abs_lateral, abs(snapshot.position.y))
        obstacle_hits = [
            hit or contact_count > 0
            for hit, contact_count in zip(obstacle_hits, snapshot.obstacle_contacts)
        ]
        if (
            step_index > stall_steps
            and snapshot.path_distance < config.length - config.marble_radius
            and snapshot.speed < stall_speed
        ):
            stalled = True
            break
        if snapshot.path_distance >= config.length - config.marble_radius:
            break

    return CourseEvaluation(
        max_distance=max_distance,
        max_abs_lateral=max_abs_lateral,
        obstacle_hits=tuple(obstacle_hits),
        stalled=stalled,
        exited_ramp=max_distance >= config.length - config.marble_radius,
    )


def generate_obstacle_course(
    config: SimulationConfig,
    *,
    obstacle_count: int,
    seed: int,
    max_layout_attempts: int = 12,
    max_slot_attempts: int = 20,
) -> tuple[GuideObstacle, ...]:
    if obstacle_count == 0:
        return ()

    start_distance = min(10.0, config.length * 0.28)
    end_distance = config.length - 1.4
    if end_distance <= start_distance:
        raise ValueError("Ramp is too short to place a generated obstacle course.")

    base_spacing = (end_distance - start_distance) / max(1, obstacle_count - 1)
    if base_spacing < 0.68:
        raise ValueError("Ramp is too short for the requested obstacle count.")

    for layout_attempt in range(max_layout_attempts):
        rng = Random(seed + layout_attempt)
        placed: list[GuideObstacle] = []
        stage_index = 0
        while len(placed) < obstacle_count:
            target_distance = start_distance + base_spacing * len(placed)
            preferred_side = -1 if stage_index % 2 == 0 else 1
            pattern = _pick_obstacle_pattern(
                config=config,
                placed=tuple(placed),
                target_distance=target_distance,
                preferred_side=preferred_side,
                rng=rng,
                seed=seed,
                remaining=obstacle_count - len(placed),
                max_slot_attempts=max_slot_attempts,
            )
            if pattern is None:
                break
            placed.extend(pattern)
            stage_index += 1
        if len(placed) == obstacle_count:
            final_layout = tuple(placed)
            final_config = replace(
                config,
                obstacles=final_layout,
                obstacle_count=len(final_layout),
            )
            evaluation = evaluate_course(final_config, max_time=24.0)
            required_hits = max(3, int(len(final_layout) * 0.5))
            if (
                evaluation.exited_ramp
                and not evaluation.stalled
                and sum(evaluation.obstacle_hits) >= required_hits
            ):
                return final_layout

    raise ValueError("Could not generate a playable obstacle course for this ramp.")


def _pick_obstacle_pattern(
    *,
    config: SimulationConfig,
    placed: tuple[GuideObstacle, ...],
    target_distance: float,
    preferred_side: int,
    rng: Random,
    seed: int,
    remaining: int,
    max_slot_attempts: int,
) -> tuple[GuideObstacle, ...] | None:
    usable_half_width = config.width * 0.5 - config.rail_width - 0.06
    min_distance = (
        placed[-1].distance_along_ramp + max(0.82, placed[-1].length * 0.55)
        if placed
        else 3.2
    )
    max_distance = config.length - 1.2
    if min_distance >= max_distance:
        return None

    for _ in range(max_slot_attempts):
        distance = min(
            max(target_distance + rng.uniform(-0.24, 0.24), min_distance),
            max_distance,
        )
        patterns = ["staggered", "desk-bumper", "rail-guard", "sweeper"]
        if remaining >= 2:
            patterns.append("pinch")
        pattern_name = rng.choice(patterns)
        pattern = _build_pattern(
            pattern_name=pattern_name,
            distance=distance,
            preferred_side=preferred_side,
            usable_half_width=usable_half_width,
            rng=rng,
        )
        if len(pattern) > remaining:
            continue
        if any(
            abs(obstacle.lateral_offset) + obstacle.width * 0.5 >= config.width * 0.5
            for obstacle in pattern
        ):
            continue
        candidate_layout = (*placed, *pattern)
        if any(
            obstacles_overlap(first, second)
            for index, first in enumerate(candidate_layout)
            for second in candidate_layout[index + 1 :]
        ):
            continue
        candidate_config = replace(
            config,
            obstacles=candidate_layout,
            obstacle_count=len(candidate_layout),
            course_seed=seed,
        )
        evaluation = evaluate_course(candidate_config, max_time=8.5 + len(candidate_layout) * 0.55)
        furthest_distance = max(obstacle.distance_along_ramp for obstacle in pattern)
        hit_count = sum(evaluation.obstacle_hits[-len(pattern) :])
        if (
            not evaluation.stalled
            and evaluation.max_distance >= furthest_distance + 0.75
            and hit_count >= max(1, len(pattern) - 1)
        ):
            return pattern
    return None


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

    if pattern_name == "pinch":
        gap_center = swing_side * rng.uniform(0.0, usable_half_width * 0.22)
        gap_width = rng.uniform(0.42, 0.58)
        block_width = max(0.22, inside_edge - gap_width * 0.5 - abs(gap_center))
        left_center = (gap_center - gap_width * 0.5 - block_width * 0.5)
        right_center = (gap_center + gap_width * 0.5 + block_width * 0.5)
        return (
            GuideObstacle(distance, left_center, 0.58, block_width, 0.24, rng.uniform(-6.0, 6.0), "eraser"),
            GuideObstacle(distance + rng.uniform(0.0, 0.08), right_center, 0.58, block_width, 0.24, rng.uniform(-6.0, 6.0), "eraser"),
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
                rng.uniform(1.3, 1.9),
                rng.uniform(0.18, 0.24),
                rng.uniform(0.20, 0.28),
                -swing_side * rng.uniform(28.0, 42.0),
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

        self.ground_np, self.ground_body = self._create_ground()
        self.ramp_nodes = self._create_ramp()
        self.rail_nodes = self._create_rails()
        self.obstacle_nodes = self._create_obstacles()
        self.marble_np, self.marble_body = self._create_marble()

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
                        segment.length * 0.5,
                        self.config.width * 0.5,
                        self.config.ramp_thickness * 0.5,
                    )
                )
            )
            body.setFriction(self.config.ramp_friction)
            body.setRestitution(self.config.restitution)
            node_path = self.root.attachNewNode(body)
            node_path.setPos(ramp_center_position(self.config, segment))
            node_path.setH(segment.heading_deg)
            node_path.setR(self.config.angle_deg)
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
                node_path.setH(segment.heading_deg)
                node_path.setR(self.config.angle_deg)
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
            node_path.setH(obstacle.heading_deg)
            node_path.setR(self.config.angle_deg)
            self.world.attachRigidBody(body)
            obstacle_nodes.append((node_path, body, obstacle))
        return obstacle_nodes

    def _create_marble(self) -> tuple[NodePath, BulletRigidBodyNode]:
        body = BulletRigidBodyNode("marble")
        body.addShape(BulletSphereShape(self.config.marble_radius))
        body.setMass(self.config.marble_mass)
        body.setFriction(self.config.marble_friction)
        body.setRestitution(self.config.restitution)
        body.setLinearDamping(0.01)
        body.setAngularDamping(0.01)
        body.setCcdMotionThreshold(self.config.marble_radius * 0.25)
        body.setCcdSweptSphereRadius(self.config.marble_radius * 0.95)

        node_path = self.root.attachNewNode(body)
        start_pos = ramp_surface_point(self.config, 0.0) + ramp_normal(self.config) * (
            self.config.marble_radius + 0.005
        )
        node_path.setPos(start_pos)
        self.world.attachRigidBody(body)
        return node_path, body

    def reset(self) -> None:
        self.time = 0.0
        self.steering_input = 0.0
        self.brake_input = 0.0
        start_pos = ramp_surface_point(self.config, 0.0) + ramp_normal(self.config) * (
            self.config.marble_radius + 0.005
        )
        self.marble_np.setPos(start_pos)
        self.marble_np.setQuat(NodePath("identity").getQuat())
        self.marble_body.setLinearVelocity(Vec3(0.0, 0.0, 0.0))
        self.marble_body.setAngularVelocity(Vec3(0.0, 0.0, 0.0))
        self.marble_body.clearForces()

    def set_steering_input(self, lateral_input: float) -> None:
        self.steering_input = max(-1.0, min(1.0, lateral_input))

    def set_brake_input(self, brake_input: float) -> None:
        self.brake_input = max(0.0, min(1.0, brake_input))

    def step(self, dt: float) -> SimulationSnapshot:
        dt = max(0.0, dt)
        if dt > 0.0:
            self._apply_steering_force()
            self.world.doPhysics(dt, 8, 1.0 / 480.0)
            self._apply_brake_drag(dt)
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
        damping = exp(-self.config.brake_drag * self.brake_input * dt)
        self.marble_body.setLinearVelocity(self.marble_body.getLinearVelocity() * damping)
        self.marble_body.setAngularVelocity(self.marble_body.getAngularVelocity() * damping)

    def snapshot(self) -> SimulationSnapshot:
        return SimulationSnapshot(
            time=self.time,
            position=self.marble_np.getPos(),
            path_distance=path_distance_for_position(self.config, self.marble_np.getPos()),
            linear_velocity=self.marble_body.getLinearVelocity(),
            angular_velocity=self.marble_body.getAngularVelocity(),
            obstacle_contacts=self.obstacle_contact_counts(),
        )

    def obstacle_contact_counts(self) -> tuple[int, ...]:
        return tuple(
            self.world.contactTestPair(self.marble_body, obstacle_body).getNumContacts()
            for _, obstacle_body, _ in self.obstacle_nodes
        )
