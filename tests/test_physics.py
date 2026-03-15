from math import cos, isclose, sin, tan

from panda3d.core import Vec3

from marbleracer.physics import (
    MarbleRampSimulation,
    SimulationConfig,
    evaluate_course,
    is_position_off_track,
    obstacle_center_position,
    ramp_normal,
    ramp_side,
    ramp_surface_point,
    recommended_evaluation_time,
    required_static_friction,
)


def run_for(simulation: MarbleRampSimulation, duration: float, dt: float = 1.0 / 240.0) -> None:
    for _ in range(int(duration / dt)):
        simulation.step(dt)


def test_required_static_friction_matches_solid_sphere_result() -> None:
    config = SimulationConfig(
        angle_deg=30.0,
        length=20.0,
        height=13.0,
        static_friction_coeff=0.3,
        obstacle_count=0,
        auto_boost_pads=False,
    )
    expected = (2.0 / 7.0) * tan(config.angle_rad)
    assert isclose(required_static_friction(config), expected, rel_tol=1e-9)


def test_marble_accelerates_down_the_ramp() -> None:
    simulation = MarbleRampSimulation()
    start = simulation.snapshot()
    run_for(simulation, 0.8)
    end = simulation.snapshot()
    assert end.path_distance > start.path_distance
    assert end.position.z < start.position.z
    assert end.speed > 1.0


def test_steering_input_changes_lateral_motion() -> None:
    simulation = MarbleRampSimulation(SimulationConfig(obstacle_count=0))
    simulation.set_steering_input(1.0)
    run_for(simulation, 0.8)
    steered = simulation.snapshot()

    baseline = MarbleRampSimulation(SimulationConfig(obstacle_count=0))
    run_for(baseline, 0.8)
    neutral = baseline.snapshot()

    assert steered.position.y > neutral.position.y + 0.5


def test_brake_input_reduces_speed() -> None:
    free_roll = MarbleRampSimulation(SimulationConfig(obstacle_count=0))
    run_for(free_roll, 1.2)
    free_state = free_roll.snapshot()

    braking = MarbleRampSimulation(SimulationConfig(obstacle_count=0))
    run_for(braking, 0.6)
    braking.set_brake_input(1.0)
    run_for(braking, 0.6)
    braking_state = braking.snapshot()

    assert braking_state.speed < free_state.speed
    assert braking_state.path_distance < free_state.path_distance


def test_snapshot_exposes_coarse_physics_signals() -> None:
    simulation = MarbleRampSimulation(SimulationConfig(obstacle_count=0))
    run_for(simulation, 0.3)
    snapshot = simulation.snapshot()

    assert snapshot.impact_severity >= 0.0
    assert snapshot.rail_contacts >= 0
    assert snapshot.lateral_speed >= 0.0
    assert isinstance(snapshot.airborne, bool)


def test_default_course_contacts_some_obstacles() -> None:
    simulation = MarbleRampSimulation()
    touched = [False] * len(simulation.config.obstacles)
    for _ in range(4200):
        snapshot = simulation.step(1.0 / 240.0)
        touched = [flag or count > 0 for flag, count in zip(touched, snapshot.obstacle_contacts)]
        if sum(touched) >= 3:
            break
    assert sum(touched) >= 3


def test_default_course_exits_ramp_without_stalling() -> None:
    config = SimulationConfig()
    result = evaluate_course(config, max_time=recommended_evaluation_time(config))
    assert result.exited_ramp
    assert not result.stalled
    assert sum(result.obstacle_hits) >= 3
    assert result.boost_activations >= 3
    assert result.max_impact_severity < 0.9
    assert result.airborne_fraction < 0.2
    assert 0.20 < result.max_abs_lateral < config.width * 0.5
    assert 6 <= len(config.obstacles) <= 8


def test_obstacles_are_seated_on_ramp_surface() -> None:
    config = SimulationConfig()
    for obstacle in config.obstacles:
        center = obstacle_center_position(config, obstacle)
        ramp_surface_z = config.height - obstacle.distance_along_ramp * sin(config.angle_rad)
        expected_center_z = ramp_surface_z + cos(config.angle_rad) * (obstacle.height * 0.5)
        assert isclose(center.z, expected_center_z, rel_tol=1e-6, abs_tol=1e-5)


def test_can_request_dense_generated_course() -> None:
    config = SimulationConfig(obstacle_count=10, course_seed=7)
    assert len(config.obstacles) == 10
    result = evaluate_course(config, max_time=recommended_evaluation_time(config))
    assert result.exited_ramp
    assert not result.stalled


def test_default_course_is_long_simple_and_fixed() -> None:
    config = SimulationConfig()
    assert config.length >= 80.0
    assert len(config.segment_headings_deg) >= 18
    assert 6 <= len(config.obstacles) <= 8
    assert {obstacle.kind for obstacle in config.obstacles} == {"block"}
    assert len(config.boost_pads) == 4


def test_off_track_detection_flags_fallthrough_and_side_exit() -> None:
    config = SimulationConfig()
    distance = config.length * 0.35
    on_track = ramp_surface_point(config, distance) + ramp_normal(config, distance) * config.marble_radius
    below_track = on_track + Vec3(0.0, 0.0, -1.3)
    side_exit = on_track + ramp_side(config, distance) * config.width

    assert not is_position_off_track(config, on_track, reference_distance=distance)
    assert is_position_off_track(config, below_track, reference_distance=distance)
    assert is_position_off_track(config, side_exit, reference_distance=distance)
