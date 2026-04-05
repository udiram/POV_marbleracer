from __future__ import annotations

import numpy as np
from panda3d.core import Vec3

from marbleracer.physics import SimulationSnapshot
from marbleracer.rl_env import (
    MarbleBotTrainingEnv,
    _is_off_track_candidate,
    brake_velocity_scale,
    default_training_config,
)


def reward_sum(info: dict[str, object]) -> float:
    terms = info["reward_terms"]
    return float(sum(float(value) for value in terms.values()))


def test_training_env_reset_is_deterministic() -> None:
    env = MarbleBotTrainingEnv(config=default_training_config(with_obstacles=False, level_name="time-trial-short"))
    first, first_info = env.reset(seed=7)
    second, second_info = env.reset(seed=7)
    np.testing.assert_allclose(first, second)
    assert first_info["path_distance"] == second_info["path_distance"]


def test_training_env_step_returns_finite_values_and_reward_breakdown_matches() -> None:
    env = MarbleBotTrainingEnv(config=default_training_config(with_obstacles=False, level_name="time-trial-short"))
    observation, _ = env.reset(seed=7)
    assert observation.shape == (23,)
    next_observation, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert next_observation.shape == (23,)
    assert np.isfinite(next_observation).all()
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert np.isclose(reward, reward_sum(info))
    assert info["termination_reason"] in {"running", "finish", "off_track", "stall", "timeout"}


def test_coast_survives_initial_lock_and_makes_progress() -> None:
    env = MarbleBotTrainingEnv(
        config=default_training_config(with_obstacles=False, level_name="time-trial-short"),
        max_time=3.0,
    )
    env.reset(seed=7)
    for _ in range(18):
        _, _, terminated, truncated, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert not terminated
        assert not truncated
    assert info["progress"] > 0.9


def test_coast_finishes_short_track_without_false_off_track_reset() -> None:
    env = MarbleBotTrainingEnv(
        config=default_training_config(with_obstacles=False, level_name="time-trial-short"),
        max_time=8.0,
    )
    env.reset(seed=7)
    for _ in range(400):
        _, _, terminated, truncated, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        if terminated or truncated:
            break
    assert info["termination_reason"] == "finish"
    assert info["finished"] is True
    assert info["resets"] == 0
    assert info["finish_time"] is not None and info["finish_time"] < 6.0


def test_left_and_right_produce_opposite_lateral_displacement() -> None:
    config = default_training_config(with_obstacles=False, level_name="time-trial-short")
    left_env = MarbleBotTrainingEnv(config=config, max_time=3.0)
    right_env = MarbleBotTrainingEnv(config=config, max_time=3.0)
    left_env.reset(seed=7)
    right_env.reset(seed=7)
    for _ in range(48):
        left_env.step(np.array([1.0, 0.0], dtype=np.float32))
        right_env.step(np.array([-1.0, 0.0], dtype=np.float32))
    left_y = float(left_env.simulation.snapshot().position.y)
    right_y = float(right_env.simulation.snapshot().position.y)
    assert left_y > 0.2
    assert right_y < -0.2


def test_brake_reduces_speed_and_progress_relative_to_coast() -> None:
    config = default_training_config(with_obstacles=False, level_name="time-trial-short")
    coast_env = MarbleBotTrainingEnv(config=config, max_time=3.0)
    brake_env = MarbleBotTrainingEnv(config=config, max_time=3.0)
    coast_env.reset(seed=7)
    brake_env.reset(seed=7)
    for _ in range(60):
        coast_env.step(np.array([0.0, 0.0], dtype=np.float32))
        brake_env.step(np.array([0.0, 1.0], dtype=np.float32))
    assert brake_env.simulation.snapshot().speed < coast_env.simulation.snapshot().speed
    assert brake_env._progress < coast_env._progress
    assert brake_velocity_scale(1.0, brake_env.dt, config) < 1.0


def test_boost_pad_activation_increases_speed() -> None:
    config = default_training_config(with_obstacles=False, level_name="time-trial-short")
    env = MarbleBotTrainingEnv(config=config, max_time=3.0)
    boost_pad = config.boost_pads[0]
    env.reset(seed=7, options={"spawn_distance": max(0.0, boost_pad.distance_along_ramp - 1.0), "disable_launch_assist": True})
    before_speed = env.simulation.snapshot().speed
    triggered = False
    for _ in range(80):
        _, _, terminated, truncated, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        if info["boost_active"]:
            triggered = True
            break
        assert not terminated
        assert not truncated
    assert triggered
    assert env.simulation.snapshot().speed > before_speed


def test_rail_contact_and_obstacle_contact_are_detected() -> None:
    config = default_training_config(with_obstacles=True, level_name="default")
    rail_env = MarbleBotTrainingEnv(config=config, max_time=3.0)
    rail_env.reset(seed=7, options={"spawn_distance": 15.0, "disable_launch_assist": True})
    rail_hit = False
    for _ in range(80):
        _, _, terminated, truncated, info = rail_env.step(np.array([1.0, 0.0], dtype=np.float32))
        if info["rail_contacts"] > 0:
            rail_hit = True
            break
        if terminated or truncated:
            break
    assert rail_hit

    obstacle = next(obstacle for obstacle in config.obstacles if obstacle.lateral_offset > 0.0)
    obstacle_env = MarbleBotTrainingEnv(config=config, max_time=3.0)
    obstacle_env.reset(
        seed=7,
        options={"spawn_distance": max(0.0, obstacle.distance_along_ramp - 1.2), "disable_launch_assist": True},
    )
    obstacle_hit = False
    for _ in range(120):
        _, _, terminated, truncated, info = obstacle_env.step(np.array([1.0, 0.0], dtype=np.float32))
        if info["obstacle_contacts"] > 0:
            obstacle_hit = True
            break
        if terminated or truncated:
            break
    assert obstacle_hit


def test_finish_detection_near_finish_line() -> None:
    config = default_training_config(with_obstacles=False, level_name="time-trial-short")
    env = MarbleBotTrainingEnv(config=config, max_time=2.0)
    env.reset(seed=7, options={"spawn_distance": 37.0, "disable_launch_assist": True})
    for _ in range(80):
        _, _, terminated, truncated, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        if terminated or truncated:
            break
    assert info["termination_reason"] == "finish"
    assert info["finished"] is True
    assert info["reward_terms"]["finish"] > 0.0


def test_finish_reward_beats_early_crash_reward() -> None:
    config = default_training_config(with_obstacles=False, level_name="time-trial-short")
    finish_env = MarbleBotTrainingEnv(config=config, max_time=2.0)
    finish_env.reset(seed=7, options={"spawn_distance": 37.0, "disable_launch_assist": True})
    finish_info = None
    for _ in range(80):
        _, _, terminated, truncated, finish_info = finish_env.step(np.array([0.0, 0.0], dtype=np.float32))
        if terminated or truncated:
            break

    crash_env = MarbleBotTrainingEnv(config=config, max_time=3.0)
    crash_env.reset(seed=7)
    crash_info = None
    for _ in range(120):
        _, _, terminated, truncated, crash_info = crash_env.step(np.array([1.0, 0.0], dtype=np.float32))
        if terminated or truncated:
            break

    assert finish_info is not None and crash_info is not None
    assert finish_info["episode_reward"] > crash_info["episode_reward"]


def test_coast_reward_beats_full_brake_reward() -> None:
    config = default_training_config(with_obstacles=False, level_name="time-trial-short")
    coast_env = MarbleBotTrainingEnv(config=config, max_time=4.0)
    brake_env = MarbleBotTrainingEnv(config=config, max_time=4.0)
    coast_env.reset(seed=7)
    brake_env.reset(seed=7)
    coast_info = {}
    brake_info = {}
    for _ in range(60):
        _, _, coast_term, coast_trunc, coast_info = coast_env.step(np.array([0.0, 0.0], dtype=np.float32))
        _, _, brake_term, brake_trunc, brake_info = brake_env.step(np.array([0.0, 1.0], dtype=np.float32))
        if coast_term or coast_trunc:
            break
        if brake_term or brake_trunc:
            break
    assert coast_info["episode_reward"] > brake_info["episode_reward"]


def test_obstacle_contact_reward_is_worse_than_cleaner_line() -> None:
    config = default_training_config(with_obstacles=True, level_name="default")
    obstacle = next(obstacle for obstacle in config.obstacles if obstacle.lateral_offset > 0.0)
    hit_env = MarbleBotTrainingEnv(config=config, max_time=3.0)
    avoid_env = MarbleBotTrainingEnv(config=config, max_time=3.0)
    options = {"spawn_distance": max(0.0, obstacle.distance_along_ramp - 1.2), "disable_launch_assist": True}
    hit_env.reset(seed=7, options=options)
    avoid_env.reset(seed=7, options=options)
    hit_info = {}
    avoid_info = {}
    for _ in range(60):
        _, _, hit_term, hit_trunc, hit_info = hit_env.step(np.array([1.0, 0.0], dtype=np.float32))
        _, _, avoid_term, avoid_trunc, avoid_info = avoid_env.step(np.array([-1.0, 0.0], dtype=np.float32))
        if hit_term or hit_trunc or avoid_term or avoid_trunc:
            break
    assert hit_info["obstacle_contacts"] >= avoid_info["obstacle_contacts"]
    assert hit_info["episode_reward"] < avoid_info["episode_reward"]


def test_supported_late_obstacle_state_is_not_an_off_track_candidate() -> None:
    config = default_training_config(with_obstacles=True, level_name="default")
    # Regression for the checkpoint replay that was cutting off near the last obstacle
    # despite the marble still having contact and sitting close to the lane surface.
    snapshot = SimulationSnapshot(
        time=13.2833333333329,
        position=Vec3(74.76734924316406, 5.175302028656006, 17.24524688720703),
        path_distance=74.76734924316406,
        linear_velocity=Vec3(1.9566723108291626, 0.0, 0.0),
        angular_velocity=Vec3(0.0, 0.0, 0.0),
        rail_contacts=0,
        obstacle_contacts=(0, 0, 0, 0),
        airborne=False,
        impact_severity=0.0,
        boost_pad_index=None,
        boost_active=False,
        lateral_speed=0.0,
    )
    assert _is_off_track_candidate(config, snapshot, 74.76734924316406) is False


def test_clear_below_track_freefall_is_an_off_track_candidate() -> None:
    config = default_training_config(with_obstacles=False, level_name="time-trial-short")
    snapshot = SimulationSnapshot(
        time=2.0,
        position=Vec3(18.0, 2.0, 4.0),
        path_distance=18.0,
        linear_velocity=Vec3(3.0, 0.0, -9.0),
        angular_velocity=Vec3(0.0, 0.0, 0.0),
        boost_active=False,
        boost_pad_index=None,
        obstacle_contacts=(),
        rail_contacts=0,
        impact_severity=0.0,
        airborne=True,
        lateral_speed=0.0,
    )
    assert _is_off_track_candidate(config, snapshot, 18.0) is True
