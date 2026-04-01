from marbleracer.optimization import GeneticActionPlanner, MarbleRaceEnv
from marbleracer.physics import SimulationConfig


def short_test_config() -> SimulationConfig:
    return SimulationConfig(
        length=32.0,
        height=12.0,
        obstacle_count=0,
        auto_boost_pads=False,
    )


def test_neutral_plan_completes_empty_course() -> None:
    env = MarbleRaceEnv(
        short_test_config(),
        dt=1.0 / 120.0,
        action_repeat=4,
        max_time=8.0,
    )
    plan = [(0.0, 0.0) for _ in range(env.horizon_steps())]
    result = env.rollout(plan)

    assert result.completed
    assert result.finish_time is not None
    assert result.progress_ratio == 1.0
    assert result.finish_time < env.max_time


def test_hard_steering_is_worse_than_neutral_on_short_course() -> None:
    env = MarbleRaceEnv(
        short_test_config(),
        dt=1.0 / 120.0,
        action_repeat=4,
        max_time=8.0,
    )
    neutral = [(0.0, 0.0) for _ in range(env.horizon_steps())]
    hard_steer = [(1.0, 0.0) for _ in range(env.horizon_steps())]
    neutral_result = env.rollout(neutral)
    steer_result = env.rollout(hard_steer)

    assert steer_result.completed
    assert neutral_result.finish_time is not None
    assert steer_result.finish_time is not None
    assert steer_result.finish_time > neutral_result.finish_time
    assert steer_result.total_reward < neutral_result.total_reward


def test_time_based_reward_prefers_finishing_over_braking() -> None:
    env = MarbleRaceEnv(
        short_test_config(),
        dt=1.0 / 120.0,
        action_repeat=4,
        max_time=8.0,
    )
    neutral = [(0.0, 0.0) for _ in range(env.horizon_steps())]
    full_brake = [(0.0, 1.0) for _ in range(env.horizon_steps())]

    neutral_result = env.rollout(neutral)
    braking_result = env.rollout(full_brake)

    assert neutral_result.completed
    assert not braking_result.completed
    assert neutral_result.total_reward > braking_result.total_reward


def test_genetic_planner_tracks_best_seen_plan() -> None:
    planner = GeneticActionPlanner(
        lambda: MarbleRaceEnv(
            short_test_config(),
            dt=1.0 / 120.0,
            action_repeat=4,
            max_time=6.0,
        ),
        population_size=6,
        elite_count=2,
        generations=2,
        seed=5,
    )

    result = planner.optimize()

    assert len(result.history) == 2
    assert len(result.best_plan) > 0
    assert len(result.best_plan[0]) == 2
    assert result.best_result.total_reward == max(entry.best_reward for entry in result.history)
