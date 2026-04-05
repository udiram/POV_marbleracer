from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from marbleracer.bot_controller import (
    BotObservation,
    ExportedBotPolicy,
    HeuristicBotController,
    ResidualLearnedBotController,
    export_sb3_policy,
    make_bot_controller,
)
from marbleracer.rl_env import default_training_config


def sample_observation() -> BotObservation:
    return BotObservation(
        progress_fraction=0.25,
        speed=4.0,
        forward_speed=3.8,
        lateral_speed=0.2,
        vertical_speed=-0.1,
        signed_lane_error=0.05,
        airborne=0.0,
        total_contacts=0.0,
        boost_active=0.0,
        next_boost_distance=4.0,
        next_boost_lateral=0.1,
        next_obstacle_distance=6.0,
        next_obstacle_lateral=-0.2,
        next_obstacle_width=0.6,
        next_obstacle_dynamic=1.0,
        next_obstacle_current_lateral=-0.1,
        next_obstacle_lateral_velocity=0.3,
        heading_now=2.0,
        heading_lookahead=5.0,
        bank_now=1.0,
        bank_lookahead=4.0,
        base_target_lane_offset=0.24,
        base_target_speed=7.2,
    )


def test_exported_policy_matches_torch_actor(tmp_path) -> None:
    torch.manual_seed(7)
    policy_net = nn.Sequential(
        nn.Linear(23, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
    )
    action_net = nn.Linear(64, 3)
    policy = SimpleNamespace(
        mlp_extractor=SimpleNamespace(policy_net=policy_net),
        action_net=action_net,
    )
    manifest_path = tmp_path / "policy_manifest.json"
    export_sb3_policy(policy, manifest_path)
    exported = ExportedBotPolicy.load(manifest_path)
    assert exported is not None

    observation = sample_observation().as_array()
    with torch.no_grad():
        expected = action_net(policy_net(torch.from_numpy(observation))).numpy()
    actual = exported.predict(observation)
    np.testing.assert_allclose(actual, np.clip(expected, -1.0, 1.0), atol=1e-5)


def test_make_bot_controller_falls_back_when_policy_not_applicable() -> None:
    config = default_training_config(with_obstacles=True)
    heuristic = make_bot_controller("mixed", level_key="technical-curve", bot_index=1, config=config, policy=None)
    assert isinstance(heuristic, HeuristicBotController)


def test_learned_controller_wraps_heuristic_for_default_bot() -> None:
    config = default_training_config(with_obstacles=True)
    exported = ExportedBotPolicy(
        hidden_weights=(np.zeros((64, 23), dtype=np.float32), np.zeros((64, 64), dtype=np.float32)),
        hidden_biases=(np.zeros(64, dtype=np.float32), np.zeros(64, dtype=np.float32)),
        action_weight=np.zeros((3, 64), dtype=np.float32),
        action_bias=np.array([0.5, -0.5, 1.0], dtype=np.float32),
        level_key="default-showcase",
        action_low=np.full(3, -1.0, dtype=np.float32),
        action_high=np.full(3, 1.0, dtype=np.float32),
    )
    controller = make_bot_controller(
        "mixed",
        level_key="default-showcase",
        bot_index=1,
        config=config,
        policy=exported,
    )
    assert isinstance(controller, ResidualLearnedBotController)
    plan = controller.plan_control(sample_observation())
    assert plan.target_lane_offset > sample_observation().base_target_lane_offset
    assert plan.target_speed < sample_observation().base_target_speed
    assert plan.brake > 0.9
