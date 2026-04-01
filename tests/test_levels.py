from __future__ import annotations

from marbleracer.levels import (
    LevelTargetTimes,
    TrackLevel,
    level_from_config,
    level_to_config,
    list_level_names,
    list_levels,
    load_level,
    save_level,
)
from marbleracer.physics import BoostPad, GuideObstacle, SimulationConfig


def test_default_level_can_be_loaded_by_name() -> None:
    level = load_level("default")
    config = level_to_config(level)

    assert level.name == "Default Showcase"
    assert config.length == level.track["length"]
    assert len(config.boost_pads) == 5
    assert len(config.obstacles) == 10
    assert level.target_times is not None
    assert level.unlock_index == 0
    assert "default" in list_level_names()


def test_level_round_trip_preserves_track_boosts_and_obstacles(tmp_path) -> None:
    level = TrackLevel(
        name="Round Trip",
        description="Custom validation level.",
        track={
            "angle_deg": 18.0,
            "length": 64.0,
            "height": 25.0,
            "width": 2.4,
            "segment_headings_deg": (0.0, 6.0, -3.0, 4.0),
            "segment_bank_deg": (0.0, 2.0, -2.0, 1.0),
        },
        boost_pads=(
            BoostPad(10.0, 0.0, 1.4, 1.2, 6.0),
            BoostPad(28.0, 0.3, 1.6, 0.8, 6.5),
        ),
        obstacles=(
            GuideObstacle(18.0, -0.3, 0.8, 0.25, 0.22, 5.0, "block"),
            GuideObstacle(36.0, 0.4, 0.9, 0.3, 0.24, -5.0, "block"),
        ),
        difficulty="hard",
        theme="technical",
        target_times=LevelTargetTimes(gold=11.0, silver=12.0, bronze=13.5),
        unlock_index=2,
    )

    saved_path = save_level(level, tmp_path / "round-trip.json")
    reloaded = load_level(saved_path)
    config = level_to_config(reloaded)

    assert reloaded.name == level.name
    assert reloaded.description == level.description
    assert config.length == 64.0
    assert config.width == 2.4
    assert config.segment_headings_deg == (0.0, 6.0, -3.0, 4.0)
    assert len(config.boost_pads) == 2
    assert len(config.obstacles) == 2
    assert reloaded.target_times == LevelTargetTimes(gold=11.0, silver=12.0, bronze=13.5)
    assert reloaded.unlock_index == 2


def test_level_from_default_config_matches_current_fixed_course() -> None:
    config = SimulationConfig()
    level = level_from_config(config, name="Snapshot")
    restored = level_to_config(level)

    assert restored.length == config.length
    assert restored.segment_headings_deg == config.segment_headings_deg
    assert restored.segment_bank_deg == config.segment_bank_deg
    assert restored.boost_pads == config.boost_pads
    assert restored.obstacles == config.obstacles


def test_saved_levels_are_sorted_by_unlock_then_name() -> None:
    levels = list_levels()

    assert [level.name for level in levels[:2]] == ["Default Showcase", "Time Trial Short"]
    assert levels[-1].unlock_index >= levels[0].unlock_index
