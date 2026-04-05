"""Panda3D marble ramp simulation."""

from .levels import (
    LevelTargetTimes,
    TrackLevel,
    level_from_config,
    level_to_config,
    list_level_names,
    list_levels,
    load_level,
    save_level,
)
from .physics import GuideObstacle, SimulationConfig, evaluate_course, generate_obstacle_course
from .progress import GhostSample, LevelProgress, ProgressData, load_progress, reset_progress, save_progress

__all__ = [
    "GuideObstacle",
    "GhostSample",
    "MarbleRampApp",
    "LevelProgress",
    "LevelTargetTimes",
    "ProgressData",
    "SimulationConfig",
    "TrackLevel",
    "evaluate_course",
    "generate_obstacle_course",
    "level_from_config",
    "level_to_config",
    "list_level_names",
    "list_levels",
    "load_level",
    "load_progress",
    "reset_progress",
    "save_level",
    "save_progress",
]


def __getattr__(name: str):
    if name == "MarbleRampApp":
        from .app import MarbleRampApp

        return MarbleRampApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
