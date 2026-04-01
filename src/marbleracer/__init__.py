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
    "GeneticActionPlanner",
    "GuideObstacle",
    "GhostSample",
    "MarbleRaceEnv",
    "MarbleRampApp",
    "MarbleTrainingVizApp",
    "LevelProgress",
    "LevelTargetTimes",
    "ProgressData",
    "SimulationConfig",
    "TrackLevel",
    "TrainingVizConfig",
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
    "train_genetic_controller",
]


def __getattr__(name: str):
    if name == "MarbleRampApp":
        from .app import MarbleRampApp

        return MarbleRampApp
    if name in {"MarbleTrainingVizApp", "TrainingVizConfig"}:
        from .training_viz import MarbleTrainingVizApp, TrainingVizConfig

        exports = {
            "MarbleTrainingVizApp": MarbleTrainingVizApp,
            "TrainingVizConfig": TrainingVizConfig,
        }
        return exports[name]
    if name in {"GeneticActionPlanner", "MarbleRaceEnv", "train_genetic_controller"}:
        from .optimization import (
            GeneticActionPlanner,
            MarbleRaceEnv,
            train_genetic_controller,
        )

        exports = {
            "GeneticActionPlanner": GeneticActionPlanner,
            "MarbleRaceEnv": MarbleRaceEnv,
            "train_genetic_controller": train_genetic_controller,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
