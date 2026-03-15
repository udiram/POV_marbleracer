"""Panda3D marble ramp simulation."""

from .physics import GuideObstacle, SimulationConfig, evaluate_course, generate_obstacle_course

__all__ = ["GuideObstacle", "MarbleRampApp", "SimulationConfig", "evaluate_course", "generate_obstacle_course"]


def __getattr__(name: str):
    if name == "MarbleRampApp":
        from .app import MarbleRampApp

        return MarbleRampApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
