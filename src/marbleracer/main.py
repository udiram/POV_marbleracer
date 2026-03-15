from __future__ import annotations

import argparse

from .app import MarbleRampApp
from .physics import SimulationConfig


def main() -> None:
    defaults = SimulationConfig()
    parser = argparse.ArgumentParser(description="Run the Panda3D marble ramp simulation.")
    parser.add_argument(
        "--obstacle-count",
        type=int,
        default=defaults.obstacle_count,
        help="Number of generated obstacles to place on the ramp.",
    )
    parser.add_argument(
        "--course-seed",
        type=int,
        default=defaults.course_seed,
        help="Random seed used to generate the obstacle layout.",
    )
    args = parser.parse_args()
    MarbleRampApp(
        SimulationConfig(
            obstacle_count=args.obstacle_count,
            course_seed=args.course_seed,
        )
    ).run()


if __name__ == "__main__":
    main()
