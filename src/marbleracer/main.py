from __future__ import annotations

import argparse

from .levels import level_to_config, list_level_names, list_levels, load_level
from .physics import SimulationConfig
from .progress import reset_progress


def main() -> None:
    defaults = SimulationConfig()
    parser = argparse.ArgumentParser(description="Run the Panda3D marble ramp simulation.")
    parser.add_argument(
        "--level",
        type=str,
        help="Level name or path to a level JSON file.",
    )
    parser.add_argument(
        "--list-levels",
        action="store_true",
        help="List saved levels and exit.",
    )
    parser.add_argument(
        "--builder",
        action="store_true",
        help="Launch the standalone track builder.",
    )
    parser.add_argument(
        "--menu-disabled",
        action="store_true",
        help="Bypass the track-select menu and launch directly into the selected level.",
    )
    parser.add_argument(
        "--mode",
        choices=("time_trial", "bot_race"),
        default=None,
        help="Launch directly into a specific gameplay mode.",
    )
    parser.add_argument(
        "--reset-progress",
        action="store_true",
        help="Clear the local progression save before launching.",
    )
    parser.add_argument(
        "--bot-controller",
        choices=("heuristic", "learned", "mixed"),
        default="mixed",
        help="Select the bot driver. 'mixed' uses a learned bot when a compatible policy artifact is available.",
    )
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
    if args.list_levels:
        for level_name in list_level_names():
            print(level_name)
        return
    if args.builder:
        from .builder import main as builder_main

        builder_main()
        return
    if args.reset_progress:
        reset_progress()

    loaded_levels = tuple(list_levels())
    loaded_level = load_level(args.level) if args.level else None
    if loaded_level is not None:
        config = level_to_config(loaded_level)
        known_names = {level.name for level in loaded_levels}
        if loaded_level.name not in known_names:
            loaded_levels = loaded_levels + (loaded_level,)
    else:
        config = SimulationConfig(
            obstacle_count=args.obstacle_count,
            course_seed=args.course_seed,
        )
    from .app import MarbleRampApp

    MarbleRampApp(
        config if not loaded_levels else None,
        levels=loaded_levels if loaded_levels else None,
        initial_level=args.level,
        menu_disabled=(args.menu_disabled and args.level is not None) or args.mode is not None,
        start_mode=args.mode,
        bot_controller_mode=args.bot_controller,
    ).run()


if __name__ == "__main__":
    main()
