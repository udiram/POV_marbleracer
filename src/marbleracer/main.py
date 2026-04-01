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
        "--reset-progress",
        action="store_true",
        help="Clear the local progression save before launching.",
    )
    parser.add_argument(
        "--train-viz",
        action="store_true",
        help="Launch the population training visualizer instead of the single-marble game.",
    )
    parser.add_argument("--population-size", type=int, default=18, help="Population size for training visualization.")
    parser.add_argument("--elite-count", type=int, default=4, help="Elite survivors for each training generation.")
    parser.add_argument("--generations", type=int, default=8, help="Number of visualized generations to run.")
    parser.add_argument("--training-seed", type=int, default=0, help="Random seed for the training population.")
    parser.add_argument("--steps-per-frame", type=int, default=4, help="Simulation steps to advance per rendered frame in training mode.")
    parser.add_argument("--output-dir", type=str, default=None, help="Artifact output directory for training visualization.")
    parser.add_argument("--capture-frames", action="store_true", help="Save each rendered training frame to disk.")
    parser.add_argument("--compile-video", action="store_true", help="Compile captured training frames into an MP4 with ffmpeg.")
    parser.add_argument("--auto-close", action="store_true", help="Close automatically after the visual training run completes.")
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
    if args.train_viz:
        from pathlib import Path

        from .training_viz import MarbleTrainingVizApp, TrainingVizConfig

        MarbleTrainingVizApp(
            config,
            TrainingVizConfig(
                population_size=args.population_size,
                elite_count=args.elite_count,
                generations=args.generations,
                seed=args.training_seed,
                steps_per_frame=args.steps_per_frame,
                output_dir=Path(args.output_dir) if args.output_dir else None,
                capture_frames=args.capture_frames,
                compile_video=args.compile_video,
                auto_close=args.auto_close,
            ),
        ).run()
        return
    from .app import MarbleRampApp

    MarbleRampApp(
        config if not loaded_levels else None,
        levels=loaded_levels if loaded_levels else None,
        initial_level=args.level,
        menu_disabled=args.menu_disabled and args.level is not None,
    ).run()


if __name__ == "__main__":
    main()
