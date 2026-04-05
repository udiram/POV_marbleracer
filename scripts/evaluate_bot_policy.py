from __future__ import annotations

import argparse
from statistics import mean, median

from marbleracer.bot_controller import ExportedBotPolicy, make_bot_controller
from marbleracer.rl_env import DEFAULT_BOT_SPEED_BIAS, EpisodeResult, default_training_config, rollout_controller


def summarize(label: str, results: list[EpisodeResult]) -> None:
    finish_times = [result.finish_time for result in results if result.finish_time is not None]
    print(label)
    print(f"  finish_rate: {sum(result.finished for result in results) / max(1, len(results)):.3f}")
    print(f"  median_finish_time: {median(finish_times):.3f}" if finish_times else "  median_finish_time: n/a")
    print(f"  mean_finish_time: {mean(finish_times):.3f}" if finish_times else "  mean_finish_time: n/a")
    print(f"  mean_progress: {mean(result.max_progress for result in results):.3f}")
    print(f"  mean_resets: {mean(result.resets for result in results):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate heuristic vs learned default bot rollouts.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--mode", choices=("heuristic", "learned", "mixed"), default="mixed")
    parser.add_argument("--policy-manifest", type=str, default=None)
    args = parser.parse_args()

    config = default_training_config(with_obstacles=True)
    policy = ExportedBotPolicy.load(args.policy_manifest)
    heuristic_results: list[EpisodeResult] = []
    selected_results: list[EpisodeResult] = []

    for _ in range(args.episodes):
        heuristic = make_bot_controller(
            "heuristic",
            level_key="default-showcase",
            bot_index=1,
            config=config,
            policy=policy,
        )
        selected = make_bot_controller(
            args.mode,
            level_key="default-showcase",
            bot_index=1,
            config=config,
            policy=policy,
        )
        heuristic_results.append(rollout_controller(heuristic, config, speed_bias=DEFAULT_BOT_SPEED_BIAS))
        selected_results.append(rollout_controller(selected, config, speed_bias=DEFAULT_BOT_SPEED_BIAS))

    summarize("heuristic", heuristic_results)
    summarize(args.mode, selected_results)


if __name__ == "__main__":
    main()
