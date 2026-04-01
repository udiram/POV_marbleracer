from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from random import Random
from typing import Callable, Sequence

from .physics import (
    MarbleRampSimulation,
    SimulationConfig,
    is_position_off_track,
    lateral_offset_for_position,
    ramp_exit_distance,
    recommended_evaluation_time,
)


Action = tuple[float, float]
Observation = tuple[float, ...]
ActionPlan = tuple[Action, ...]

DEFAULT_DISCRETE_ACTIONS: tuple[Action, ...] = (
    (-1.0, 0.0),
    (-0.5, 0.0),
    (0.0, 0.0),
    (0.5, 0.0),
    (1.0, 0.0),
    (0.0, 0.5),
    (0.0, 1.0),
)


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    total_reward: float
    completed: bool
    finish_time: float | None
    max_path_distance: float
    progress_ratio: float
    steps: int
    off_track: bool
    stalled: bool
    timed_out: bool
    boost_activations: int
    max_impact_severity: float


@dataclass(frozen=True, slots=True)
class GenerationStats:
    generation: int
    best_reward: float
    mean_reward: float
    completion_rate: float
    best_finish_time: float | None


@dataclass(frozen=True, slots=True)
class GeneticSearchResult:
    best_plan: ActionPlan
    best_result: EpisodeResult
    history: tuple[GenerationStats, ...]


class MarbleRaceEnv:
    def __init__(
        self,
        config: SimulationConfig | None = None,
        *,
        dt: float = 1.0 / 120.0,
        action_repeat: int = 4,
        max_time: float | None = None,
        stall_window: float = 0.9,
        stall_speed: float = 0.08,
        progress_reward_scale: float = 3.5,
        completion_bonus: float | None = None,
        impact_penalty_scale: float = 0.12,
        off_track_penalty: float = 6.0,
        stall_penalty: float = 3.0,
    ) -> None:
        self.config = config or SimulationConfig()
        self.dt = dt
        self.action_repeat = max(1, action_repeat)
        self.control_dt = self.dt * self.action_repeat
        self.max_time = recommended_evaluation_time(self.config) if max_time is None else max_time
        self.stall_window = stall_window
        self.stall_speed = stall_speed
        self.progress_reward_scale = progress_reward_scale
        self.completion_bonus = self.max_time if completion_bonus is None else completion_bonus
        self.impact_penalty_scale = impact_penalty_scale
        self.off_track_penalty = off_track_penalty
        self.stall_penalty = stall_penalty
        self._exit_distance = ramp_exit_distance(self.config)
        self.simulation: MarbleRampSimulation | None = None
        self.reset()

    @property
    def observation_size(self) -> int:
        return 10

    @property
    def discrete_actions(self) -> tuple[Action, ...]:
        return DEFAULT_DISCRETE_ACTIONS

    @property
    def snapshot(self):
        if self.simulation is None:
            raise RuntimeError("Environment has not been reset.")
        return self.simulation.snapshot()

    def reset(self) -> Observation:
        self.simulation = MarbleRampSimulation(self.config)
        self.simulation.settle_start_contact()
        self._steps = 0
        self._total_reward = 0.0
        self._done = False
        self._off_track = False
        self._stalled = False
        self._timed_out = False
        self._completed = False
        self._boost_activations: set[int] = set()
        self._max_path_distance = 0.0
        self._max_impact_severity = 0.0
        self._last_progress_time = 0.0
        self._last_progress_distance = 0.0
        self._last_observation = self._observation()
        return self._last_observation

    def _observation(self) -> Observation:
        snapshot = self.snapshot
        lateral_offset = lateral_offset_for_position(self.config, snapshot.position)
        return (
            snapshot.path_distance / max(self.config.length, 1e-6),
            lateral_offset / max(self.config.width * 0.5, 1e-6),
            snapshot.linear_velocity.x / 20.0,
            snapshot.linear_velocity.y / 10.0,
            snapshot.linear_velocity.z / 10.0,
            snapshot.speed / 20.0,
            snapshot.lateral_speed / 10.0,
            snapshot.impact_severity,
            1.0 if snapshot.airborne else 0.0,
            1.0 if snapshot.boost_active else 0.0,
        )

    def _clip_action(self, action: Action) -> Action:
        steering, brake = action
        return max(-1.0, min(1.0, steering)), max(0.0, min(1.0, brake))

    def decode_action(self, action: int | Sequence[float]) -> Action:
        if isinstance(action, int):
            return self.discrete_actions[int(action)]
        return self._clip_action((float(action[0]), float(action[1])))

    def step(self, action: int | Sequence[float]) -> tuple[Observation, float, bool, bool, dict[str, float | bool]]:
        if self.simulation is None:
            raise RuntimeError("Environment has not been reset.")
        if self._done:
            raise RuntimeError("Episode is complete. Call reset() before stepping again.")

        steering, brake = self.decode_action(action)
        self.simulation.set_steering_input(steering)
        self.simulation.set_brake_input(brake)

        reward = 0.0
        for _ in range(self.action_repeat):
            before = self.snapshot
            after = self.simulation.step(self.dt)
            progress_delta = max(0.0, after.path_distance - before.path_distance)
            reward += progress_delta / max(self.config.length, 1e-6) * self.progress_reward_scale
            reward -= self.dt
            reward -= max(0.0, after.impact_severity - before.impact_severity) * self.impact_penalty_scale
            self._steps += 1
            self._max_path_distance = max(self._max_path_distance, after.path_distance)
            self._max_impact_severity = max(self._max_impact_severity, after.impact_severity)
            if progress_delta > 1e-3:
                self._last_progress_time = after.time
                self._last_progress_distance = after.path_distance
            if after.boost_pad_index is not None:
                self._boost_activations.add(after.boost_pad_index)
            self._off_track = is_position_off_track(
                self.config,
                after.position,
                reference_distance=after.path_distance,
            )
            self._completed = after.path_distance >= self._exit_distance
            self._timed_out = after.time >= self.max_time
            self._stalled = (
                not self._completed
                and after.time - self._last_progress_time >= self.stall_window
                and after.path_distance - self._last_progress_distance < 0.08
                and after.speed < self.stall_speed
            )
            if self._completed or self._off_track or self._timed_out or self._stalled:
                break

        if self._completed:
            reward += self.completion_bonus - self.snapshot.time
        elif self._off_track:
            reward -= self.off_track_penalty
        elif self._stalled:
            reward -= self.stall_penalty

        self._total_reward += reward
        self._done = self._completed or self._off_track or self._timed_out or self._stalled
        self._last_observation = self._observation()
        info: dict[str, float | bool] = {
            "completed": self._completed,
            "off_track": self._off_track,
            "stalled": self._stalled,
            "timed_out": self._timed_out,
            "progress_ratio": self._max_path_distance / self._exit_distance,
            "time": self.snapshot.time,
        }
        return (
            self._last_observation,
            reward,
            self._completed or self._off_track or self._stalled,
            self._timed_out,
            info,
        )

    def result(self) -> EpisodeResult:
        snapshot = self.snapshot
        return EpisodeResult(
            total_reward=self._total_reward,
            completed=self._completed,
            finish_time=snapshot.time if self._completed else None,
            max_path_distance=self._max_path_distance,
            progress_ratio=min(1.0, self._max_path_distance / self._exit_distance),
            steps=self._steps,
            off_track=self._off_track,
            stalled=self._stalled,
            timed_out=self._timed_out,
            boost_activations=len(self._boost_activations),
            max_impact_severity=self._max_impact_severity,
        )

    def horizon_steps(self) -> int:
        return max(1, ceil(self.max_time / max(self.control_dt, 1e-6)))

    def rollout(self, action_plan: Sequence[Sequence[float]]) -> EpisodeResult:
        self.reset()
        neutral = (0.0, 0.0)
        for index in range(self.horizon_steps()):
            action = action_plan[index] if index < len(action_plan) else neutral
            _, _, terminated, truncated, _ = self.step(action)
            if terminated or truncated:
                break
        return self.result()


class GeneticActionPlanner:
    def __init__(
        self,
        env_factory: Callable[[], MarbleRaceEnv],
        *,
        population_size: int = 48,
        elite_count: int = 8,
        generations: int = 24,
        mutation_std: float = 0.16,
        brake_mutation_std: float = 0.10,
        seed: int = 0,
        keep_zero_action: bool = True,
    ) -> None:
        if elite_count <= 0 or elite_count > population_size:
            raise ValueError("Elite count must be in the range [1, population_size].")
        self.env_factory = env_factory
        self.population_size = population_size
        self.elite_count = elite_count
        self.generations = generations
        self.mutation_std = mutation_std
        self.brake_mutation_std = brake_mutation_std
        self.keep_zero_action = keep_zero_action
        self.rng = Random(seed)

    def _random_action(self) -> Action:
        return (self.rng.uniform(-1.0, 1.0), self.rng.uniform(0.0, 1.0))

    def _initial_population(self, horizon: int) -> list[list[Action]]:
        population = [
            [self._random_action() for _ in range(horizon)]
            for _ in range(self.population_size)
        ]
        if self.keep_zero_action:
            population[0] = [(0.0, 0.0) for _ in range(horizon)]
        return population

    def _evaluate_population(
        self,
        env: MarbleRaceEnv,
        population: Sequence[Sequence[Action]],
    ) -> tuple[list[float], list[EpisodeResult]]:
        scores: list[float] = []
        results: list[EpisodeResult] = []
        for candidate in population:
            result = env.rollout(candidate)
            scores.append(result.total_reward)
            results.append(result)
        return scores, results

    def _mutate_action(self, action: Action) -> Action:
        steering = max(-1.0, min(1.0, action[0] + self.rng.gauss(0.0, self.mutation_std)))
        brake = max(0.0, min(1.0, action[1] + self.rng.gauss(0.0, self.brake_mutation_std)))
        return steering, brake

    def _breed_next_generation(
        self,
        elites: Sequence[Sequence[Action]],
        horizon: int,
    ) -> list[list[Action]]:
        next_population = [list(elite) for elite in elites]
        while len(next_population) < self.population_size:
            parent_a = elites[self.rng.randrange(len(elites))]
            parent_b = elites[self.rng.randrange(len(elites))]
            child: list[Action] = []
            for step in range(horizon):
                inherited = parent_a[step] if self.rng.random() < 0.5 else parent_b[step]
                child.append(self._mutate_action(inherited))
            next_population.append(child)
        if self.keep_zero_action:
            next_population[0] = [(0.0, 0.0) for _ in range(horizon)]
        return next_population

    def optimize(self) -> GeneticSearchResult:
        env = self.env_factory()
        horizon = env.horizon_steps()
        population = self._initial_population(horizon)
        history: list[GenerationStats] = []
        best_plan: ActionPlan = tuple(population[0])
        best_result = env.rollout(best_plan)

        for generation in range(self.generations):
            scores, results = self._evaluate_population(env, population)
            ranking = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
            elites = [list(population[index]) for index in ranking[: self.elite_count]]
            best_index = ranking[0]
            generation_best = results[best_index]
            if generation_best.total_reward >= best_result.total_reward:
                best_result = generation_best
                best_plan = tuple(population[best_index])

            completed = [result for result in results if result.completed]
            history.append(
                GenerationStats(
                    generation=generation,
                    best_reward=scores[best_index],
                    mean_reward=sum(scores) / len(scores),
                    completion_rate=len(completed) / len(results),
                    best_finish_time=min((result.finish_time for result in completed), default=None),
                )
            )
            population = self._breed_next_generation(elites, horizon)

        return GeneticSearchResult(
            best_plan=best_plan,
            best_result=best_result,
            history=tuple(history),
        )


def train_genetic_controller(
    config: SimulationConfig | None = None,
    *,
    population_size: int = 48,
    elite_count: int = 8,
    generations: int = 24,
    seed: int = 0,
    dt: float = 1.0 / 120.0,
    action_repeat: int = 4,
    max_time: float | None = None,
) -> GeneticSearchResult:
    return GeneticActionPlanner(
        lambda: MarbleRaceEnv(
            config=config,
            dt=dt,
            action_repeat=action_repeat,
            max_time=max_time,
        ),
        population_size=population_size,
        elite_count=elite_count,
        generations=generations,
        seed=seed,
    ).optimize()


def _plan_to_jsonable(plan: Sequence[Action]) -> list[list[float]]:
    return [[float(steer), float(brake)] for steer, brake in plan]


def main() -> None:
    defaults = SimulationConfig()
    parser = argparse.ArgumentParser(
        description="Optimize a marble racing control plan with a genetic algorithm."
    )
    parser.add_argument("--obstacle-count", type=int, default=defaults.obstacle_count)
    parser.add_argument("--course-seed", type=int, default=defaults.course_seed)
    parser.add_argument("--population-size", type=int, default=48)
    parser.add_argument("--elite-count", type=int, default=8)
    parser.add_argument("--generations", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=1.0 / 120.0)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--max-time", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON file for the best action plan.")
    args = parser.parse_args()

    result = train_genetic_controller(
        SimulationConfig(
            obstacle_count=args.obstacle_count,
            course_seed=args.course_seed,
        ),
        population_size=args.population_size,
        elite_count=args.elite_count,
        generations=args.generations,
        seed=args.seed,
        dt=args.dt,
        action_repeat=args.action_repeat,
        max_time=args.max_time,
    )

    summary = {
        "best_reward": result.best_result.total_reward,
        "completed": result.best_result.completed,
        "finish_time": result.best_result.finish_time,
        "progress_ratio": result.best_result.progress_ratio,
        "boost_activations": result.best_result.boost_activations,
        "max_impact_severity": result.best_result.max_impact_severity,
        "history": [asdict(entry) for entry in result.history],
    }
    print(json.dumps(summary, indent=2))

    if args.output is not None:
        payload = {
            "summary": summary,
            "best_plan": _plan_to_jsonable(result.best_plan),
        }
        args.output.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
