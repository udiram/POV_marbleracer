from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from panda3d.core import Vec3

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROGRESS_PATH = PROJECT_ROOT / ".marbleracer_progress.json"

MEDAL_ORDER: dict[str, int] = {
    "NONE": 0,
    "BRONZE": 1,
    "SILVER": 2,
    "GOLD": 3,
    "PLATINUM": 4,
}


@dataclass(frozen=True, slots=True)
class GhostSample:
    time: float
    path_distance: float
    position: tuple[float, float, float]

    @classmethod
    def from_vec3(cls, *, time: float, path_distance: float, position: Vec3) -> GhostSample:
        return cls(time=time, path_distance=path_distance, position=(float(position.x), float(position.y), float(position.z)))


@dataclass(frozen=True, slots=True)
class LevelProgress:
    unlocked: bool = False
    best_time: float | None = None
    best_medal: str = "NONE"
    ghost: tuple[GhostSample, ...] = ()


@dataclass(frozen=True, slots=True)
class ProgressData:
    levels: dict[str, LevelProgress] = field(default_factory=dict)


def progress_path(path: str | Path | None = None) -> Path:
    return Path(path) if path is not None else DEFAULT_PROGRESS_PATH


def load_progress(path: str | Path | None = None) -> ProgressData:
    resolved_path = progress_path(path)
    if not resolved_path.exists():
        return ProgressData()
    payload = json.loads(resolved_path.read_text())
    if not isinstance(payload, dict):
        return ProgressData()
    levels_payload = payload.get("levels", {})
    if not isinstance(levels_payload, dict):
        return ProgressData()
    levels: dict[str, LevelProgress] = {}
    for level_key, level_data in levels_payload.items():
        if not isinstance(level_key, str) or not isinstance(level_data, dict):
            continue
        ghost_payload = level_data.get("ghost", [])
        ghost: list[GhostSample] = []
        if isinstance(ghost_payload, list):
            for item in ghost_payload:
                if not isinstance(item, dict):
                    continue
                position = item.get("position")
                if not isinstance(position, list) or len(position) != 3:
                    continue
                try:
                    ghost.append(
                        GhostSample(
                            time=float(item["time"]),
                            path_distance=float(item.get("path_distance", 0.0)),
                            position=(float(position[0]), float(position[1]), float(position[2])),
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    continue
        best_time = level_data.get("best_time")
        if best_time is not None:
            try:
                best_time = float(best_time)
            except (TypeError, ValueError):
                best_time = None
        medal = str(level_data.get("best_medal", "NONE")).upper()
        if medal not in MEDAL_ORDER:
            medal = "NONE"
        levels[level_key] = LevelProgress(
            unlocked=bool(level_data.get("unlocked", False)),
            best_time=best_time,
            best_medal=medal,
            ghost=tuple(ghost),
        )
    return ProgressData(levels=levels)


def save_progress(progress: ProgressData, path: str | Path | None = None) -> Path:
    resolved_path = progress_path(path)
    payload = {
        "levels": {
            level_key: {
                "unlocked": level_progress.unlocked,
                "best_time": level_progress.best_time,
                "best_medal": level_progress.best_medal,
                "ghost": [asdict(sample) for sample in level_progress.ghost],
            }
            for level_key, level_progress in progress.levels.items()
        }
    }
    resolved_path.write_text(json.dumps(payload, indent=2) + "\n")
    return resolved_path


def reset_progress(path: str | Path | None = None) -> Path:
    resolved_path = progress_path(path)
    if resolved_path.exists():
        resolved_path.unlink()
    return resolved_path


def level_progress(progress: ProgressData, level_key: str) -> LevelProgress:
    return progress.levels.get(level_key, LevelProgress())


def update_level_progress(
    progress: ProgressData,
    level_key: str,
    *,
    unlocked: bool | None = None,
    best_time: float | None = None,
    best_medal: str | None = None,
    ghost: tuple[GhostSample, ...] | None = None,
) -> ProgressData:
    current = level_progress(progress, level_key)
    next_medal = current.best_medal
    if best_medal is not None and MEDAL_ORDER.get(best_medal, 0) >= MEDAL_ORDER.get(next_medal, 0):
        next_medal = best_medal
    next_best_time = current.best_time
    if best_time is not None and (next_best_time is None or best_time < next_best_time):
        next_best_time = best_time
    next_ghost = current.ghost if ghost is None else ghost
    levels = dict(progress.levels)
    levels[level_key] = LevelProgress(
        unlocked=current.unlocked if unlocked is None else unlocked or current.unlocked,
        best_time=next_best_time,
        best_medal=next_medal,
        ghost=next_ghost,
    )
    return ProgressData(levels=levels)
