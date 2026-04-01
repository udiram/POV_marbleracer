from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .physics import BoostPad, GuideObstacle, SimulationConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEVELS_DIR = PROJECT_ROOT / "levels"


@dataclass(frozen=True, slots=True)
class LevelTargetTimes:
    gold: float
    silver: float
    bronze: float


@dataclass(frozen=True, slots=True)
class TrackLevel:
    name: str
    description: str
    track: dict[str, object]
    boost_pads: tuple[BoostPad, ...]
    obstacles: tuple[GuideObstacle, ...]
    difficulty: str = "custom"
    theme: str = "showcase"
    target_times: LevelTargetTimes | None = None
    unlock_index: int = 0
    editor: dict[str, object] | None = None


def level_directory() -> Path:
    LEVELS_DIR.mkdir(parents=True, exist_ok=True)
    return LEVELS_DIR


def list_level_paths() -> list[Path]:
    return sorted(level_directory().glob("*.json"))


def list_level_names() -> list[str]:
    return [path.stem for path in list_level_paths()]


def list_levels() -> list[TrackLevel]:
    levels = [load_level(path) for path in list_level_paths()]
    return sorted(levels, key=lambda level: (level.unlock_index, level.name.lower()))


def resolve_level_path(level_ref: str | Path) -> Path:
    path = Path(level_ref)
    if path.suffix == ".json" and path.exists():
        return path
    if path.exists():
        return path
    candidate = level_directory() / f"{path.stem}.json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Level {level_ref!r} was not found.")


def _coerce_float_tuple(values: object, field_name: str) -> tuple[float, ...]:
    if not isinstance(values, list):
        raise ValueError(f"{field_name} must be a list of numbers.")
    try:
        return tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain only numbers.") from exc


def _coerce_boost_pads(items: object) -> tuple[BoostPad, ...]:
    if not isinstance(items, list):
        raise ValueError("boost_pads must be a list.")
    try:
        return tuple(
            BoostPad(
                distance_along_ramp=float(item["distance_along_ramp"]),
                lateral_offset=float(item["lateral_offset"]),
                length=float(item["length"]),
                width=float(item["width"]),
                acceleration=float(item["acceleration"]),
            )
            for item in items
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid boost pad entry in level file.") from exc


def _coerce_obstacles(items: object) -> tuple[GuideObstacle, ...]:
    if not isinstance(items, list):
        raise ValueError("obstacles must be a list.")
    try:
        return tuple(
            GuideObstacle(
                distance_along_ramp=float(item["distance_along_ramp"]),
                lateral_offset=float(item["lateral_offset"]),
                length=float(item["length"]),
                width=float(item["width"]),
                height=float(item["height"]),
                heading_deg=float(item.get("heading_deg", 0.0)),
                kind=str(item.get("kind", "block")),
            )
            for item in items
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid obstacle entry in level file.") from exc


def _coerce_target_times(payload: object) -> LevelTargetTimes | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("metadata.target_times must be an object.")
    try:
        target_times = LevelTargetTimes(
            gold=float(payload["gold"]),
            silver=float(payload["silver"]),
            bronze=float(payload["bronze"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("metadata.target_times must define numeric gold, silver, and bronze times.") from exc
    if not (target_times.gold <= target_times.silver <= target_times.bronze):
        raise ValueError("metadata.target_times must satisfy gold <= silver <= bronze.")
    return target_times


def load_level(level_ref: str | Path) -> TrackLevel:
    path = resolve_level_path(level_ref)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Level file must contain a JSON object.")
    track = payload.get("track")
    if not isinstance(track, dict):
        raise ValueError("Level file is missing a track object.")
    headings = _coerce_float_tuple(track.get("segment_headings_deg"), "track.segment_headings_deg")
    banks = _coerce_float_tuple(track.get("segment_bank_deg"), "track.segment_bank_deg")
    metadata = payload.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError("Level metadata must be a JSON object when provided.")
    return TrackLevel(
        name=str(payload.get("name", path.stem.replace("_", " ").title())),
        description=str(payload.get("description", "")),
        track={
            "angle_deg": float(track.get("angle_deg", SimulationConfig.angle_deg)),
            "length": float(track.get("length", SimulationConfig.length)),
            "height": float(track.get("height", SimulationConfig.height)),
            "width": float(track.get("width", SimulationConfig.width)),
            "segment_headings_deg": headings,
            "segment_bank_deg": banks,
        },
        boost_pads=_coerce_boost_pads(payload.get("boost_pads", [])),
        obstacles=_coerce_obstacles(payload.get("obstacles", [])),
        difficulty=str(metadata.get("difficulty", "custom")),
        theme=str(metadata.get("theme", "showcase")),
        target_times=_coerce_target_times(metadata.get("target_times")),
        unlock_index=int(metadata.get("unlock_index", 0)),
        editor=payload.get("editor") if isinstance(payload.get("editor"), dict) else None,
    )


def level_to_config(level: TrackLevel) -> SimulationConfig:
    track = level.track
    return SimulationConfig(
        angle_deg=float(track["angle_deg"]),
        length=float(track["length"]),
        height=float(track["height"]),
        width=float(track["width"]),
        segment_headings_deg=tuple(float(value) for value in track["segment_headings_deg"]),
        segment_bank_deg=tuple(float(value) for value in track["segment_bank_deg"]),
        boost_pads=level.boost_pads,
        obstacles=level.obstacles,
        auto_boost_pads=False,
        obstacle_count=len(level.obstacles),
    )


def save_level(level: TrackLevel, destination: str | Path) -> Path:
    path = Path(destination)
    if path.suffix != ".json":
        path = path.with_suffix(".json")
    if not path.is_absolute():
        path = level_directory() / path.name
    level_to_config(level)
    payload = {
        "name": level.name,
        "description": level.description,
        "metadata": {
            "difficulty": level.difficulty,
            "theme": level.theme,
            "unlock_index": level.unlock_index,
        },
        "track": {
            "angle_deg": float(level.track["angle_deg"]),
            "length": float(level.track["length"]),
            "height": float(level.track["height"]),
            "width": float(level.track["width"]),
            "segment_headings_deg": [float(value) for value in level.track["segment_headings_deg"]],
            "segment_bank_deg": [float(value) for value in level.track["segment_bank_deg"]],
        },
        "boost_pads": [asdict(boost_pad) for boost_pad in level.boost_pads],
        "obstacles": [asdict(obstacle) for obstacle in level.obstacles],
    }
    if level.target_times is not None:
        payload["metadata"]["target_times"] = asdict(level.target_times)
    if level.editor:
        payload["editor"] = level.editor
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def level_from_config(
    config: SimulationConfig,
    *,
    name: str = "Custom Level",
    description: str = "",
    difficulty: str = "custom",
    theme: str = "showcase",
    target_times: LevelTargetTimes | None = None,
    unlock_index: int = 0,
) -> TrackLevel:
    return TrackLevel(
        name=name,
        description=description,
        track={
            "angle_deg": config.angle_deg,
            "length": config.length,
            "height": config.height,
            "width": config.width,
            "segment_headings_deg": tuple(config.segment_headings_deg),
            "segment_bank_deg": tuple(config.segment_bank_deg),
        },
        boost_pads=tuple(config.boost_pads),
        obstacles=tuple(config.obstacles),
        difficulty=difficulty,
        theme=theme,
        target_times=target_times,
        unlock_index=unlock_index,
        editor=None,
    )
