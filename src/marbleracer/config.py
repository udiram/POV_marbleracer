from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from panda3d.core import Vec3, Vec4


def _vec3(values: list[float]) -> Vec3:
    return Vec3(*values)


def _vec4(values: list[float]) -> Vec4:
    return Vec4(*values)


@dataclass(slots=True)
class MarbleTuning:
    radius: float
    mass: float
    steering_torque: float
    max_speed: float
    linear_damping: float
    angular_damping: float
    grip_assist: float
    air_control: float
    recovery_height: float


@dataclass(slots=True)
class PieceConfig:
    name: str
    kind: str
    size: Vec3
    pos: Vec3
    hpr: Vec3
    color: Vec4


@dataclass(slots=True)
class Checkpoint:
    name: str
    pos: Vec3
    radius: float


@dataclass(slots=True)
class HazardConfig:
    name: str
    size: Vec3
    base_pos: Vec3
    axis: Vec3
    amplitude: float
    speed: float
    color: Vec4


@dataclass(slots=True)
class TrackConfig:
    name: str
    laps: int
    table_size: Vec3
    table_pos: Vec3
    table_color: Vec4
    marble_tuning: MarbleTuning
    start_positions: list[Vec3]
    pieces: list[PieceConfig]
    rails: list[PieceConfig]
    checkpoints: list[Checkpoint]
    hazards: list[HazardConfig]


def load_track_config(path: str | Path) -> TrackConfig:
    raw = json.loads(Path(path).read_text())
    marble_tuning = MarbleTuning(**raw["marble_tuning"])
    pieces = [
        PieceConfig(
            name=piece["name"],
            kind=piece["kind"],
            size=_vec3(piece["size"]),
            pos=_vec3(piece["pos"]),
            hpr=_vec3(piece["hpr"]),
            color=_vec4(piece["color"]),
        )
        for piece in raw["pieces"]
    ]
    rails = [
        PieceConfig(
            name=f"rail_{index}",
            kind="box",
            size=_vec3(rail["size"]),
            pos=_vec3(rail["pos"]),
            hpr=Vec3(0, 0, 0),
            color=Vec4(0.25, 0.21, 0.18, 1.0),
        )
        for index, rail in enumerate(raw.get("rails", []))
    ]
    checkpoints = [
        Checkpoint(name=item["name"], pos=_vec3(item["pos"]), radius=item["radius"])
        for item in raw["checkpoints"]
    ]
    hazards = [
        HazardConfig(
            name=item["name"],
            size=_vec3(item["size"]),
            base_pos=_vec3(item["base_pos"]),
            axis=_vec3(item["axis"]),
            amplitude=item["amplitude"],
            speed=item["speed"],
            color=_vec4(item["color"]),
        )
        for item in raw.get("hazards", [])
    ]
    table = raw["table"]
    return TrackConfig(
        name=raw["name"],
        laps=raw["laps"],
        table_size=_vec3(table["size"]),
        table_pos=_vec3(table["pos"]),
        table_color=_vec4(table["color"]),
        marble_tuning=marble_tuning,
        start_positions=[_vec3(pos) for pos in raw["start_positions"]],
        pieces=pieces,
        rails=rails,
        checkpoints=checkpoints,
        hazards=hazards,
    )


def default_track_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "default_track.json"
