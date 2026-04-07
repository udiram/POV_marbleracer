from __future__ import annotations

import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from marbleracer.levels import level_to_config, load_level
from marbleracer.physics import (
    MarbleRampSimulation,
    SimulationConfig,
    build_ramp_segments,
    boost_pad_center_position,
    lane_center_offsets,
    lane_surface_width,
    obstacle_center_position,
    ramp_normal,
    ramp_side,
    ramp_surface_point,
    ramp_tangent,
    required_static_friction,
)


@dataclass(frozen=True)
class Telemetry:
    time: np.ndarray
    distance: np.ndarray
    speed: np.ndarray
    lateral_speed: np.ndarray
    impact: np.ndarray
    boost_active: np.ndarray
    airborne: np.ndarray
    rail_contacts: np.ndarray


PALETTE = {
    "paper": "#f6f1e8",
    "panel": "#fffdf9",
    "panel_alt": "#f1ece3",
    "ink": "#1c2228",
    "muted": "#6d7782",
    "grid": "#d9d2c7",
    "grid_soft": "#ebe4d9",
    "track": "#1b6b84",
    "track_soft": "#a9d2db",
    "bank_pos": "#1b9e77",
    "bank_neg": "#d95f02",
    "boost": "#efb947",
    "obstacle": "#c44536",
    "accent": "#314c73",
    "accent_soft": "#b9cde3",
    "good": "#2a9d8f",
    "warn": "#e76f51",
    "shadow": "#d8d0c3",
}


def svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        "<defs>"
        '<filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">'
        '<feDropShadow dx="0" dy="10" stdDeviation="12" flood-color="#d6cebf" flood-opacity="0.35"/>'
        "</filter>"
        '<linearGradient id="heroWash" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#f8f4ec"/>'
        '<stop offset="100%" stop-color="#efe7db"/>'
        "</linearGradient>"
        '<linearGradient id="trackFade" x1="0%" y1="0%" x2="100%" y2="0%">'
        f'<stop offset="0%" stop-color="{PALETTE["track_soft"]}" stop-opacity="0.35"/>'
        f'<stop offset="100%" stop-color="{PALETTE["track"]}" stop-opacity="0.20"/>'
        "</linearGradient>"
        "</defs>"
        '<rect width="100%" height="100%" fill="url(#heroWash)"/>'
        f'<circle cx="{width - 130:.0f}" cy="120" r="160" fill="#efe5d4" opacity="0.55"/>'
        f'<circle cx="130" cy="{height - 110:.0f}" r="200" fill="#f1e7d8" opacity="0.45"/>'
    )


def svg_footer() -> str:
    return "</svg>\n"


def polyline(points: Iterable[tuple[float, float]], *, stroke: str, stroke_width: float, fill: str = "none", dash: str | None = None, opacity: float | None = None) -> str:
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    attrs = [
        f'points="{pts}"',
        f'stroke="{stroke}"',
        f'stroke-width="{stroke_width}"',
        f'fill="{fill}"',
        'stroke-linecap="round"',
        'stroke-linejoin="round"',
    ]
    if dash:
        attrs.append(f'stroke-dasharray="{dash}"')
    if opacity is not None:
        attrs.append(f'opacity="{opacity}"')
    return f"<polyline {' '.join(attrs)}/>"


def line(x1: float, y1: float, x2: float, y2: float, *, stroke: str, stroke_width: float, dash: str | None = None, opacity: float | None = None) -> str:
    attrs = [
        f'x1="{x1:.2f}"',
        f'y1="{y1:.2f}"',
        f'x2="{x2:.2f}"',
        f'y2="{y2:.2f}"',
        f'stroke="{stroke}"',
        f'stroke-width="{stroke_width}"',
        'stroke-linecap="round"',
    ]
    if dash:
        attrs.append(f'stroke-dasharray="{dash}"')
    if opacity is not None:
        attrs.append(f'opacity="{opacity}"')
    return f"<line {' '.join(attrs)}/>"


def circle(cx: float, cy: float, r: float, *, fill: str, stroke: str = "none", stroke_width: float = 0.0, opacity: float | None = None) -> str:
    attrs = [
        f'cx="{cx:.2f}"',
        f'cy="{cy:.2f}"',
        f'r="{r:.2f}"',
        f'fill="{fill}"',
        f'stroke="{stroke}"',
        f'stroke-width="{stroke_width}"',
    ]
    if opacity is not None:
        attrs.append(f'opacity="{opacity}"')
    return f"<circle {' '.join(attrs)}/>"


def rect(x: float, y: float, width: float, height: float, *, fill: str, stroke: str = "none", stroke_width: float = 0.0, rx: float = 0.0, opacity: float | None = None) -> str:
    attrs = [
        f'x="{x:.2f}"',
        f'y="{y:.2f}"',
        f'width="{width:.2f}"',
        f'height="{height:.2f}"',
        f'fill="{fill}"',
        f'stroke="{stroke}"',
        f'stroke-width="{stroke_width}"',
    ]
    if rx > 0:
        attrs.append(f'rx="{rx:.2f}"')
    if opacity is not None:
        attrs.append(f'opacity="{opacity}"')
    return f"<rect {' '.join(attrs)}/>"


def text(x: float, y: float, content: str, *, size: int = 16, fill: str | None = None, weight: str = "400", anchor: str = "start") -> str:
    fill = fill or PALETTE["ink"]
    safe = (
        content.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" fill="{fill}" font-size="{size}" '
        f'font-family="Avenir Next, Avenir, Helvetica, Arial, sans-serif" '
        f'font-weight="{weight}" text-anchor="{anchor}">{safe}</text>'
    )


def rounded_panel(x: float, y: float, width: float, height: float, *, fill: str | None = None) -> str:
    return rect(
        x,
        y,
        width,
        height,
        fill=fill or PALETTE["panel"],
        stroke=PALETTE["grid"],
        stroke_width=1.5,
        rx=24,
    ).replace("/>", ' filter="url(#softShadow)"/>')


def section_header(x: float, y: float, title: str, subtitle: str | None = None) -> list[str]:
    out = [text(x, y, title, size=22, weight="650")]
    if subtitle:
        out.append(text(x, y + 24, subtitle, size=13, fill=PALETTE["muted"]))
    return out


def add_grid(svg: list[str], *, x: float, y: float, w: float, h: float, rows: int = 4, cols: int = 5) -> None:
    for i in range(rows + 1):
        yy = y + h * i / rows
        svg.append(line(x, yy, x + w, yy, stroke=PALETTE["grid_soft"], stroke_width=1.0))
    for i in range(cols + 1):
        xx = x + w * i / cols
        svg.append(line(xx, y, xx, y + h, stroke=PALETTE["grid_soft"], stroke_width=1.0, opacity=0.8))


def scale_series(values: np.ndarray, *, x: float, y: float, w: float, h: float, x_values: np.ndarray | None = None, y_min: float | None = None, y_max: float | None = None) -> list[tuple[float, float]]:
    if len(values) == 0:
        return []
    xv = np.linspace(0.0, 1.0, len(values)) if x_values is None else (x_values - x_values.min()) / max(x_values.max() - x_values.min(), 1e-9)
    y_lo = float(np.min(values) if y_min is None else y_min)
    y_hi = float(np.max(values) if y_max is None else y_max)
    span = max(y_hi - y_lo, 1e-9)
    return [
        (x + float(tx) * w, y + h - ((float(v) - y_lo) / span) * h)
        for tx, v in zip(xv, values)
    ]


def polygon(points: Iterable[tuple[float, float]], *, fill: str, stroke: str = "none", stroke_width: float = 0.0, opacity: float | None = None) -> str:
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    attrs = [
        f'points="{pts}"',
        f'fill="{fill}"',
        f'stroke="{stroke}"',
        f'stroke-width="{stroke_width}"',
        'stroke-linejoin="round"',
    ]
    if opacity is not None:
        attrs.append(f'opacity="{opacity}"')
    return f"<polygon {' '.join(attrs)}/>"


def add_axis_labels(svg: list[str], *, x: float, y: float, w: float, h: float, x_ticks: list[tuple[float, str]] | None = None, y_ticks: list[tuple[float, str]] | None = None) -> None:
    if x_ticks:
        for frac, label in x_ticks:
            xx = x + w * frac
            svg.append(line(xx, y + h, xx, y + h + 7, stroke=PALETTE["muted"], stroke_width=1.2))
            svg.append(text(xx, y + h + 24, label, size=12, fill=PALETTE["muted"], anchor="middle"))
    if y_ticks:
        for frac, label in y_ticks:
            yy = y + h - h * frac
            svg.append(line(x - 7, yy, x, yy, stroke=PALETTE["muted"], stroke_width=1.2))
            svg.append(text(x - 12, yy + 4, label, size=12, fill=PALETTE["muted"], anchor="end"))


def wrapped_text(svg: list[str], *, x: float, y: float, width_px: float, content: str, size: int = 14, fill: str | None = None, line_gap: int = 20) -> None:
    fill = fill or PALETTE["ink"]
    words = content.split()
    lines: list[str] = []
    current = ""
    max_chars = max(12, int(width_px / (size * 0.58)))
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    for i, line_text in enumerate(lines):
        svg.append(text(x, y + i * line_gap, line_text, size=size, fill=fill))


def path_bounds(points: np.ndarray) -> tuple[float, float, float, float]:
    return float(points[:, 0].min()), float(points[:, 0].max()), float(points[:, 1].min()), float(points[:, 1].max())


def project(points: np.ndarray, *, width: int, height: int, margin: int) -> np.ndarray:
    min_x, max_x, min_y, max_y = path_bounds(points)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    scale = min((width - 2 * margin) / span_x, (height - 2 * margin) / span_y)
    centered = points.copy()
    centered[:, 0] = (points[:, 0] - min_x) * scale + margin
    centered[:, 1] = (max_y - points[:, 1]) * scale + margin
    return centered


def project_to_box(
    points: np.ndarray,
    *,
    box_x: float,
    box_y: float,
    box_w: float,
    box_h: float,
    pad: float,
    bounds: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    min_x, max_x, min_y, max_y = bounds or path_bounds(points)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    scale = min((box_w - 2 * pad) / span_x, (box_h - 2 * pad) / span_y)
    out = points.copy().astype(float)
    used_w = span_x * scale
    used_h = span_y * scale
    x_offset = box_x + pad + (box_w - 2 * pad - used_w) * 0.5
    y_offset = box_y + pad + (box_h - 2 * pad - used_h) * 0.5
    out[:, 0] = (points[:, 0] - min_x) * scale + x_offset
    out[:, 1] = (max_y - points[:, 1]) * scale + y_offset
    return out


def sample_track(config: SimulationConfig, count: int = 480) -> dict[str, np.ndarray]:
    distances = np.linspace(0.0, config.length, count)
    xyz = np.array([[p.x, p.y, p.z] for p in (ramp_surface_point(config, d) for d in distances)], dtype=float)
    normals = np.array([[n.x, n.y, n.z] for n in (ramp_normal(config, d) for d in distances)], dtype=float)
    sides = np.array([[s.x, s.y, s.z] for s in (ramp_side(config, d) for d in distances)], dtype=float)
    tangents = np.array([[t.x, t.y, t.z] for t in (ramp_tangent(config, d) for d in distances)], dtype=float)
    segments = build_ramp_segments(config)
    bank_deg = np.array([segments[min(int((len(segments) - 1) * i / max(1, count - 1)), len(segments) - 1)].bank_deg for i in range(count)], dtype=float)
    heading_deg = np.array([segments[min(int((len(segments) - 1) * i / max(1, count - 1)), len(segments) - 1)].heading_deg for i in range(count)], dtype=float)
    return {
        "distance": distances,
        "xyz": xyz,
        "normals": normals,
        "sides": sides,
        "tangents": tangents,
        "bank_deg": bank_deg,
        "heading_deg": heading_deg,
    }


def run_baseline_rollout(config: SimulationConfig, *, duration: float = 12.0, dt: float = 1.0 / 120.0) -> Telemetry:
    sim = MarbleRampSimulation(config)
    sim.settle_start_contact(settle_steps=12)
    sim.add_forward_speed(5.0)

    steps = int(duration / dt)
    t: list[float] = []
    distance: list[float] = []
    speed: list[float] = []
    lateral_speed: list[float] = []
    impact: list[float] = []
    boost_active: list[float] = []
    airborne: list[float] = []
    rail_contacts: list[float] = []

    still_counter = 0
    max_distance = 0.0
    for _ in range(steps):
        snap = sim.step(dt)
        t.append(float(snap.time))
        distance.append(float(snap.path_distance))
        speed.append(float(snap.speed))
        lateral_speed.append(float(snap.lateral_speed))
        impact.append(float(snap.impact_severity))
        boost_active.append(1.0 if snap.boost_active else 0.0)
        airborne.append(1.0 if snap.airborne else 0.0)
        rail_contacts.append(float(snap.rail_contacts))
        if snap.path_distance > max_distance + 0.05:
            max_distance = snap.path_distance
            still_counter = 0
        else:
            still_counter += 1
        if still_counter > int(1.5 / dt):
            break

    return Telemetry(
        time=np.array(t),
        distance=np.array(distance),
        speed=np.array(speed),
        lateral_speed=np.array(lateral_speed),
        impact=np.array(impact),
        boost_active=np.array(boost_active),
        airborne=np.array(airborne),
        rail_contacts=np.array(rail_contacts),
    )


def write_svg(path: Path, content: str) -> None:
    path.write_text(content)


def maybe_convert_svg(svg_path: Path) -> None:
    converter = shutil.which("rsvg-convert")
    if converter is None:
        return
    pdf_path = svg_path.with_suffix(".pdf")
    subprocess.run([converter, "-f", "pdf", "-o", str(pdf_path), str(svg_path)], check=True)


def generate_track_overview(config: SimulationConfig, output_dir: Path) -> Path:
    width, height = 1360, 920
    sampled = sample_track(config)
    plan_box = (760.0, 132.0, 540.0, 650.0)
    plan = project_to_box(
        sampled["xyz"][:, :2],
        box_x=plan_box[0],
        box_y=plan_box[1],
        box_w=plan_box[2],
        box_h=plan_box[3],
        pad=34.0,
    )
    distances = sampled["distance"]
    z = sampled["xyz"][:, 2]
    bank_deg = sampled["bank_deg"]
    heading_deg = sampled["heading_deg"]

    left = 62
    plot_top = 132
    plot_height = 248
    plot_width = 620
    z_min, z_max = float(z.min()), float(z.max())
    elev_pts = scale_series(z, x=left + 24, y=plot_top + 96, w=plot_width - 48, h=plot_height - 112, x_values=distances, y_min=z_min, y_max=z_max)

    bank_top = 412
    bank_height = 208
    bank_bound = max(1.0, float(np.max(np.abs(bank_deg))))
    bank_pts = scale_series(bank_deg, x=left + 24, y=bank_top + 72, w=plot_width - 48, h=bank_height - 116, x_values=distances, y_min=-bank_bound, y_max=bank_bound)
    heading_bound = max(1.0, float(np.max(np.abs(heading_deg))))
    heading_top = 652
    heading_height = 148
    heading_pts = scale_series(heading_deg, x=left + 24, y=heading_top + 62, w=plot_width - 48, h=heading_height - 92, x_values=distances, y_min=-heading_bound, y_max=heading_bound)

    svg = [svg_header(width, height)]
    svg.extend(section_header(64, 60, "Figure 1. Default track geometry", "Course footprint paired with elevation, banking, and heading extracted directly from the authored level."))
    svg.append(rounded_panel(left, plot_top, plot_width, 668))
    svg.append(rounded_panel(plan_box[0], plan_box[1], plan_box[2], plan_box[3]))

    svg.extend(section_header(left + 26, plot_top + 38, "Longitudinal profiles", "Distance along ramp is used as the shared x-axis for all three plots."))
    add_grid(svg, x=left + 24, y=plot_top + 96, w=plot_width - 48, h=plot_height - 112, rows=3, cols=6)
    elev_fill = [(left + 24, plot_top + plot_height - 16), *elev_pts, (left + plot_width - 24, plot_top + plot_height - 16)]
    svg.append(polygon(elev_fill, fill="url(#trackFade)", opacity=0.9))
    svg.append(polyline(elev_pts, stroke=PALETTE["track"], stroke_width=4.5))
    svg.append(text(left + 24, plot_top + 64, "Elevation", size=17, weight="650"))
    add_axis_labels(
        svg,
        x=left + 24,
        y=plot_top + 96,
        w=plot_width - 48,
        h=plot_height - 112,
        x_ticks=[(0.0, "0 m"), (0.25, f"{config.length*0.25:.0f}"), (0.5, f"{config.length*0.5:.0f}"), (0.75, f"{config.length*0.75:.0f}"), (1.0, f"{config.length:.0f}")],
        y_ticks=[(0.0, f"{z_min:.1f}"), (0.5, f"{(z_min+z_max)/2:.1f}"), (1.0, f"{z_max:.1f}")],
    )
    svg.append(text(left + plot_width - 28, plot_top + 64, "z [m]", size=13, fill=PALETTE["muted"], anchor="end"))

    svg.extend(section_header(left + 26, bank_top + 32, "Signed banking", "Positive values bank the track inward; negative values reverse the lean."))
    add_grid(svg, x=left + 24, y=bank_top + 72, w=plot_width - 48, h=bank_height - 116, rows=4, cols=6)
    zero_y = bank_top + 72 + (bank_height - 116) * 0.5
    svg.append(line(left + 24, zero_y, left + plot_width - 24, zero_y, stroke=PALETTE["muted"], stroke_width=1.3, dash="8 8"))
    svg.append(polyline(bank_pts, stroke=PALETTE["accent"], stroke_width=4.2))
    add_axis_labels(
        svg,
        x=left + 24,
        y=bank_top + 72,
        w=plot_width - 48,
        h=bank_height - 116,
        y_ticks=[(0.0, f"-{bank_bound:.1f}"), (0.5, "0"), (1.0, f"+{bank_bound:.1f}")],
    )
    svg.append(text(left + plot_width - 28, bank_top + 32, "bank [deg]", size=13, fill=PALETTE["muted"], anchor="end"))

    svg.extend(section_header(left + 26, heading_top + 30, "Plan heading", "Heading remains mostly shallow until the late sweep into the finish sector."))
    add_grid(svg, x=left + 24, y=heading_top + 62, w=plot_width - 48, h=heading_height - 92, rows=3, cols=6)
    svg.append(line(left + 24, heading_top + 62 + (heading_height - 92) * 0.5, left + plot_width - 24, heading_top + 62 + (heading_height - 92) * 0.5, stroke=PALETTE["muted"], stroke_width=1.2, dash="8 8"))
    svg.append(polyline(heading_pts, stroke=PALETTE["obstacle"], stroke_width=3.8))
    add_axis_labels(
        svg,
        x=left + 24,
        y=heading_top + 62,
        w=plot_width - 48,
        h=heading_height - 92,
        x_ticks=[(0.0, "0"), (0.5, f"{config.length*0.5:.0f}"), (1.0, f"{config.length:.0f} m")],
        y_ticks=[(0.0, f"-{heading_bound:.0f}"), (0.5, "0"), (1.0, f"+{heading_bound:.0f}")],
    )
    svg.append(text(left + plot_width - 28, heading_top + 30, "heading [deg]", size=13, fill=PALETTE["muted"], anchor="end"))

    svg.extend(section_header(plan_box[0] + 28, plan_box[1] + 40, "Plan view", "Point color encodes local bank angle; start and finish are explicitly marked."))
    svg.append(polyline([tuple(pt) for pt in plan], stroke=PALETTE["track_soft"], stroke_width=26, opacity=0.55))
    svg.append(polyline([tuple(pt) for pt in plan], stroke=PALETTE["track"], stroke_width=6.5, opacity=0.9))

    stride = max(1, len(plan) // 55)
    for idx in range(0, len(plan), stride):
        ratio = (bank_deg[idx] + bank_bound) / (2.0 * bank_bound)
        color = interpolate_color(PALETTE["bank_neg"], PALETTE["bank_pos"], ratio)
        pt = plan[idx]
        svg.append(circle(pt[0], pt[1], 7.5, fill=color, stroke=PALETTE["paper"], stroke_width=1.2))

    start = plan[0]
    finish = plan[-1]
    svg.append(circle(start[0], start[1], 11, fill=PALETTE["good"], stroke=PALETTE["panel"], stroke_width=3))
    svg.append(circle(finish[0], finish[1], 11, fill=PALETTE["warn"], stroke=PALETTE["panel"], stroke_width=3))
    svg.append(text(start[0] - 8, start[1] - 16, "START", size=12, fill=PALETTE["good"], weight="700"))
    svg.append(text(finish[0] + 18, finish[1] + 5, "FINISH", size=12, fill=PALETTE["warn"], weight="700"))

    legend_x = plan_box[0] + 28
    legend_y = 812
    svg.append(text(legend_x, legend_y, "Bank-angle scale", size=15, weight="650"))
    for i in range(9):
        x = legend_x + 130 + i * 28
        svg.append(rect(x, legend_y - 14, 28, 12, fill=interpolate_color(PALETTE["bank_neg"], PALETTE["bank_pos"], i / 8.0)))
    svg.append(text(legend_x + 120, legend_y + 18, "negative", size=13, fill=PALETTE["bank_neg"]))
    svg.append(text(legend_x + 370, legend_y + 18, "positive", size=13, fill=PALETTE["bank_pos"], anchor="end"))

    info_x = plan_box[0] + 28
    info_y = 842
    stats = [
        f"Track length: {config.length:.1f} m",
        f"Vertical drop: {config.height - config.end_height:.1f} m",
        f"Nominal slope: {config.angle_deg:.1f}°",
        f"Track width: {config.width:.2f} m",
    ]
    for i, item in enumerate(stats):
        x = info_x + (i % 2) * 220
        y = info_y + (i // 2) * 30
        svg.append(text(x, y, item, size=15))

    svg.append(svg_footer())
    path = output_dir / "track_overview.svg"
    write_svg(path, "".join(svg))
    maybe_convert_svg(path)
    return path


def generate_course_features(config: SimulationConfig, output_dir: Path) -> Path:
    width, height = 1360, 920
    sampled = sample_track(config, count=560)
    box_x, box_y, box_w, box_h = 62.0, 132.0, 1236.0, 650.0
    pad = 60.0
    plan_bounds = path_bounds(sampled["xyz"][:, :2])
    shifted_plan = project_to_box(
        sampled["xyz"][:, :2],
        box_x=box_x,
        box_y=box_y,
        box_w=box_w,
        box_h=box_h,
        pad=pad,
        bounds=plan_bounds,
    )

    centerline = [tuple(pt) for pt in shifted_plan]
    outer_left = []
    outer_right = []
    for d, xyz, side in zip(sampled["distance"], sampled["xyz"], sampled["sides"]):
        lane_half = config.width * 0.5
        outer_left.append((xyz[0] + side[0] * lane_half, xyz[1] + side[1] * lane_half))
        outer_right.append((xyz[0] - side[0] * lane_half, xyz[1] - side[1] * lane_half))
    outer_left = project_to_box(
        np.array(outer_left),
        box_x=box_x,
        box_y=box_y,
        box_w=box_w,
        box_h=box_h,
        pad=pad,
        bounds=plan_bounds,
    )
    outer_right = project_to_box(
        np.array(outer_right),
        box_x=box_x,
        box_y=box_y,
        box_w=box_w,
        box_h=box_h,
        pad=pad,
        bounds=plan_bounds,
    )

    svg = [svg_header(width, height)]
    svg.extend(section_header(64, 60, "Figure 2. Gameplay elements on the authored course", "Boost pads, obstacle bodies, and the nominal center line are shown in track coordinates."))
    svg.append(rounded_panel(box_x, box_y, box_w, box_h))
    svg.extend(section_header(box_x + 30, box_y + 42, "Default level footprint", "Outer rails are implied by the pale boundary lines; the dashed centerline is only a guide for the report."))

    svg.append(polyline([tuple(pt) for pt in outer_left], stroke="#c6dde4", stroke_width=18, opacity=0.9))
    svg.append(polyline([tuple(pt) for pt in outer_right], stroke="#c6dde4", stroke_width=18, opacity=0.9))
    svg.append(polyline(centerline, stroke=PALETTE["track_soft"], stroke_width=32, opacity=0.7))
    svg.append(polyline(centerline, stroke=PALETTE["track"], stroke_width=5.0, dash="9 12"))

    min_x, max_x, min_y, max_y = plan_bounds
    scale = min((box_w - 2 * pad) / max(max_x - min_x, 1e-6), (box_h - 2 * pad) / max(max_y - min_y, 1e-6))
    used_w = (max_x - min_x) * scale
    used_h = (max_y - min_y) * scale
    x_offset = box_x + pad + (box_w - 2 * pad - used_w) * 0.5
    y_offset = box_y + pad + (box_h - 2 * pad - used_h) * 0.5

    for i, boost in enumerate(config.boost_pads):
        pos = boost_pad_center_position(config, boost)
        px = x_offset + (pos.x - min_x) * scale
        py = y_offset + (max_y - pos.y) * scale
        radius = max(10.0, boost.width * scale * 0.28)
        svg.append(circle(px, py, radius + 4, fill=PALETTE["boost"], opacity=0.18))
        svg.append(circle(px, py, radius, fill=PALETTE["boost"], stroke=PALETTE["ink"], stroke_width=1.6))
        svg.append(text(px, py + 5, f"B{i+1}", size=13, fill=PALETTE["ink"], weight="700", anchor="middle"))

    for i, obstacle in enumerate(config.obstacles):
        pos = obstacle_center_position(config, obstacle)
        px = x_offset + (pos.x - min_x) * scale
        py = y_offset + (max_y - pos.y) * scale
        w = max(18.0, obstacle.width * scale * 0.75)
        h = max(12.0, obstacle.length * scale * 0.32)
        svg.append(rect(px - w / 2, py - h / 2, w, h, fill=PALETTE["obstacle"], stroke=PALETTE["panel"], stroke_width=1.8, rx=5))
        svg.append(rect(px - w / 2, py - h / 2, w, max(3.0, h * 0.28), fill="#da6a5c", rx=5))
        svg.append(text(px + 13, py - 12, f"O{i+1}", size=11, fill=PALETTE["obstacle"], weight="700"))

    legend_x = 92
    legend_y = 844
    svg.append(circle(legend_x, legend_y - 6, 10, fill=PALETTE["boost"], stroke=PALETTE["ink"], stroke_width=1.4))
    svg.append(text(legend_x + 22, legend_y, "Boost pads", size=15))
    svg.append(rect(legend_x + 170, legend_y - 16, 28, 18, fill=PALETTE["obstacle"], stroke=PALETTE["paper"], stroke_width=1.2, rx=4))
    svg.append(text(legend_x + 208, legend_y, "Obstacles", size=15))
    svg.append(line(legend_x + 340, legend_y - 7, legend_x + 388, legend_y - 7, stroke=PALETTE["track"], stroke_width=3.5, dash="10 12"))
    svg.append(text(legend_x + 400, legend_y, "Center racing line", size=15))

    note_x = 1250
    note_y = 844
    svg.append(text(note_x, note_y - 22, f"{len(config.boost_pads)} authored boosts", size=15, fill=PALETTE["muted"], anchor="end"))
    svg.append(text(note_x, note_y, f"{len(config.obstacles)} authored obstacles", size=15, fill=PALETTE["muted"], anchor="end"))
    svg.append(text(note_x, note_y + 22, f"{len(build_ramp_segments(config))} Bullet ramp segments", size=15, fill=PALETTE["muted"], anchor="end"))

    svg.append(svg_footer())
    path = output_dir / "course_features.svg"
    write_svg(path, "".join(svg))
    maybe_convert_svg(path)
    return path


def generate_physics_summary(config: SimulationConfig, output_dir: Path) -> Path:
    width, height = 1320, 900
    required_mu = required_static_friction(config)
    values = [
        ("Static friction threshold", required_mu, PALETTE["accent"]),
        ("Configured static friction", config.static_friction_coeff, PALETTE["good"]),
        ("Ramp friction", config.ramp_friction, PALETTE["track"]),
        ("Marble friction", config.marble_friction, "#5c7cfa"),
        ("Rail friction", config.rail_friction, "#8d99ae"),
        ("Obstacle friction", config.obstacle_friction, PALETTE["warn"]),
        ("Restitution", config.restitution, "#b56576"),
        ("Brake drag", config.brake_drag, "#6d597a"),
        ("Steering accel", config.steering_acceleration, "#355070"),
    ]
    max_val = max(v for _, v, _ in values) * 1.08

    svg = [svg_header(width, height)]
    svg.extend(section_header(64, 60, "Figure 3. Physics tuning summary", "The project combines physically motivated limits with deliberately game-tuned response values."))

    chart_x = 62
    chart_y = 132
    chart_w = 820
    row_h = 58
    svg.append(rounded_panel(chart_x, chart_y, chart_w, 676))
    svg.extend(section_header(chart_x + 26, chart_y + 40, "SimulationConfig comparison", "Bars share a common scale so the deliberately tiny restitution remains visually obvious."))
    for i, (label, value, color) in enumerate(values):
        y = chart_y + 86 + i * row_h
        svg.append(text(chart_x + 26, y + 21, label, size=15))
        svg.append(rect(chart_x + 296, y + 4, chart_w - 336, 24, fill=PALETTE["panel_alt"], rx=12))
        bar_w = (value / max_val) * (chart_w - 280)
        svg.append(rect(chart_x + 296, y + 4, bar_w, 24, fill=color, rx=12))
        svg.append(text(chart_x + chart_w - 26, y + 22, f"{value:.3f}" if value < 1 else f"{value:.2f}", size=15, anchor="end"))

    box_x = 920
    box_y = 132
    svg.append(rounded_panel(box_x, box_y, 336, 280))
    svg.extend(section_header(box_x + 24, box_y + 42, "Pure-rolling check", "The static friction threshold is computed from the incline angle."))
    svg.append(text(box_x + 24, box_y + 112, "μs,req = (2/7) tan(θ)", size=26, weight="650"))
    svg.append(text(box_x + 24, box_y + 154, f"θ = {config.angle_deg:.2f}°", size=17))
    svg.append(text(box_x + 24, box_y + 182, f"μs,req = {required_mu:.3f}", size=17))
    svg.append(text(box_x + 24, box_y + 210, f"configured μs = {config.static_friction_coeff:.3f}", size=17))
    verdict = "Rolling supported" if config.static_friction_coeff >= required_mu else "Would slip"
    verdict_color = PALETTE["good"] if config.static_friction_coeff >= required_mu else PALETTE["warn"]
    svg.append(rect(box_x + 22, box_y + 232, 178, 30, fill=verdict_color, rx=15, opacity=0.16))
    svg.append(text(box_x + 40, box_y + 252, verdict, size=16, fill=verdict_color, weight="700"))

    box2_y = 444
    svg.append(rounded_panel(box_x, box2_y, 336, 364))
    svg.append(text(box_x + 24, box2_y + 42, "What matters in play", size=22, weight="650"))
    wrapped_text(
        svg,
        x=box_x + 24,
        y=box2_y + 66,
        width_px=280,
        content="These choices are consistent with the report's simulation-versus-arcade framing.",
        size=13,
        fill=PALETTE["muted"],
        line_gap=16,
    )
    bullets = [
        "Low restitution keeps impacts from feeling pinball-like.",
        "Brake drag is exponential rather than a hard velocity clamp.",
        "Steering is a lateral force, not a yaw rotation hack.",
        "Ramp and marble friction exceed the pure-rolling threshold.",
    ]
    for i, item in enumerate(bullets):
        yy = box2_y + 118 + i * 62
        svg.append(circle(box_x + 30, yy - 5, 4.5, fill=PALETTE["accent"]))
        wrapped_text(svg, x=box_x + 44, y=yy, width_px=250, content=item, size=14, fill=PALETTE["ink"], line_gap=18)

    svg.append(svg_footer())
    path = output_dir / "physics_summary.svg"
    write_svg(path, "".join(svg))
    maybe_convert_svg(path)
    return path


def generate_rollout_telemetry(config: SimulationConfig, output_dir: Path) -> Path:
    telemetry = run_baseline_rollout(config)
    width, height = 1360, 980
    svg = [svg_header(width, height)]
    svg.extend(section_header(64, 60, "Figure 4. Headless rollout telemetry", "A quick scripted rollout used for diagnostics, not as a claim of optimal play."))

    left = 62
    top = 132
    plot_w = 1236
    plot_h = 170
    panel_h = 192
    panels = [
        ("Path distance (m)", telemetry.distance, PALETTE["track"]),
        ("Speed (m/s)", telemetry.speed, PALETTE["accent"]),
        ("Lateral speed (m/s)", telemetry.lateral_speed, PALETTE["warn"]),
        ("Impact severity", telemetry.impact, PALETTE["obstacle"]),
    ]

    t_max = float(telemetry.time[-1]) if len(telemetry.time) else 1.0
    for panel_index, (label, series, color) in enumerate(panels):
        y_panel = top + panel_index * panel_h
        y0 = y_panel + 18
        max_val = max(float(np.max(series)), 1e-6)
        svg.append(rounded_panel(left, y_panel, plot_w, panel_h - 14))
        svg.append(text(left + 22, y0 + 20, label, size=18, weight="650"))
        add_grid(svg, x=left + 22, y=y0 + 38, w=plot_w - 44, h=plot_h - 34, rows=3, cols=6)
        pts = scale_series(series, x=left + 22, y=y0 + 38, w=plot_w - 44, h=plot_h - 34, x_values=telemetry.time, y_min=0.0, y_max=max_val)
        area = [(left + 22, y0 + plot_h + 4), *pts, (left + plot_w - 22, y0 + plot_h + 4)]
        svg.append(polygon(area, fill=color, opacity=0.10))
        svg.append(polyline(pts, stroke=color, stroke_width=3.6))
        svg.append(text(left + plot_w - 22, y0 + 20, f"max {max_val:.2f}", size=14, fill=PALETTE["muted"], anchor="end"))
        add_axis_labels(
            svg,
            x=left + 22,
            y=y0 + 38,
            w=plot_w - 44,
            h=plot_h - 34,
            x_ticks=[(0.0, "0 s"), (0.5, f"{t_max/2:.1f}"), (1.0, f"{t_max:.1f}")],
            y_ticks=[(0.0, "0"), (0.5, f"{max_val/2:.1f}"), (1.0, f"{max_val:.1f}")],
        )

        active_regions = boolean_regions(telemetry.time, telemetry.boost_active if panel_index == 1 else telemetry.airborne if panel_index == 2 else telemetry.impact > 0.45 if panel_index == 3 else telemetry.rail_contacts > 0)
        for start, end in active_regions:
            x = left + 22 + (start / t_max) * (plot_w - 44)
            w = max(2.0, (end - start) / t_max * (plot_w - 44))
            fill = PALETTE["boost"] if panel_index == 1 else PALETTE["accent_soft"] if panel_index == 2 else "#f4a261" if panel_index == 3 else "#cbd5e1"
            svg.append(rect(x, y0 + 38, w, plot_h - 34, fill=fill, opacity=0.12))

    summary_x = 64
    summary_y = 932
    svg.append(rect(summary_x, summary_y - 26, 1236, 42, fill=PALETTE["panel"], stroke=PALETTE["grid"], stroke_width=1.2, rx=20))
    svg.append(text(summary_x + 24, summary_y, f"Rollout duration: {telemetry.time[-1]:.2f} s", size=15, fill=PALETTE["muted"]))
    svg.append(text(summary_x + 340, summary_y, f"Furthest progress: {telemetry.distance.max():.2f} m", size=15, fill=PALETTE["muted"]))
    svg.append(text(summary_x + 690, summary_y, f"Peak speed: {telemetry.speed.max():.2f} m/s", size=15, fill=PALETTE["muted"]))
    svg.append(text(summary_x + 980, summary_y, f"Peak impact: {telemetry.impact.max():.2f}", size=15, fill=PALETTE["muted"]))

    svg.append(svg_footer())
    path = output_dir / "rollout_telemetry.svg"
    write_svg(path, "".join(svg))
    maybe_convert_svg(path)
    return path


def boolean_regions(time: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    if len(time) == 0:
        return []
    mask_bool = np.asarray(mask, dtype=bool)
    regions: list[tuple[float, float]] = []
    start = None
    for idx, active in enumerate(mask_bool):
        if active and start is None:
            start = float(time[idx])
        if not active and start is not None:
            regions.append((start, float(time[idx])))
            start = None
    if start is not None:
        regions.append((start, float(time[-1])))
    return regions


def interpolate_color(hex_a: str, hex_b: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    a = tuple(int(hex_a[i : i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(hex_b[i : i + 2], 16) for i in (1, 3, 5))
    rgb = tuple(round(x * (1.0 - t) + y * t) for x, y in zip(a, b))
    return "#" + "".join(f"{c:02x}" for c in rgb)


def write_caption_file(output_dir: Path) -> None:
    caption_text = """track_overview.pdf: Overall course geometry with plan, elevation, and banking profile.
course_features.pdf: Placement of authored boost pads and obstacles on the default track.
physics_summary.pdf: Key SimulationConfig values and the pure-rolling friction threshold check.
rollout_telemetry.pdf: Headless baseline rollout showing distance, speed, lateral motion, and impact severity.
asset_sheet.pdf: Offscreen renders of the main report assets using the actual Panda3D scene geometry.
asset_track.png: Full in-engine render of the authored track.
asset_marble.png: Player marble render.
asset_ghost_marble.png: Ghost replay marble render.
asset_bot_marble.png: Bot marble render.
asset_boost_pad.png: Boost pad render on a short ramp slice.
asset_rail.png: Rail render on a short ramp slice.
asset_block_obstacle.png: Static block obstacle render.
asset_pendulum_obstacle.png: Pendulum obstacle render.
asset_sweeper_obstacle.png: Sweeper obstacle render.
asset_start_pad.png: Start pad render.
asset_finish_gate.png: Finish gate render.
"""
    (output_dir / "captions.txt").write_text(caption_text)


def generate_asset_renders(config: SimulationConfig, output_dir: Path) -> list[Path]:
    from panda3d.core import (
        AmbientLight,
        CardMaker,
        DirectionalLight,
        LineSegs,
        NodePath,
        PNMImage,
        PointLight,
        Vec3,
        Vec4,
        loadPrcFileData,
    )

    loadPrcFileData("", "window-type offscreen")
    loadPrcFileData("", "audio-library-name null")
    loadPrcFileData("", "win-size 1800 1320")

    from marbleracer.app import MarbleRampApp

    def direct_children(root: NodePath) -> list[NodePath]:
        return [root.getChild(i) for i in range(root.getNumChildren())]

    def matching_children(root: NodePath, *prefixes: str) -> list[NodePath]:
        out: list[NodePath] = []
        for child in direct_children(root):
            name = child.getName()
            if any(name == prefix or name.startswith(prefix) for prefix in prefixes):
                out.append(child)
        return out

    def nearby_track_context(root: NodePath, *, distance_x: float, radius: float = 10.0) -> list[NodePath]:
        context: list[NodePath] = []
        for child in direct_children(root):
            name = child.getName()
            if not (name.startswith("ramp-") or name.startswith("track-skin-") or name.startswith("rail-")):
                continue
            pos = child.getPos()
            if abs(pos.x - distance_x) <= radius:
                context.append(child)
        return context

    def unique_nodes(nodes: list[NodePath]) -> list[NodePath]:
        unique: list[NodePath] = []
        seen: set[int] = set()
        for node in nodes:
            key = int(node.this)
            if key in seen:
                continue
            seen.add(key)
            unique.append(node)
        return unique

    def crop_png_to_content(path: Path, *, margin: int = 44, threshold: float = 0.035) -> None:
        image = PNMImage()
        if not image.read(str(path)):
            return
        width = image.getXSize()
        height = image.getYSize()
        if width <= 0 or height <= 0:
            return
        bg = (image.getRed(0, 0), image.getGreen(0, 0), image.getBlue(0, 0))
        min_x = width
        min_y = height
        max_x = -1
        max_y = -1
        for y in range(height):
            for x in range(width):
                diff = (
                    abs(image.getRed(x, y) - bg[0])
                    + abs(image.getGreen(x, y) - bg[1])
                    + abs(image.getBlue(x, y) - bg[2])
                )
                if diff > threshold:
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)
        if max_x < min_x or max_y < min_y:
            return
        min_x = max(0, min_x - margin)
        min_y = max(0, min_y - margin)
        max_x = min(width - 1, max_x + margin)
        max_y = min(height - 1, max_y + margin)
        cropped = PNMImage(max_x - min_x + 1, max_y - min_y + 1, 4)
        cropped.copySubImage(image, 0, 0, min_x, min_y, cropped.getXSize(), cropped.getYSize())
        cropped.write(str(path))

    def render_asset(
        app: MarbleRampApp,
        nodes: list[NodePath],
        out_path: Path,
        *,
        direction: tuple[float, float, float],
        fov: float = 32.0,
        pad: float = 1.25,
        add_floor: bool = True,
        floor_scale: float = 1.2,
        fit_mode: str = "balanced",
        color_boost: tuple[float, float, float] = (1.0, 1.0, 1.0),
        focus_nodes: list[NodePath] | None = None,
        focus_color_boost: tuple[float, float, float] = (1.0, 1.0, 1.0),
        context_color_boost: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Path:
        app.scene_root.hide()
        studio = app.render.attachNewNode("asset-studio")
        studio.setLightOff(1)
        focus_ids = {int(node.this) for node in (focus_nodes or [])}
        copied: list[NodePath] = []
        for node in unique_nodes(nodes):
            copied_node = node.copyTo(studio)
            boost = focus_color_boost if int(node.this) in focus_ids else context_color_boost
            copied_node.setColorScale(
                color_boost[0] * boost[0],
                color_boost[1] * boost[1],
                color_boost[2] * boost[2],
                1.0,
            )
            copied.append(copied_node)

        min_point, max_point = studio.getTightBounds()
        center = (min_point + max_point) * 0.5
        studio.setPos(studio, -center)
        min_point, max_point = studio.getTightBounds()
        extent = max_point - min_point
        dims = sorted((abs(extent.x), abs(extent.y), abs(extent.z)), reverse=True)
        if fit_mode == "full":
            radius = max(extent.length() * 0.48, dims[0] * 0.42, 0.35)
        elif fit_mode == "close":
            radius = max(dims[1] * 1.30 + dims[0] * 0.08, dims[2] * 1.65, 0.35)
        else:
            radius = max(dims[1] * 1.10 + dims[0] * 0.14, dims[2] * 1.45, 0.35)

        if add_floor:
            cm = CardMaker("asset-floor")
            floor_size = max(extent.x, extent.y, 1.6) * floor_scale
            cm.setFrame(-floor_size * 0.5, floor_size * 0.5, -floor_size * 0.5, floor_size * 0.5)
            floor = studio.attachNewNode(cm.generate())
            floor.setPos(0.0, 0.0, min_point.z - 0.08)
            floor.setP(-90.0)
            floor.setColor(0.82, 0.87, 0.93, 1.0)

        backdrop = studio.attachNewNode("asset-backdrop")
        cm = CardMaker("asset-back")
        back_size = max(extent.x, extent.z, 1.8) * 2.6
        cm.setFrame(-back_size * 0.5, back_size * 0.5, -back_size * 0.5, back_size * 0.5)
        back_card = backdrop.attachNewNode(cm.generate())
        back_card.setColor(0.89, 0.93, 0.98, 1.0)

        look = Vec3(0.0, 0.0, extent.z * 0.08)
        view_dir = Vec3(*direction)
        view_dir.normalize()
        camera_distance = radius * pad / max(0.18, math.tan(math.radians(fov * 0.5)))
        backdrop.setPos(look - view_dir * (radius * 1.7))
        backdrop.lookAt(look + view_dir * 2.0)

        ambient = AmbientLight("asset-ambient")
        ambient.setColor(Vec4(0.72, 0.74, 0.78, 1.0))
        ambient_np = studio.attachNewNode(ambient)
        studio.setLight(ambient_np)

        key = DirectionalLight("asset-key")
        key.setColor(Vec4(1.22, 1.18, 1.08, 1.0))
        key_np = studio.attachNewNode(key)
        key_np.setHpr(-38.0, -34.0, 0.0)
        studio.setLight(key_np)

        fill = DirectionalLight("asset-fill")
        fill.setColor(Vec4(0.66, 0.78, 0.96, 1.0))
        fill_np = studio.attachNewNode(fill)
        fill_np.setHpr(132.0, -18.0, 0.0)
        studio.setLight(fill_np)

        rim = PointLight("asset-rim")
        rim.setColor(Vec4(0.76, 0.88, 1.10, 1.0))
        rim_np = studio.attachNewNode(rim)
        rim_np.setPos(-radius * 1.1, radius * 1.4, radius * 1.3)
        studio.setLight(rim_np)

        app.camLens.setFov(fov)
        app.camera.setPos(look + view_dir * camera_distance)
        app.camera.lookAt(look)
        app.setBackgroundColor(0.79, 0.86, 0.95, 1.0)
        app.render.clearFog()
        for _ in range(3):
            app.graphicsEngine.renderFrame()
        app.win.saveScreenshot(str(out_path))
        crop_png_to_content(out_path)
        studio.removeNode()
        app.scene_root.show()
        return out_path

    def stage_obstacle_pose(obstacle) -> None:
        if obstacle.motion_kind == "pendulum":
            target_angle = math.pi * 0.5
        elif obstacle.motion_kind == "sweeper":
            target_angle = math.pi * 0.2
        else:
            return
        if obstacle.motion_speed <= 0.0:
            return
        sample_time = (target_angle - obstacle.motion_phase) / obstacle.motion_speed
        app._animate_moving_obstacles(max(0.0, sample_time))

    def add_motion_hint(root: NodePath, obstacle) -> NodePath | None:
        hint = root.attachNewNode("motion-hint")
        hint.setLightOff(1)
        hint.setDepthWrite(False)
        hint.setDepthTest(False)
        hint.setBin("fixed", 20)

        line = LineSegs("motion-hint-line")
        line.setThickness(5.0)

        if obstacle.motion_kind == "pendulum":
            pivot_height = obstacle.pivot_height if obstacle.pivot_height > 0.0 else 1.05
            rod_length = max(obstacle.height * 0.5 + 0.16, pivot_height - obstacle.height * 0.5)
            radius = max(rod_length * 0.46, 0.18)
            center = Vec3(0.0, 0.0, -radius * 0.08)
            sweep = max(obstacle.motion_amplitude * 1.35, 0.24)
            samples = 25
            for index in range(samples):
                frac = index / (samples - 1)
                angle = -sweep + (2.0 * sweep * frac)
                point = Vec3(math.sin(angle) * radius, 0.0, -math.cos(angle) * radius) + center
                line.setColor(0.98, 0.98, 1.0, 0.92 if 0 < index < samples - 1 else 0.0)
                if index == 0:
                    line.moveTo(point)
                else:
                    line.drawTo(point)

            arrow = LineSegs("motion-hint-arrow")
            arrow.setThickness(5.0)
            arrow.setColor(0.98, 0.98, 1.0, 0.95)
            tip_angle = sweep
            tip = Vec3(math.sin(tip_angle) * radius, 0.0, -math.cos(tip_angle) * radius) + center
            left = tip + Vec3(-0.06, 0.0, -0.02)
            right = tip + Vec3(-0.02, 0.0, -0.06)
            arrow.moveTo(left)
            arrow.drawTo(tip)
            arrow.drawTo(right)
            arrow_np = hint.attachNewNode(arrow.create())
            arrow_np.setPos(0.0, obstacle.width * 0.95, -rod_length * 0.12)
            arrow_np.setHpr(0.0, 0.0, 0.0)

            line_np = hint.attachNewNode(line.create())
            line_np.setPos(0.0, obstacle.width * 0.95, -rod_length * 0.12)
            return hint

        if obstacle.motion_kind == "sweeper":
            span = max(obstacle.motion_amplitude * 2.2, 0.26)
            z = obstacle.height * 1.15
            start = Vec3(-span, 0.0, z)
            end = Vec3(span, 0.0, z)
            line.setColor(0.98, 0.98, 1.0, 0.92)
            line.moveTo(start)
            line.drawTo(end)
            line_np = hint.attachNewNode(line.create())

            arrow = LineSegs("motion-hint-arrow")
            arrow.setThickness(5.0)
            arrow.setColor(0.98, 0.98, 1.0, 0.95)
            arrow.moveTo(Vec3(span - 0.07, 0.0, z + 0.04))
            arrow.drawTo(end)
            arrow.drawTo(Vec3(span - 0.07, 0.0, z - 0.04))
            arrow.moveTo(Vec3(-span + 0.07, 0.0, z + 0.04))
            arrow.drawTo(start)
            arrow.drawTo(Vec3(-span + 0.07, 0.0, z - 0.04))
            arrow_np = hint.attachNewNode(arrow.create())
            line_np.setPos(0.0, 0.0, 0.0)
            arrow_np.setPos(0.0, 0.0, 0.0)
            return hint

        hint.removeNode()
        return None

    def build_asset_sheet(output_dir: Path, entries: list[tuple[str, str]]) -> Path:
        width, height = 1400, 1280
        margin = 54
        cols = 3
        card_w = 402
        card_h = 324
        start_y = 130
        gap_x = 44
        gap_y = 36
        svg = [svg_header(width, height)]
        svg.extend(section_header(54, 58, "Figure 5. In-engine asset renders", "Each tile is an offscreen Panda3D render captured from the actual scene assets used in the report."))
        for index, (filename, label) in enumerate(entries):
            col = index % cols
            row = index // cols
            x = margin + col * (card_w + gap_x)
            y = start_y + row * (card_h + gap_y)
            svg.append(rounded_panel(x, y, card_w, card_h))
            svg.append(text(x + 20, y + 34, label, size=18, weight="650"))
            svg.append(
                f'<image href="{filename}" x="{x + 18:.2f}" y="{y + 52:.2f}" width="{card_w - 36:.2f}" height="{card_h - 86:.2f}" preserveAspectRatio="xMidYMid meet"/>'
            )
            svg.append(text(x + 20, y + card_h - 18, filename, size=12, fill=PALETTE["muted"]))
        svg.append(svg_footer())
        path = output_dir / "asset_sheet.svg"
        write_svg(path, "".join(svg))
        maybe_convert_svg(path)
        return path

    app = MarbleRampApp(config, menu_disabled=True, start_mode="time_trial")
    app.aspect2d.hide()
    app.render2d.hide()
    if hasattr(app, "pixel2d"):
        app.pixel2d.hide()

    top_nodes = direct_children(app.scene_root)
    obstacle_roots = [node for node in top_nodes if node.getName() == "obstacle"]
    obstacle_pairs = list(zip(app.sim_config.obstacles, obstacle_roots))

    block_root = next(root for obstacle, root in obstacle_pairs if obstacle.kind == "block")
    block_distance = next(obstacle.distance_along_ramp for obstacle, _ in obstacle_pairs if obstacle.kind == "block")
    pendulum_root = next(root for obstacle, root in obstacle_pairs if obstacle.kind == "pendulum")
    pendulum_distance = next(obstacle.distance_along_ramp for obstacle, _ in obstacle_pairs if obstacle.kind == "pendulum")
    sweeper_root = next(root for obstacle, root in obstacle_pairs if obstacle.kind == "sweeper")
    sweeper_distance = next(obstacle.distance_along_ramp for obstacle, _ in obstacle_pairs if obstacle.kind == "sweeper")

    finish_gate = next(node for node in top_nodes if node.getName() == "finish-gate")
    start_pad = next(node for node in top_nodes if node.getName() == "start-pad")
    rail_root = min(matching_children(app.scene_root, "rail-"), key=lambda node: abs(node.getPos().x - app.sim_config.length * 0.24))
    boost_root = app.boost_pads[min(1, len(app.boost_pads) - 1)].root
    boost_distance = app.boost_pads[min(1, len(app.boost_pads) - 1)].distance
    ghost_root = app.ghost_playback.root
    ghost_root.show()

    generated: list[Path] = []
    generated.append(
        render_asset(
            app,
            matching_children(app.scene_root, "ramp-", "track-skin-", "rail-", "boost-pad-", "finish-gate", "start-pad", "obstacle"),
            output_dir / "asset_track.png",
            direction=(0.95, -0.85, 0.72),
            fov=21.0,
            pad=2.8,
            add_floor=False,
            fit_mode="full",
            color_boost=(1.10, 1.12, 1.16),
        )
    )
    generated.append(render_asset(app, [app.marble_root], output_dir / "asset_marble.png", direction=(2.1, -2.2, 1.2), fov=26.0, pad=2.1, fit_mode="close", color_boost=(0.98, 0.99, 1.00)))
    generated.append(render_asset(app, [ghost_root], output_dir / "asset_ghost_marble.png", direction=(2.0, -2.0, 1.1), fov=26.0, pad=2.2, fit_mode="close", color_boost=(0.94, 0.97, 1.02)))
    generated.append(
        render_asset(
            app,
            nearby_track_context(app.scene_root, distance_x=boost_root.getPos().x, radius=3.2) + [boost_root],
            output_dir / "asset_boost_pad.png",
            direction=(1.8, -2.4, 1.1),
            fov=24.0,
            pad=1.55,
            fit_mode="close",
            color_boost=(1.10, 1.12, 1.18),
        )
    )
    generated.append(
        render_asset(
            app,
            nearby_track_context(app.scene_root, distance_x=rail_root.getPos().x, radius=2.6) + [rail_root],
            output_dir / "asset_rail.png",
            direction=(1.6, -2.2, 1.05),
            fov=24.0,
            pad=1.50,
            fit_mode="close",
            color_boost=(1.10, 1.12, 1.16),
        )
    )
    generated.append(
        render_asset(
            app,
            [block_root],
            output_dir / "asset_block_obstacle.png",
            direction=(1.05, -1.25, 0.92),
            fov=22.0,
            pad=1.18,
            fit_mode="close",
            color_boost=(1.10, 1.12, 1.16),
            focus_nodes=[block_root],
            focus_color_boost=(1.60, 1.48, 1.28),
            context_color_boost=(0.76, 0.82, 0.92),
        )
    )
    stage_obstacle_pose(next(obstacle for obstacle, root in obstacle_pairs if root == pendulum_root))
    pendulum_obstacle = next(obstacle for obstacle, root in obstacle_pairs if root == pendulum_root)
    pendulum_hint = add_motion_hint(pendulum_root, pendulum_obstacle)
    generated.append(
        render_asset(
            app,
            [pendulum_root],
            output_dir / "asset_pendulum_obstacle.png",
            direction=(0.84, -1.02, 1.18),
            fov=22.0,
            pad=1.18,
            fit_mode="close",
            color_boost=(1.10, 1.12, 1.16),
            focus_nodes=[pendulum_root],
            focus_color_boost=(1.58, 1.46, 1.24),
            context_color_boost=(0.74, 0.80, 0.92),
        )
    )
    if pendulum_hint is not None and not pendulum_hint.isEmpty():
        pendulum_hint.removeNode()
    sweeper_obstacle = next(obstacle for obstacle, root in obstacle_pairs if root == sweeper_root)
    stage_obstacle_pose(sweeper_obstacle)
    sweeper_hint = add_motion_hint(sweeper_root, sweeper_obstacle)
    generated.append(
        render_asset(
            app,
            [sweeper_root],
            output_dir / "asset_sweeper_obstacle.png",
            direction=(0.96, -1.10, 1.02),
            fov=22.0,
            pad=1.18,
            fit_mode="close",
            color_boost=(1.10, 1.12, 1.16),
            focus_nodes=[sweeper_root],
            focus_color_boost=(1.36, 1.60, 1.48),
            context_color_boost=(0.74, 0.80, 0.90),
        )
    )
    if sweeper_hint is not None and not sweeper_hint.isEmpty():
        sweeper_hint.removeNode()
    generated.append(
        render_asset(
            app,
            nearby_track_context(app.scene_root, distance_x=start_pad.getPos().x, radius=3.0) + [start_pad],
            output_dir / "asset_start_pad.png",
            direction=(1.7, -2.5, 1.1),
            fov=24.0,
            pad=1.55,
            fit_mode="close",
            color_boost=(1.10, 1.12, 1.16),
        )
    )
    generated.append(
        render_asset(
            app,
            nearby_track_context(app.scene_root, distance_x=finish_gate.getPos().x, radius=3.8) + [finish_gate],
            output_dir / "asset_finish_gate.png",
            direction=(1.7, -2.4, 1.15),
            fov=24.0,
            pad=1.60,
            fit_mode="close",
            color_boost=(1.10, 1.12, 1.16),
        )
    )
    app.destroy()

    bot_app = MarbleRampApp(config, menu_disabled=True, start_mode="bot_race")
    bot_app.aspect2d.hide()
    bot_app.render2d.hide()
    if hasattr(bot_app, "pixel2d"):
        bot_app.pixel2d.hide()
    if bot_app.bot_marbles:
        generated.append(
            render_asset(
                bot_app,
                [bot_app.bot_marbles[0].visual_root],
                output_dir / "asset_bot_marble.png",
                direction=(2.0, -2.2, 1.1),
                fov=26.0,
                pad=2.1,
                fit_mode="close",
                color_boost=(0.98, 0.99, 1.00),
            )
        )
    bot_app.destroy()

    sheet_entries = [
        ("asset_track.png", "Track"),
        ("asset_marble.png", "Player Marble"),
        ("asset_ghost_marble.png", "Ghost Marble"),
        ("asset_bot_marble.png", "Bot Marble"),
        ("asset_boost_pad.png", "Boost Pad"),
        ("asset_rail.png", "Rail"),
        ("asset_block_obstacle.png", "Block Obstacle"),
        ("asset_pendulum_obstacle.png", "Pendulum"),
        ("asset_sweeper_obstacle.png", "Sweeper"),
        ("asset_start_pad.png", "Start Pad"),
        ("asset_finish_gate.png", "Finish Gate"),
    ]
    generated.append(build_asset_sheet(output_dir, sheet_entries))
    return generated


def main() -> None:
    output_dir = PROJECT_ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    level = load_level("default")
    config = level_to_config(level)

    generated = [
        generate_track_overview(config, output_dir),
        generate_course_features(config, output_dir),
        generate_physics_summary(config, output_dir),
        generate_rollout_telemetry(config, output_dir),
    ]
    generated.extend(generate_asset_renders(config, output_dir))
    write_caption_file(output_dir)
    print("Generated:")
    for path in generated:
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
