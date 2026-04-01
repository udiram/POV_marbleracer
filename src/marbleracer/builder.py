from __future__ import annotations

import math
import os
import subprocess
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path

from .levels import TrackLevel, level_directory, list_level_paths, load_level, save_level
from .physics import BoostPad, GuideObstacle

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 840
CANVAS_WIDTH = 1240
CANVAS_HEIGHT = 680
PATH_SAMPLES = 32


@dataclass
class ControlPoint:
    x: float
    y: float


class BuilderApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Marble Racer Track Builder")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg="#111827")

        self.level_name = "level-1"
        self.level_description = "Base track."
        self.current_path: Path | None = None
        self.track_width = 2.32
        self.start_height = 39.5
        self.end_height = 8.8
        self.control_points: list[ControlPoint] = []
        self.boost_pads: list[BoostPad] = []
        self.obstacles: list[GuideObstacle] = []
        self.selected_index: int | None = None
        self.dragging = False

        self.level_var = tk.StringVar()
        self.start_var = tk.StringVar()
        self.end_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.metrics_var = tk.StringVar()

        self._build_ui()
        self._bind_events()
        self._apply_level(load_level("default"))

    def _build_ui(self) -> None:
        top = tk.Frame(self.root, bg="#111827")
        top.pack(fill=tk.X, padx=12, pady=(10, 6))

        for label, command in (
            ("New", self.new_level),
            ("Save", self.save_current_level),
            ("Next", self.open_level),
            ("Play", self.play_current_level),
        ):
            tk.Button(
                top,
                text=label,
                command=command,
                bg="#1f2937",
                fg="#e5eef9",
                activebackground="#334155",
                activeforeground="#ffffff",
                relief=tk.FLAT,
                padx=14,
                pady=6,
            ).pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(top, text="Level", bg="#111827", fg="#dbe7f6").pack(side=tk.LEFT, padx=(12, 6))
        tk.Entry(top, textvariable=self.level_var, width=22).pack(side=tk.LEFT)

        tk.Label(top, text="Start", bg="#111827", fg="#dbe7f6").pack(side=tk.LEFT, padx=(16, 6))
        tk.Entry(top, textvariable=self.start_var, width=8).pack(side=tk.LEFT)
        tk.Label(top, text="End", bg="#111827", fg="#dbe7f6").pack(side=tk.LEFT, padx=(12, 6))
        tk.Entry(top, textvariable=self.end_var, width=8).pack(side=tk.LEFT)
        tk.Button(
            top,
            text="Apply Heights",
            command=self._apply_height_inputs,
            bg="#1f2937",
            fg="#e5eef9",
            activebackground="#334155",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=6,
        ).pack(side=tk.LEFT, padx=(10, 0))

        tk.Button(
            top,
            text="Width -",
            command=lambda: self._adjust_width(-0.1),
            bg="#1f2937",
            fg="#e5eef9",
            activebackground="#334155",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=10,
            pady=6,
        ).pack(side=tk.RIGHT, padx=(8, 0))
        tk.Button(
            top,
            text="Width +",
            command=lambda: self._adjust_width(0.1),
            bg="#1f2937",
            fg="#e5eef9",
            activebackground="#334155",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=10,
            pady=6,
        ).pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            bg="#0b1220",
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))

        bottom = tk.Frame(self.root, bg="#111827")
        bottom.pack(fill=tk.X, padx=12, pady=(0, 12))
        tk.Label(bottom, textvariable=self.status_var, bg="#111827", fg="#f8b84b", anchor="w").pack(fill=tk.X)
        tk.Label(bottom, textvariable=self.metrics_var, bg="#111827", fg="#9fb2ca", anchor="w").pack(fill=tk.X, pady=(4, 0))

    def _bind_events(self) -> None:
        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.root.bind("<Delete>", self._delete_selected)
        self.root.bind("<BackSpace>", self._delete_selected)
        self.root.bind("<Control-s>", lambda _event: self.save_current_level())
        self.root.bind("<Control-o>", lambda _event: self.open_level())
        self.root.bind("<Control-n>", lambda _event: self.new_level())

    def run(self) -> None:
        self.root.mainloop()

    def new_level(self) -> None:
        self.level_name = f"level-{len(list_level_paths()) + 1}"
        self.level_description = "Base track."
        self.current_path = None
        self.track_width = 2.32
        self.start_height = 39.5
        self.end_height = 8.8
        self.control_points = [
            ControlPoint(0.0, 0.0),
            ControlPoint(24.0, 4.0),
            ControlPoint(58.0, 13.0),
            ControlPoint(88.0, -6.0),
            ControlPoint(118.0, -14.0),
        ]
        self.boost_pads = []
        self.obstacles = []
        self.selected_index = None
        self.status_var.set("New track. Click to add points. Drag to reshape.")
        self._refresh()

    def open_level(self) -> None:
        paths = list_level_paths()
        if not paths:
            self.status_var.set("No saved levels found.")
            return
        if self.current_path in paths:
            next_index = (paths.index(self.current_path) + 1) % len(paths)
        else:
            next_index = 0
        path = paths[next_index]
        self._apply_level(load_level(path), source_path=path)

    def save_current_level(self) -> None:
        try:
            if self.current_path is None:
                self.current_path = level_directory() / f"{self._slugify(self.level_var.get())}.json"
            saved_path = save_level(self._build_level(), self.current_path)
        except Exception as exc:
            self.status_var.set(f"Save blocked: {exc}")
            return
        self.current_path = saved_path
        self.status_var.set(f"Saved {saved_path.name}")
        self._refresh()

    def play_current_level(self) -> None:
        try:
            path = save_level(
                self._build_level(),
                self.current_path or (level_directory() / f"{self._slugify(self.level_var.get())}.json"),
            )
        except Exception as exc:
            self.status_var.set(f"Play blocked: {exc}")
            return
        self.current_path = path
        env = dict(os.environ)
        src_path = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}:{env['PYTHONPATH']}"
        subprocess.Popen([sys.executable, "-m", "marbleracer.main", "--level", str(path)], env=env)
        self.status_var.set(f"Playing {path.stem}")
        self._refresh()

    def _apply_level(self, level: TrackLevel, *, source_path: Path | None = None) -> None:
        self.level_name = level.name
        self.level_description = level.description
        self.current_path = source_path
        self.boost_pads = list(level.boost_pads)
        self.obstacles = list(level.obstacles)
        editor = level.editor or {}
        stored = editor.get("control_points")
        if isinstance(stored, list) and len(stored) >= 2:
            self.control_points = [
                ControlPoint(float(item.get("x", 0.0)), float(item.get("y", 0.0)))
                for item in stored
                if isinstance(item, dict)
            ]
        else:
            self.control_points = self._control_points_from_track(level)
        self.track_width = float(editor.get("track_width", level.track["width"]))
        self.start_height = float(level.track["height"])
        self.end_height = self._end_height_from_track(level)
        self.selected_index = None
        self.status_var.set(
            "Click empty space to add a point. Drag a point to move it. Delete removes the selected point."
        )
        self._refresh()

    def _control_points_from_track(self, level: TrackLevel) -> list[ControlPoint]:
        headings = tuple(float(value) for value in level.track["segment_headings_deg"])
        if not headings:
            return [ControlPoint(0.0, 0.0), ControlPoint(80.0, 0.0)]
        angle = math.radians(float(level.track["angle_deg"]))
        horizontal_length = float(level.track["length"]) * math.cos(angle)
        step = horizontal_length / max(1, len(headings))
        x = 0.0
        y = 0.0
        points = [ControlPoint(x, y)]
        for heading in headings:
            x += step * math.cos(math.radians(heading))
            y += step * math.sin(math.radians(heading))
            points.append(ControlPoint(x, y))
        stride = max(1, len(points) // 6)
        reduced = points[::stride]
        if reduced[-1] != points[-1]:
            reduced.append(points[-1])
        return reduced

    def _end_height_from_track(self, level: TrackLevel) -> float:
        angle = math.radians(float(level.track["angle_deg"]))
        return float(level.track["height"]) - float(level.track["length"]) * math.sin(angle)

    def _apply_height_inputs(self) -> None:
        try:
            start_height = float(self.start_var.get())
            end_height = float(self.end_var.get())
        except ValueError:
            self.status_var.set("Start and end heights must be numbers.")
            return
        if start_height <= end_height:
            self.status_var.set("Start height must be above end height.")
            return
        self.start_height = start_height
        self.end_height = end_height
        self.status_var.set("Updated heights.")
        self._refresh()

    def _adjust_width(self, delta: float) -> None:
        self.track_width = max(1.2, min(6.0, self.track_width + delta))
        self.status_var.set(f"Track width {self.track_width:.2f}m")
        self._refresh()

    def _on_press(self, event: tk.Event) -> None:
        index = self._nearest_point(float(event.x), float(event.y))
        if index is not None:
            self.selected_index = index
            self.dragging = True
            self.status_var.set(f"Editing point {index + 1}")
            self._refresh()
            return
        x, y = self._canvas_to_world(float(event.x), float(event.y))
        insert_index = self._find_insert_index(x, y)
        self.control_points.insert(insert_index, ControlPoint(x, y))
        self.selected_index = insert_index
        self.dragging = True
        self.status_var.set(f"Added point {insert_index + 1}")
        self._refresh()

    def _on_drag(self, event: tk.Event) -> None:
        if not self.dragging or self.selected_index is None:
            return
        x, y = self._canvas_to_world(float(event.x), float(event.y))
        self.control_points[self.selected_index].x = x
        self.control_points[self.selected_index].y = y
        self._refresh()

    def _on_release(self, _event: tk.Event) -> None:
        self.dragging = False

    def _delete_selected(self, _event: tk.Event | None = None) -> None:
        if self.selected_index is None:
            return
        if len(self.control_points) <= 2:
            self.status_var.set("At least two points are required.")
            return
        self.control_points.pop(self.selected_index)
        self.selected_index = None
        self.status_var.set("Removed point")
        self._refresh()

    def _refresh(self) -> None:
        self.level_var.set(self.level_name)
        self.start_var.set(f"{self.start_height:.1f}")
        self.end_var.set(f"{self.end_height:.1f}")
        path_length = self._path_length()
        drop = self.start_height - self.end_height
        slope = math.degrees(math.atan2(max(drop, 0.0), max(path_length, 1e-6)))
        self.metrics_var.set(
            f"path {path_length:.1f}m   slope {slope:.1f}deg   width {self.track_width:.2f}m"
        )
        self._draw_canvas()

    def _draw_canvas(self) -> None:
        self.canvas.delete("all")
        width = int(self.canvas.winfo_width() or CANVAS_WIDTH)
        height = int(self.canvas.winfo_height() or CANVAS_HEIGHT)
        self._draw_grid(width, height)

        self.canvas.create_text(
            18,
            18,
            anchor="nw",
            fill="#93a7c4",
            font=("Helvetica", 12),
            text="Track line only. Click to add. Drag to move. Delete to remove. Save exports the base track.",
        )

        samples = self._sample_points()
        if len(samples) >= 2:
            flat: list[float] = []
            for x, y in samples:
                px, py = self._world_to_canvas(x, y)
                flat.extend((px, py))
            self.canvas.create_line(*flat, fill="#43c9ff", width=5, smooth=True)

        if self.control_points:
            sx, sy = self._world_to_canvas(self.control_points[0].x, self.control_points[0].y)
            ex, ey = self._world_to_canvas(self.control_points[-1].x, self.control_points[-1].y)
            self.canvas.create_text(sx, sy - 18, text=f"START {self.start_height:.1f}m", fill="#dce7f7")
            self.canvas.create_text(ex, ey - 18, text=f"END {self.end_height:.1f}m", fill="#dce7f7")

        for index, point in enumerate(self.control_points):
            px, py = self._world_to_canvas(point.x, point.y)
            radius = 9 if index == self.selected_index else 7
            color = "#f8b84b" if index == self.selected_index else "#ff7b39"
            self.canvas.create_oval(px - radius, py - radius, px + radius, py + radius, fill=color, outline="")
            self.canvas.create_text(px, py + 18, text=str(index + 1), fill="#dce7f7", font=("Helvetica", 10, "bold"))

    def _draw_grid(self, width: int, height: int) -> None:
        for x in range(0, width + 1, 40):
            self.canvas.create_line(x, 0, x, height, fill="#162033")
        for y in range(0, height + 1, 40):
            self.canvas.create_line(0, y, width, y, fill="#162033")

    def _build_level(self) -> TrackLevel:
        if len(self.control_points) < 2:
            raise ValueError("At least two control points are required.")
        self.level_name = self.level_var.get().strip() or self.level_name
        horizontal_length = self._path_length()
        drop = self.start_height - self.end_height
        if horizontal_length <= 0.1:
            raise ValueError("Track is too short.")
        if drop <= 0.0:
            raise ValueError("Start height must be above end height.")
        samples = self._sample_points()
        headings: list[float] = []
        for index, point in enumerate(samples):
            prev = samples[max(0, index - 1)]
            nxt = samples[min(len(samples) - 1, index + 1)]
            dx = nxt[0] - prev[0]
            dy = nxt[1] - prev[1]
            heading = math.degrees(math.atan2(dy, dx)) if abs(dx) > 1e-6 or abs(dy) > 1e-6 else 0.0
            headings.append(heading)
        angle_deg = math.degrees(math.atan2(drop, horizontal_length))
        length = math.hypot(horizontal_length, drop)
        return TrackLevel(
            name=self.level_name,
            description=self.level_description,
            track={
                "angle_deg": angle_deg,
                "length": length,
                "height": self.start_height,
                "width": self.track_width,
                "segment_headings_deg": tuple(headings),
                "segment_bank_deg": tuple(0.0 for _ in headings),
            },
            boost_pads=tuple(self.boost_pads),
            obstacles=tuple(self.obstacles),
            editor={
                "control_points": [{"x": point.x, "y": point.y, "bank_deg": 0.0} for point in self.control_points],
                "track_width": self.track_width,
                "track_angle_deg": angle_deg,
                "end_clearance": self.end_height,
            },
        )

    def _sample_points(self) -> list[tuple[float, float]]:
        if len(self.control_points) < 2:
            return [(0.0, 0.0)]
        samples: list[tuple[float, float]] = []
        points = self.control_points
        for index in range(len(points) - 1):
            p0 = points[max(0, index - 1)]
            p1 = points[index]
            p2 = points[index + 1]
            p3 = points[min(len(points) - 1, index + 2)]
            for step in range(PATH_SAMPLES):
                t = step / PATH_SAMPLES
                x = self._catmull_rom(p0.x, p1.x, p2.x, p3.x, t)
                y = self._catmull_rom(p0.y, p1.y, p2.y, p3.y, t)
                samples.append((x, y))
        samples.append((points[-1].x, points[-1].y))
        return samples

    def _path_length(self) -> float:
        samples = self._sample_points()
        return sum(
            math.hypot(samples[index][0] - samples[index - 1][0], samples[index][1] - samples[index - 1][1])
            for index in range(1, len(samples))
        )

    def _bounds(self) -> tuple[float, float, float, float]:
        xs = [point.x for point in self.control_points] or [0.0, 120.0]
        ys = [point.y for point in self.control_points] or [0.0, 0.0]
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        margin = max(12.0, max(max_x - min_x, max_y - min_y, 1.0) * 0.15)
        return min_x - margin, max_x + margin, min_y - margin, max_y + margin

    def _world_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        width = float(self.canvas.winfo_width() or CANVAS_WIDTH)
        height = float(self.canvas.winfo_height() or CANVAS_HEIGHT)
        min_x, max_x, min_y, max_y = self._bounds()
        scale_x = (width - 40.0) / max(max_x - min_x, 1.0)
        scale_y = (height - 40.0) / max(max_y - min_y, 1.0)
        scale = min(scale_x, scale_y)
        offset_x = (width - (max_x - min_x) * scale) * 0.5
        offset_y = (height - (max_y - min_y) * scale) * 0.5
        px = offset_x + (x - min_x) * scale
        py = height - (offset_y + (y - min_y) * scale)
        return px, py

    def _canvas_to_world(self, px: float, py: float) -> tuple[float, float]:
        width = float(self.canvas.winfo_width() or CANVAS_WIDTH)
        height = float(self.canvas.winfo_height() or CANVAS_HEIGHT)
        min_x, max_x, min_y, max_y = self._bounds()
        scale_x = (width - 40.0) / max(max_x - min_x, 1.0)
        scale_y = (height - 40.0) / max(max_y - min_y, 1.0)
        scale = min(scale_x, scale_y)
        offset_x = (width - (max_x - min_x) * scale) * 0.5
        offset_y = (height - (max_y - min_y) * scale) * 0.5
        x = min_x + (px - offset_x) / scale
        y = min_y + ((height - py) - offset_y) / scale
        return x, y

    def _nearest_point(self, px: float, py: float) -> int | None:
        if not self.control_points:
            return None
        best_index = None
        best_distance = float("inf")
        for index, point in enumerate(self.control_points):
            cx, cy = self._world_to_canvas(point.x, point.y)
            distance = math.hypot(px - cx, py - cy)
            if distance < best_distance:
                best_distance = distance
                best_index = index
        return best_index if best_distance <= 16.0 else None

    def _find_insert_index(self, x: float, y: float) -> int:
        if len(self.control_points) < 2:
            return len(self.control_points)
        best_index = len(self.control_points)
        best_metric = float("inf")
        for index in range(len(self.control_points) - 1):
            ax = self.control_points[index].x
            ay = self.control_points[index].y
            bx = self.control_points[index + 1].x
            by = self.control_points[index + 1].y
            abx = bx - ax
            aby = by - ay
            denom = abx * abx + aby * aby
            t = 0.0 if denom <= 1e-6 else max(0.0, min(1.0, ((x - ax) * abx + (y - ay) * aby) / denom))
            px = ax + abx * t
            py = ay + aby * t
            metric = (px - x) ** 2 + (py - y) ** 2
            if metric < best_metric:
                best_metric = metric
                best_index = index + 1
        return best_index

    def _catmull_rom(self, p0: float, p1: float, p2: float, p3: float, t: float) -> float:
        return 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t
        )

    def _slugify(self, text: str) -> str:
        clean = "".join(char.lower() if char.isalnum() else "-" for char in (text or "level"))
        return clean.strip("-") or "level"


def main() -> None:
    BuilderApp().run()


if __name__ == "__main__":
    main()
