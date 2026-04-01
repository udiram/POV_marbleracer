from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import (
    AmbientLight,
    AntialiasAttrib,
    CardMaker,
    DirectionalLight,
    Filename,
    Fog,
    NodePath,
    PNMImage,
    PointLight,
    TextNode,
    Texture,
    TransparencyAttrib,
    Vec3,
    Vec4,
    WindowProperties,
    loadPrcFileData,
)

from .app import MODEL_CANDIDATES, patch_loader, simplepbr
from .optimization import EpisodeResult, GeneticActionPlanner, GenerationStats, MarbleRaceEnv
from .physics import (
    GuideObstacle,
    SimulationConfig,
    boost_pad_center_position,
    obstacle_center_position,
    ramp_normal,
    ramp_segment_at_distance,
    ramp_side,
    ramp_surface_point,
)

loadPrcFileData("", "window-title Marble Training Viz")


@dataclass(frozen=True, slots=True)
class TrainingVizConfig:
    population_size: int = 18
    elite_count: int = 4
    generations: int = 8
    seed: int = 0
    dt: float = 1.0 / 120.0
    max_time: float | None = None
    steps_per_frame: int = 4
    hold_frames_between_generations: int = 30
    output_dir: Path | None = None
    capture_frames: bool = False
    compile_video: bool = False
    auto_close: bool = False


@dataclass(slots=True)
class CandidateRun:
    index: int
    env: MarbleRaceEnv
    plan: tuple[tuple[float, float], ...]
    marble_root: NodePath
    glow_root: NodePath
    color: Vec4
    trail: list[NodePath]
    finished: bool = False
    last_snapshot_time: float = 0.0


def resolve_model_path(name: str) -> str | None:
    for candidate in MODEL_CANDIDATES.get(name, ()):
        if candidate.exists():
            return str(candidate)
    return None


def generation_leader_key(result: EpisodeResult) -> tuple[int, float, float, float]:
    finish_bias = -result.finish_time if result.finish_time is not None else float("-inf")
    return (
        1 if result.completed else 0,
        result.progress_ratio,
        finish_bias,
        result.total_reward,
    )


def generation_leader_index(results: list[EpisodeResult]) -> int:
    return max(range(len(results)), key=lambda index: generation_leader_key(results[index]))


def export_training_summary(
    *,
    output_dir: Path,
    config: TrainingVizConfig,
    run_config: SimulationConfig,
    history: list[GenerationStats],
    winner_history: list[dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    training_config = asdict(config)
    training_config["output_dir"] = str(config.output_dir) if config.output_dir is not None else None
    summary = {
        "training": training_config,
        "course": {
            "length": run_config.length,
            "height": run_config.height,
            "width": run_config.width,
            "angle_deg": run_config.angle_deg,
            "obstacle_count": len(run_config.obstacles),
            "boost_count": len(run_config.boost_pads),
        },
        "history": [asdict(item) for item in history],
        "winners": winner_history,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    with (output_dir / "generation_history.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "generation",
                "best_reward",
                "mean_reward",
                "completion_rate",
                "best_finish_time",
                "winner_index",
                "winner_progress_ratio",
                "winner_completed",
                "winner_finish_time",
            ]
        )
        for stats, winner in zip(history, winner_history):
            writer.writerow(
                [
                    stats.generation,
                    stats.best_reward,
                    stats.mean_reward,
                    stats.completion_rate,
                    stats.best_finish_time,
                    winner["winner_index"],
                    winner["progress_ratio"],
                    winner["completed"],
                    winner["finish_time"],
                ]
            )


class MarbleTrainingVizApp(ShowBase):
    def __init__(self, sim_config: SimulationConfig, viz_config: TrainingVizConfig) -> None:
        super().__init__()
        self.sim_config = sim_config
        self.viz_config = viz_config
        self._texture_cache: dict[str, Texture] = {}
        self._chart_textures: dict[str, Texture] = {}
        self._chart_nodes: dict[str, NodePath] = {}
        self._population: list[list[tuple[float, float]]] = []
        self._candidate_runs: list[CandidateRun] = []
        self._history: list[GenerationStats] = []
        self._winner_history: list[dict[str, object]] = []
        self._generation_index = 0
        self._hold_frames = 0
        self._frame_index = 0
        self._leader_index = 0
        self._last_results: list[EpisodeResult] = []
        self._finished = False
        self._horizon_steps = 0
        self._trainer = GeneticActionPlanner(
            lambda: MarbleRaceEnv(
                config=self.sim_config,
                dt=self.viz_config.dt,
                action_repeat=1,
                max_time=self.viz_config.max_time,
            ),
            population_size=self.viz_config.population_size,
            elite_count=self.viz_config.elite_count,
            generations=self.viz_config.generations,
            seed=self.viz_config.seed,
        )
        env_template = MarbleRaceEnv(
            config=self.sim_config,
            dt=self.viz_config.dt,
            action_repeat=1,
            max_time=self.viz_config.max_time,
        )
        self._horizon_steps = env_template.horizon_steps()
        self._population = self._trainer._initial_population(self._horizon_steps)

        self._setup_window()
        self._setup_scene()
        self._build_world()
        self._build_ui()
        self._setup_camera()
        self._start_generation(self._population)
        self.taskMgr.add(self._tick, "training-viz-tick")

    def _setup_window(self) -> None:
        props = WindowProperties()
        props.setSize(1600, 960)
        props.setTitle("Marble Training Visualization")
        if self.win is not None:
            self.win.requestProperties(props)
        self.setBackgroundColor(0.015, 0.025, 0.045, 1.0)
        self.render.setAntialias(AntialiasAttrib.MAuto)
        self.render.setShaderAuto()
        if patch_loader is not None:
            patch_loader(self.loader)
        if simplepbr is not None and self.win is not None:
            simplepbr.init(enable_shadows=False, use_occlusion_maps=False, max_lights=8)

    def _setup_scene(self) -> None:
        fog = Fog("training-fog")
        fog.setColor(0.03, 0.04, 0.07)
        fog.setExpDensity(0.004)
        self.render.setFog(fog)

        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.34, 0.36, 0.42, 1.0))
        self.render.setLight(self.render.attachNewNode(ambient))

        sun = DirectionalLight("sun")
        sun.setColor(Vec4(0.96, 0.92, 0.84, 1.0))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(-32, -34, 0)
        self.render.setLight(sun_np)

        fill = DirectionalLight("fill")
        fill.setColor(Vec4(0.20, 0.28, 0.38, 1.0))
        fill_np = self.render.attachNewNode(fill)
        fill_np.setHpr(142, -18, 0)
        self.render.setLight(fill_np)

        glow = PointLight("goal")
        glow.setColor(Vec4(0.26, 0.64, 0.96, 1.0))
        glow_np = self.render.attachNewNode(glow)
        glow_np.setPos(self.sim_config.length - 1.4, 0.0, 3.0)
        self.render.setLight(glow_np)

    def _build_world(self) -> None:
        self.scene_root = self.render.attachNewNode("training-scene")
        self._build_backdrop()
        self._build_ground()
        self._build_ramp()
        self._build_rails()
        self._build_boost_pads()
        self._build_obstacles()
        self._build_finish_gate()
        self._build_start_zone()

    def _build_backdrop(self) -> None:
        for index, (width, height, x, y, z, color, h) in enumerate(
            (
                (self.sim_config.length + 34.0, 20.0, self.sim_config.length * 0.5, -13.0, 8.2, Vec4(0.05, 0.06, 0.10, 1.0), 0.0),
                (self.sim_config.length + 34.0, 20.0, self.sim_config.length * 0.5, 13.0, 8.2, Vec4(0.05, 0.06, 0.10, 1.0), 180.0),
            )
        ):
            cm = CardMaker(f"backdrop-{index}")
            cm.setFrame(-width * 0.5, width * 0.5, -height * 0.5, height * 0.5)
            card = self.scene_root.attachNewNode(cm.generate())
            card.setColor(color)
            card.setTexture(self._get_texture("backdrop"))
            card.setPos(x, y, z)
            card.setH(h)
            card.setTwoSided(True)

    def _build_ground(self) -> None:
        platform = self.scene_root.attachNewNode("platform")
        self._attach_centered_box(
            platform,
            Vec3(self.sim_config.length + 14.0, 18.0, 0.9),
            Vec4(0.08, 0.09, 0.12, 1.0),
            style="panel",
        )
        platform.setPos(self.sim_config.length * 0.5, 0.0, -0.46)

        runway = self.scene_root.attachNewNode("runway")
        self._attach_centered_box(
            runway,
            Vec3(self.sim_config.length + 8.0, 5.8, 0.05),
            Vec4(0.10, 0.13, 0.18, 1.0),
            style="runway",
        )
        runway.setPos(self.sim_config.length * 0.5, 0.0, 0.05)

    def _build_ramp(self) -> None:
        segment_count = len(self.sim_config.segment_headings_deg) * 4
        slice_length = self.sim_config.length / max(1, segment_count)
        for index in range(segment_count):
            distance = min(self.sim_config.length - slice_length * 0.5, (index + 0.5) * slice_length)
            segment = ramp_segment_at_distance(self.sim_config, distance)
            center = ramp_surface_point(self.sim_config, distance)
            normal = ramp_normal(self.sim_config, distance)

            skin = self.scene_root.attachNewNode(f"track-skin-{index}")
            skin.setPos(center + normal * 0.003)
            skin.setHpr(segment.heading_deg, segment.bank_deg, self.sim_config.angle_deg)

            self._attach_centered_box(
                skin,
                Vec3(slice_length * 1.08, self.sim_config.width * 0.88, 0.008),
                Vec4(0.18, 0.19, 0.22, 1.0),
                style="track",
            )
            lane = skin.attachNewNode("lane")
            self._attach_centered_box(
                lane,
                Vec3(slice_length * 1.04, self.sim_config.width * 0.08, 0.010),
                Vec4(0.88, 0.90, 0.94, 1.0),
                style="screen",
            )
            lane.setZ(0.006)

    def _build_rails(self) -> None:
        lateral = self.sim_config.width * 0.5 - self.sim_config.rail_width * 0.5
        segment_count = len(self.sim_config.segment_headings_deg) * 4
        slice_length = self.sim_config.length / max(1, segment_count)
        for index in range(segment_count):
            distance = min(self.sim_config.length - slice_length * 0.5, (index + 0.5) * slice_length)
            segment = ramp_segment_at_distance(self.sim_config, distance)
            center = ramp_surface_point(self.sim_config, distance)
            side = ramp_side(self.sim_config, distance)
            normal = ramp_normal(self.sim_config, distance)
            for lane_side in (-1.0, 1.0):
                rail = self.scene_root.attachNewNode(f"rail-{index}-{lane_side}")
                rail.setPos(center + side * (lateral * lane_side) + normal * (self.sim_config.rail_height * 0.5))
                rail.setHpr(segment.heading_deg, segment.bank_deg, self.sim_config.angle_deg)
                self._attach_centered_box(
                    rail,
                    Vec3(slice_length * 1.02, self.sim_config.rail_width * 0.55, self.sim_config.rail_height),
                    Vec4(0.12, 0.14, 0.18, 1.0),
                    style="panel",
                )

    def _build_boost_pads(self) -> None:
        for index, boost in enumerate(self.sim_config.boost_pads):
            segment = ramp_segment_at_distance(self.sim_config, boost.distance_along_ramp)
            pad = self.scene_root.attachNewNode(f"boost-{index}")
            pad.setPos(boost_pad_center_position(self.sim_config, boost))
            pad.setHpr(segment.heading_deg, segment.bank_deg, self.sim_config.angle_deg)
            self._attach_centered_box(
                pad,
                Vec3(boost.length, boost.width, 0.018),
                Vec4(0.20, 0.72, 0.94, 1.0),
                style="screen",
            )

    def _build_obstacles(self) -> None:
        for index, obstacle in enumerate(self.sim_config.obstacles):
            segment = ramp_segment_at_distance(self.sim_config, obstacle.distance_along_ramp)
            node = self.scene_root.attachNewNode(f"obstacle-{index}")
            node.setPos(obstacle_center_position(self.sim_config, obstacle))
            node.setHpr(segment.heading_deg + obstacle.heading_deg, segment.bank_deg, self.sim_config.angle_deg)
            self._attach_centered_box(
                node,
                Vec3(obstacle.length, obstacle.width, obstacle.height),
                Vec4(0.94, 0.42, 0.18, 1.0),
                style="panel",
            )

    def _build_finish_gate(self) -> None:
        gate = self.scene_root.attachNewNode("finish")
        gate.setPos(self.sim_config.length - 1.0, 0.0, 0.0)
        for side in (-1, 1):
            post = gate.attachNewNode(f"post-{side}")
            self._attach_centered_box(post, Vec3(0.16, 0.16, 2.6), Vec4(0.74, 0.80, 0.88, 1.0), style="panel")
            post.setPos(0.0, side * 1.45, 1.30)
        beam = gate.attachNewNode("beam")
        self._attach_centered_box(beam, Vec3(0.18, 3.0, 0.12), Vec4(0.22, 0.72, 0.96, 1.0), style="screen")
        beam.setPos(0.0, 0.0, 2.65)

    def _build_start_zone(self) -> None:
        start = self.scene_root.attachNewNode("start")
        self._attach_centered_box(start, Vec3(2.2, 1.3, 0.05), Vec4(0.14, 0.16, 0.20, 1.0), style="panel")
        start.setPos(0.8, 0.0, self.sim_config.height - 0.36)

    def _build_ui(self) -> None:
        self.title = OnscreenText(
            text="GENETIC MARBLE TRAINING",
            pos=(-1.28, 0.92),
            align=TextNode.ALeft,
            scale=0.052,
            fg=(0.94, 0.96, 1.0, 1.0),
            mayChange=False,
        )
        self.subtitle = OnscreenText(
            text="Population rollouts, generation champion, and training curves",
            pos=(-1.28, 0.85),
            align=TextNode.ALeft,
            scale=0.030,
            fg=(0.68, 0.74, 0.84, 1.0),
            mayChange=False,
        )
        self.status = OnscreenText(
            text="",
            pos=(-1.28, 0.74),
            align=TextNode.ALeft,
            scale=0.035,
            fg=(0.90, 0.94, 0.98, 1.0),
            mayChange=True,
        )
        self.leaderboard = OnscreenText(
            text="",
            pos=(1.28, 0.88),
            align=TextNode.ARight,
            scale=0.032,
            fg=(0.88, 0.92, 0.98, 1.0),
            mayChange=True,
        )
        self.event = OnscreenText(
            text="Generation 1 is live",
            pos=(0.0, -0.88),
            align=TextNode.ACenter,
            scale=0.038,
            fg=(0.22, 0.76, 0.98, 1.0),
            mayChange=True,
        )
        self.controls = OnscreenText(
            text="Blue halo = current leader   White = elites surviving forward   Optional frame capture builds an MP4",
            pos=(0.0, -0.95),
            align=TextNode.ACenter,
            scale=0.026,
            fg=(0.66, 0.72, 0.80, 1.0),
            mayChange=False,
        )

        self._chart_nodes["history"] = self._create_chart_card("history", -1.28, -0.05, -0.82, -0.18)
        self._chart_nodes["population"] = self._create_chart_card("population", 0.10, 1.28, -0.82, -0.18)

    def _setup_camera(self) -> None:
        if self.camera is None:
            return
        anchor = Vec3(self.sim_config.length * 0.48, -self.sim_config.length * 0.16 - 4.0, max(26.0, self.sim_config.height * 0.72))
        self.camera.setPos(anchor)
        self.camera.lookAt(self.sim_config.length * 0.55, 0.0, self.sim_config.height * 0.22)
        if self.camLens is not None:
            self.camLens.setFov(42.0)

    def _create_chart_card(self, name: str, left: float, right: float, bottom: float, top: float) -> NodePath:
        cm = CardMaker(name)
        cm.setFrame(left, right, bottom, top)
        node = self.aspect2d.attachNewNode(cm.generate())
        node.setTransparency(TransparencyAttrib.MAlpha)
        texture = Texture(name)
        node.setTexture(texture)
        node.setColor(1.0, 1.0, 1.0, 1.0)
        self._chart_textures[name] = texture
        return node

    def _attach_centered_box(self, parent: NodePath, size: Vec3, color: Vec4, *, style: str | None = None) -> None:
        box = self.loader.loadModel("models/box")
        box.reparentTo(parent)
        box.setPos(-size.x * 0.5, -size.y * 0.5, -size.z * 0.5)
        box.setScale(size)
        if style is None:
            box.setTextureOff(1)
        else:
            box.setTexture(self._get_texture(style), 1)
        box.setColor(color)

    def _load_sphere(self, parent: NodePath, scale: float, color: Vec4, pos: Vec3 | None = None) -> NodePath:
        model_path = resolve_model_path("sphere")
        if model_path is None:
            model = self.loader.loadModel("models/ball")
        else:
            model = self.loader.loadModel(model_path)
        model.reparentTo(parent)
        model.setScale(scale)
        model.setColor(color)
        model.setTextureOff(1)
        if pos is not None:
            model.setPos(pos)
        return model

    def _get_texture(self, style: str) -> Texture:
        cached = self._texture_cache.get(style)
        if cached is not None:
            return cached

        image = PNMImage(256, 256, 4)
        image.fill(0.0, 0.0, 0.0)
        image.alpha_fill(1.0)
        for y in range(256):
            for x in range(256):
                if style == "panel":
                    base = 0.10 + 0.03 * (1.0 - y / 255.0)
                    stripe = 0.02 if (x + y) % 36 < 2 else 0.0
                    image.set_xel_a(x, y, base + stripe, base + stripe, base + stripe * 1.2, 1.0)
                elif style == "screen":
                    glow = 0.10 if y % 26 < 2 else 0.0
                    image.set_xel_a(x, y, 0.22 + glow, 0.42 + glow, 0.62 + glow, 1.0)
                elif style == "track":
                    seam = 0.04 if y % 64 < 2 else 0.0
                    image.set_xel_a(x, y, 0.16 + seam, 0.17 + seam, 0.19 + seam, 1.0)
                elif style == "runway":
                    guide = 0.12 if x in range(18, 28) or x in range(228, 238) else 0.0
                    image.set_xel_a(x, y, 0.08 + guide, 0.12 + guide, 0.18 + guide, 1.0)
                elif style == "backdrop":
                    fade = 0.03 + 0.04 * (1.0 - y / 255.0)
                    image.set_xel_a(x, y, fade, fade + 0.01, fade + 0.03, 1.0)
        texture = Texture(style)
        texture.load(image)
        self._texture_cache[style] = texture
        return texture

    def _candidate_color(self, index: int) -> Vec4:
        palette = (
            Vec4(0.24, 0.72, 0.96, 1.0),
            Vec4(0.96, 0.52, 0.22, 1.0),
            Vec4(0.20, 0.88, 0.72, 1.0),
            Vec4(0.98, 0.82, 0.22, 1.0),
            Vec4(0.96, 0.36, 0.62, 1.0),
            Vec4(0.76, 0.58, 0.96, 1.0),
        )
        base = palette[index % len(palette)]
        band = 0.04 * (index // len(palette))
        return Vec4(min(1.0, base.x + band), min(1.0, base.y + band), min(1.0, base.z + band), 1.0)

    def _spawn_candidate_visual(self, index: int, env: MarbleRaceEnv, plan: list[tuple[float, float]]) -> CandidateRun:
        color = self._candidate_color(index)
        root = self.scene_root.attachNewNode(f"marble-{index}")
        self._load_sphere(root, self.sim_config.marble_radius, color)
        marker = root.attachNewNode("marker")
        self._load_sphere(marker, self.sim_config.marble_radius * 0.24, Vec4(0.05, 0.08, 0.12, 1.0), Vec3(self.sim_config.marble_radius, 0.0, 0.0))

        glow = root.attachNewNode("glow")
        self._attach_centered_box(
            glow,
            Vec3(self.sim_config.marble_radius * 1.6, self.sim_config.marble_radius * 1.6, 0.02),
            Vec4(color.x, color.y, color.z, 0.70),
            style="screen",
        )
        glow.setPos(0.0, 0.0, -self.sim_config.marble_radius * 0.95)
        root.setPos(env.snapshot.position)

        trail_nodes: list[NodePath] = []
        for trail_index in range(6):
            trail = root.attachNewNode(f"trail-{trail_index}")
            self._attach_centered_box(
                trail,
                Vec3(self.sim_config.marble_radius * (1.15 - trail_index * 0.10), 0.012, 0.012),
                Vec4(color.x, color.y, color.z, max(0.12, 0.36 - trail_index * 0.05)),
                style="screen",
            )
            trail.setPos(-self.sim_config.marble_radius * (1.0 + trail_index * 0.28), 0.0, 0.0)
            trail_nodes.append(trail)

        return CandidateRun(
            index=index,
            env=env,
            plan=tuple(plan),
            marble_root=root,
            glow_root=glow,
            color=color,
            trail=trail_nodes,
        )

    def _start_generation(self, population: list[list[tuple[float, float]]]) -> None:
        for candidate in self._candidate_runs:
            candidate.marble_root.removeNode()
        self._candidate_runs.clear()
        self._leader_index = 0
        self.event.setText(f"Generation {self._generation_index + 1} is live")

        for index, plan in enumerate(population):
            env = MarbleRaceEnv(
                config=self.sim_config,
                dt=self.viz_config.dt,
                action_repeat=1,
                max_time=self.viz_config.max_time,
            )
            self._candidate_runs.append(self._spawn_candidate_visual(index, env, plan))

        self._update_status()
        self._update_charts()

    def _advance_population(self) -> None:
        all_finished = True
        current_results: list[EpisodeResult] = []
        for candidate in self._candidate_runs:
            if not candidate.finished:
                action_index = min(candidate.env._steps, len(candidate.plan) - 1)
                _, _, terminated, truncated, _ = candidate.env.step(candidate.plan[action_index])
                candidate.finished = terminated or truncated
                snapshot = candidate.env.snapshot
                candidate.marble_root.setPos(snapshot.position)
                candidate.marble_root.setQuat(candidate.env.simulation.marble_np.getQuat())
                speed_scale = min(1.8, 0.8 + snapshot.speed * 0.08)
                candidate.glow_root.setScale(speed_scale, speed_scale, 1.0)
            all_finished &= candidate.finished
            current_results.append(candidate.env.result())

        self._last_results = current_results
        self._leader_index = generation_leader_index(current_results)
        for candidate in self._candidate_runs:
            is_leader = candidate.index == self._leader_index
            candidate.glow_root.setColorScale(
                1.5 if is_leader else 0.75,
                1.5 if is_leader else 0.75,
                1.5 if is_leader else 0.75,
                1.0,
            )
            candidate.marble_root.setTransparency(TransparencyAttrib.MAlpha)
            candidate.marble_root.setAlphaScale(1.0 if is_leader else 0.72)
        self._update_status()
        self._update_charts()

        if all_finished:
            self._finish_generation(current_results)

    def _finish_generation(self, results: list[EpisodeResult]) -> None:
        scores = [result.total_reward for result in results]
        ranking = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        elites = [list(self._population[index]) for index in ranking[: self.viz_config.elite_count]]
        winner_index = generation_leader_index(results)
        completed = [result for result in results if result.completed]
        self._history.append(
            GenerationStats(
                generation=self._generation_index,
                best_reward=scores[ranking[0]],
                mean_reward=sum(scores) / len(scores),
                completion_rate=len(completed) / len(results),
                best_finish_time=min((item.finish_time for item in completed), default=None),
            )
        )
        winner = results[winner_index]
        self._winner_history.append(
            {
                "generation": self._generation_index,
                "winner_index": winner_index,
                "progress_ratio": winner.progress_ratio,
                "completed": winner.completed,
                "finish_time": winner.finish_time,
                "reward": winner.total_reward,
            }
        )
        self.event.setText(
            f"Generation {self._generation_index + 1} winner: marble {winner_index + 1}   "
            f"progress {winner.progress_ratio * 100.0:0.1f}%"
        )
        self._generation_index += 1
        self._update_charts()

        if self._generation_index >= self.viz_config.generations:
            self._finish_training()
            return

        self._population = self._trainer._breed_next_generation(elites, self._horizon_steps)
        self._hold_frames = self.viz_config.hold_frames_between_generations

    def _finish_training(self) -> None:
        if self._finished:
            return
        self._finished = True
        self.event.setText("Training complete")
        if self.viz_config.output_dir is not None:
            export_training_summary(
                output_dir=self.viz_config.output_dir,
                config=self.viz_config,
                run_config=self.sim_config,
                history=self._history,
                winner_history=self._winner_history,
            )
            if self.viz_config.capture_frames and self.viz_config.compile_video:
                self._compile_video()
        if self.viz_config.auto_close:
            self.userExit()

    def _compile_video(self) -> None:
        if self.viz_config.output_dir is None:
            return
        frame_dir = self.viz_config.output_dir / "frames"
        if not frame_dir.exists():
            return
        output_path = self.viz_config.output_dir / "training_run.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                "60",
                "-i",
                str(frame_dir / "frame_%06d.png"),
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ],
            check=False,
            capture_output=True,
        )

    def _capture_frame(self) -> None:
        if not self.viz_config.capture_frames or self.viz_config.output_dir is None or self.win is None:
            return
        frame_dir = self.viz_config.output_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        path = frame_dir / f"frame_{self._frame_index:06d}.png"
        self.win.saveScreenshot(Filename.fromOsSpecific(str(path)))
        self._frame_index += 1

    def _update_status(self) -> None:
        if not self._candidate_runs:
            return
        results = self._last_results or [candidate.env.result() for candidate in self._candidate_runs]
        leader = results[self._leader_index]
        finished = sum(1 for result in results if result.completed or result.off_track or result.stalled or result.timed_out)
        self.status.setText(
            "\n".join(
                (
                    f"Generation {min(self._generation_index + 1, self.viz_config.generations)} / {self.viz_config.generations}",
                    f"Population {self.viz_config.population_size}   Elites {self.viz_config.elite_count}",
                    f"Leader marble {self._leader_index + 1}",
                    f"Leader progress {leader.progress_ratio * 100.0:0.1f}%",
                    f"Leader reward {leader.total_reward:0.2f}",
                    f"Finished this gen {finished}/{len(results)}",
                )
            )
        )

        ranked = sorted(range(len(results)), key=lambda index: generation_leader_key(results[index]), reverse=True)[:5]
        leaderboard_lines = ["Current Standings"]
        for slot, index in enumerate(ranked, start=1):
            result = results[index]
            label = f"{slot}. M{index + 1}  {result.progress_ratio * 100.0:5.1f}%"
            if result.finish_time is not None:
                label += f"  {result.finish_time:0.2f}s"
            leaderboard_lines.append(label)
        self.leaderboard.setText("\n".join(leaderboard_lines))

    def _update_charts(self) -> None:
        self._render_history_chart()
        self._render_population_chart()

    def _plot_image(self, width: int, height: int) -> PNMImage:
        image = PNMImage(width, height, 4)
        image.fill(0.04, 0.05, 0.08)
        image.alpha_fill(0.96)
        for y in range(height):
            if y % max(1, height // 5) == 0:
                for x in range(width):
                    image.set_xel_a(x, y, 0.10, 0.12, 0.18, 0.96)
        return image

    def _draw_rect(self, image: PNMImage, left: int, top: int, right: int, bottom: int, color: Vec4) -> None:
        width = image.getXSize()
        height = image.getYSize()
        for y in range(max(0, top), min(height, bottom)):
            for x in range(max(0, left), min(width, right)):
                image.set_xel_a(x, y, color.x, color.y, color.z, color.w)

    def _draw_line(self, image: PNMImage, points: list[tuple[int, int]], color: Vec4) -> None:
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            steps = max(dx, dy, 1)
            for step in range(steps + 1):
                t = step / steps
                x = int(round(x0 + (x1 - x0) * t))
                y = int(round(y0 + (y1 - y0) * t))
                if 0 <= x < image.getXSize() and 0 <= y < image.getYSize():
                    image.set_xel_a(x, y, color.x, color.y, color.z, color.w)

    def _draw_label_band(self, image: PNMImage) -> None:
        self._draw_rect(image, 0, image.getYSize() - 22, image.getXSize(), image.getYSize(), Vec4(0.02, 0.03, 0.05, 0.96))

    def _render_history_chart(self) -> None:
        image = self._plot_image(640, 280)
        self._draw_label_band(image)
        if self._history:
            padding = 28
            usable_width = image.getXSize() - padding * 2
            usable_height = image.getYSize() - padding * 2 - 22
            rewards = [item.best_reward for item in self._history]
            means = [item.mean_reward for item in self._history]
            lo = min(rewards + means)
            hi = max(rewards + means)
            span = max(1e-6, hi - lo)

            def to_points(values: list[float]) -> list[tuple[int, int]]:
                points: list[tuple[int, int]] = []
                for index, value in enumerate(values):
                    x = padding if len(values) == 1 else padding + int(index / (len(values) - 1) * usable_width)
                    normalized = (value - lo) / span
                    y = padding + int((1.0 - normalized) * usable_height)
                    points.append((x, y))
                return points

            self._draw_line(image, to_points(means), Vec4(0.96, 0.58, 0.22, 1.0))
            self._draw_line(image, to_points(rewards), Vec4(0.24, 0.76, 0.98, 1.0))
        self._chart_textures["history"].load(image)

    def _render_population_chart(self) -> None:
        image = self._plot_image(640, 280)
        self._draw_label_band(image)
        results = self._last_results or [candidate.env.result() for candidate in self._candidate_runs]
        if results:
            padding = 20
            chart_height = image.getYSize() - 50
            bar_width = max(6, (image.getXSize() - padding * 2) // max(1, len(results)))
            for index, result in enumerate(results):
                left = padding + index * bar_width
                right = left + max(4, bar_width - 2)
                height = int(result.progress_ratio * (chart_height - padding))
                color = self._candidate_color(index)
                if index == self._leader_index:
                    color = Vec4(0.92, 0.96, 1.0, 1.0)
                self._draw_rect(image, left, chart_height - height, right, chart_height, color)
        self._chart_textures["population"].load(image)

    def _tick(self, task: Task) -> int:
        if self._finished:
            self._capture_frame()
            return Task.cont if not self.viz_config.auto_close else Task.done

        if self._hold_frames > 0:
            self._hold_frames -= 1
            if self._hold_frames == 0:
                self._start_generation(self._population)
            self._capture_frame()
            return Task.cont

        for _ in range(max(1, self.viz_config.steps_per_frame)):
            if self._finished or self._hold_frames > 0:
                break
            self._advance_population()
        self._capture_frame()
        return Task.cont


def main() -> None:
    defaults = SimulationConfig()
    parser = argparse.ArgumentParser(description="Visualize genetic marble training on a shared track.")
    parser.add_argument("--level", type=str, default=None, help="Saved level name or explicit level JSON path.")
    parser.add_argument("--obstacle-count", type=int, default=defaults.obstacle_count)
    parser.add_argument("--course-seed", type=int, default=defaults.course_seed)
    parser.add_argument("--population-size", type=int, default=18)
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps-per-frame", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--capture-frames", action="store_true")
    parser.add_argument("--compile-video", action="store_true")
    parser.add_argument("--auto-close", action="store_true")
    args = parser.parse_args()

    if args.level:
        from .levels import level_to_config, load_level

        config = level_to_config(load_level(args.level))
    else:
        config = SimulationConfig(obstacle_count=args.obstacle_count, course_seed=args.course_seed)

    app = MarbleTrainingVizApp(
        config,
        TrainingVizConfig(
            population_size=args.population_size,
            elite_count=args.elite_count,
            generations=args.generations,
            seed=args.seed,
            steps_per_frame=args.steps_per_frame,
            output_dir=args.output_dir,
            capture_frames=args.capture_frames,
            compile_video=args.compile_video,
            auto_close=args.auto_close,
        ),
    )
    app.run()


if __name__ == "__main__":
    main()
