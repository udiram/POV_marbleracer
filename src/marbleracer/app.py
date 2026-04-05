from __future__ import annotations

import os
from dataclasses import dataclass
from math import ceil, cos, degrees, exp, sin, tau
from pathlib import Path

from panda3d.bullet import BulletRigidBodyNode, BulletSphereShape
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import (
    AmbientLight,
    AntialiasAttrib,
    CardMaker,
    DirectionalLight,
    Fog,
    KeyboardButton,
    NodePath,
    PNMImage,
    PointLight,
    SamplerState,
    TextNode,
    Texture,
    TransparencyAttrib,
    Vec3,
    Vec4,
    WindowProperties,
    loadPrcFileData,
)

from .gameplay import (
    build_course_sections,
    format_delta,
    format_seconds,
    grade_time_trial,
    grade_run,
    rate_section,
)
from .levels import TrackLevel, level_from_config, level_to_config, list_levels
from .physics import (
    GuideObstacle,
    MarbleRampSimulation,
    SimulationConfig,
    boost_pad_center_position,
    lane_center_offsets,
    lane_surface_width,
    obstacle_pose,
    obstacle_center_position,
    path_distance_for_position,
    path_distance_for_position_near,
    ramp_normal,
    ramp_segment_at_distance,
    ramp_side,
    ramp_surface_point,
    ramp_tangent,
    split_gap_width,
)
from .progress import GhostSample, load_progress, save_progress, update_level_progress
from .race_rules import (
    effective_track_distance as shared_effective_track_distance,
    finish_grace_distance,
    finish_line_distance as shared_finish_line_distance,
    finish_metrics as shared_finish_metrics,
    has_crossed_finish as shared_has_crossed_finish,
    is_definitely_off_track as shared_is_definitely_off_track,
    should_finish_run as shared_should_finish_run,
    track_relative_offsets as shared_track_relative_offsets,
)
try:
    import simplepbr
except ImportError:  # pragma: no cover - optional dependency
    simplepbr = None

try:
    from gltf import patch_loader
except ImportError:  # pragma: no cover - optional dependency
    patch_loader = None

loadPrcFileData("", "window-title Marble Physics Run")
loadPrcFileData("", "sync-video true")
loadPrcFileData("", "framebuffer-multisample 1")
loadPrcFileData("", "multisamples 4")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHYSICS_REPO_ROOT = PROJECT_ROOT.parent / "Physics_2VG3"
MODEL_CANDIDATES: dict[str, tuple[Path, ...]] = {
    "sphere": (
        PHYSICS_REPO_ROOT / "HW2" / "sphere.egg.pz",
        PHYSICS_REPO_ROOT / "participation_2" / "sphere.egg.pz",
    ),
    "wheel": (PHYSICS_REPO_ROOT / "HW1" / "wheel.egg",),
    "frame": (PHYSICS_REPO_ROOT / "HW1" / "frame.egg",),
    "cylinder": (PHYSICS_REPO_ROOT / "HW2" / "Cylinder.egg",),
    "box": (PHYSICS_REPO_ROOT / "participation_2" / "box.egg",),
    "cube": (PHYSICS_REPO_ROOT / "participation_3" / "Cube.egg",),
}


def resolve_model_path(name: str) -> str | None:
    for candidate in MODEL_CANDIDATES.get(name, ()):
        if candidate.exists():
            return str(candidate)
    return None


def smoothing_alpha(response: float, dt: float) -> float:
    if response <= 0.0 or dt <= 0.0:
        return 0.0
    return 1.0 - exp(-response * dt)


def smooth_value(current: float, target: float, response: float, dt: float) -> float:
    return current + (target - current) * smoothing_alpha(response, dt)


def smooth_vec3(current: Vec3, target: Vec3, response: float, dt: float) -> Vec3:
    alpha = smoothing_alpha(response, dt)
    return current * (1.0 - alpha) + target * alpha


def clamp_frame_dt(dt: float, *, max_dt: float = 1.0 / 30.0) -> float:
    return max(0.0, min(dt, max_dt))


def smooth_binary_state(
    current: float,
    active: bool,
    dt: float,
    *,
    attack_response: float,
    release_response: float,
) -> float:
    target = 1.0 if active else 0.0
    response = attack_response if target > current else release_response
    return smooth_value(current, target, response, dt)


@dataclass(slots=True)
class PulseNode:
    node: NodePath
    base_color: Vec4
    amplitude: float
    speed: float
    phase: float = 0.0


@dataclass(slots=True)
class BoostPadVisual:
    distance: float
    root: NodePath
    glow: NodePath
    phase: float
    pulse: float = 0.0


@dataclass(frozen=True, slots=True)
class LevelRuntime:
    key: str
    level: TrackLevel
    config: SimulationConfig


@dataclass(slots=True)
class GhostPlayback:
    samples: tuple[GhostSample, ...]
    root: NodePath
    marker: NodePath


@dataclass(slots=True)
class BotMarble:
    name: str
    body_np: NodePath
    body: BulletRigidBodyNode
    visual_root: NodePath
    lane_offset: float
    speed_bias: float
    progress: float = 0.0
    finish_time: float | None = None
    finish_order: int | None = None
    off_track_timer: float = 0.0
    fall_timer: float = 0.0
    reset_warning_second: int | None = None
    previous_position: Vec3 | None = None
    previous_progress: float = 0.0
    active_boost_pad_index: int | None = None
    last_boost_pad_index: int | None = None


@dataclass(slots=True)
class ConfettiPiece:
    node: NodePath
    velocity: Vec3
    spin_hpr: Vec3
    lifetime: float
    age: float = 0.0


@dataclass(slots=True)
class MovingObstacleVisual:
    obstacle: GuideObstacle
    root: NodePath
    marker: NodePath | None = None
    arm: NodePath | None = None


class MarbleRampApp(ShowBase):
    MODE_TIME_TRIAL = "time_trial"
    MODE_BOT_RACE = "bot_race"
    MODE_OPTIONS = (MODE_TIME_TRIAL, MODE_BOT_RACE)

    def __init__(
        self,
        config: SimulationConfig | None = None,
        *,
        levels: tuple[TrackLevel, ...] | None = None,
        initial_level: str | None = None,
        menu_disabled: bool = False,
        start_mode: str | None = None,
    ) -> None:
        super().__init__()
        loaded_levels = tuple(levels or list_levels())
        if not loaded_levels:
            loaded_levels = (level_from_config(config or SimulationConfig(), name="Quick Run"),)
        self.levels = tuple(
            LevelRuntime(
                key=level.name.lower().replace(" ", "-"),
                level=level,
                config=level_to_config(level),
            )
            for level in loaded_levels
        )
        self.progress = load_progress()
        self.selected_mode_index = 0
        if start_mode in self.MODE_OPTIONS:
            self.selected_mode_index = self.MODE_OPTIONS.index(start_mode)
        self.active_mode = self.MODE_OPTIONS[self.selected_mode_index]
        self.selected_level_index = 0
        self.level_index = 0
        initial_level_index = self._find_level_index(initial_level)
        if initial_level_index is not None:
            self.selected_level_index = initial_level_index
            self.level_index = initial_level_index
        self.level_runtime = self.levels[self.level_index]
        self.sim_config = self.level_runtime.config if config is None else config
        self.simulation = MarbleRampSimulation(self.sim_config)
        self.simulation.settle_start_contact()
        self.menu_disabled = menu_disabled
        self.enable_fancy_rendering = os.environ.get("MARBLERACER_FANCY_RENDERING", "").strip() == "1"
        self.paused = False
        self.camera_follow_distance = 6.6
        self.camera_height = 4.8
        self.camera_look_ahead = 3.8
        self.camera_side_offset = -0.25
        self.camera_target_drop = 0.45
        self.camera_track_follow_offset = 0.18
        self.camera_track_max_lag = 0.48
        self.camera_focus_lateral_lag = 1.6
        self.camera_focus_vertical_lag = 1.9
        self.camera_position_lag = 5.6
        self.camera_look_lag = 6.1
        self.camera_track_lag = 4.4
        self.camera_focus_lag = 3.8
        self.camera_heading_response = 2.0
        self.camera_velocity_heading_response = 2.8
        self.camera_collision_attack = 10.0
        self.camera_collision_release = 2.2
        self.camera_roll_lag = 6.6
        self.camera_forward = ramp_tangent(self.sim_config, 0.0)
        self.camera_forward.z = 0.0
        self.camera_forward.normalize()
        self.camera_velocity_forward = Vec3(self.camera_forward)
        self.camera_track_distance = 0.0
        self.camera_collision_blend = 0.0
        self.camera_focus_lateral_offset = 0.0
        self.camera_focus_vertical_offset = 0.0
        self.camera_roll = 0.0
        self.steering_input = 0.0
        self.brake_input = 0.0
        self.primary_action_pending = False
        self._space_key_down = False
        self._enter_key_down = False
        self.prop_spinners: list[NodePath] = []
        self.speed_lines: list[NodePath] = []
        self.pulse_nodes: list[PulseNode] = []
        self.boost_pads: list[BoostPadVisual] = []
        self._texture_cache: dict[str, Texture] = {}
        self.filters = None
        self.course_sections = build_course_sections(self.sim_config.length)
        self.race_phase = "title"
        self.countdown_timer = 3.6
        self.last_countdown_value = int(ceil(self.countdown_timer))
        self.session_best_time: float | None = None
        self.finish_time: float | None = None
        self.finish_grade = None
        self.failure_reason: str | None = None
        self.major_impacts = 0
        self.section_major_impacts = 0
        self.run_stability_loss = 0.0
        self.section_stability_loss = 0.0
        self.clean_sections = 0
        self.section_boost_hits = 0
        self.boost_chain = 0
        self.best_boost_chain = 0
        self.off_track_timer = 0.0
        self.off_track_grace = 1.20
        self.fall_reset_delay = 3.0
        self.player_fall_timer = 0.0
        self.player_reset_warning_second: int | None = None
        self.start_input_lock_timer = 0.0
        self.launch_assist_timer = 0.0
        self.launch_assist_duration = 0.42
        self.last_supported_path_distance = 0.0
        self.impact_event_cooldown = 0.0
        self.next_section_index = 0
        self.section_results: list[str] = []
        self.debug_overlay_visible = False
        self.event_text = "MARBLE PHYSICS RUN"
        self.event_subtitle = "Press Space to start."
        self.event_color = Vec4(1.00, 0.78, 0.24, 1.0)
        self.event_timer = 0.0
        self.flash_alpha = 0.0
        self.flash_color = Vec4(0.96, 0.42, 0.18, 0.0)
        self.boost_flash = 0.0
        self.finish_flash = 0.0
        self.start_burst = 0.0
        self.finish_order_counter = 0
        self.last_finish_place: int | None = None
        self.finish_confetti: list[ConfettiPiece] = []
        self.ghost_playback: GhostPlayback | None = None
        self.bot_marbles: list[BotMarble] = []
        self.moving_obstacle_visuals: list[MovingObstacleVisual] = []
        self.pending_ghost_samples: list[GhostSample] = []
        self._last_ghost_sample_time = 0.0
        self._stall_timer = 0.0
        self.current_section_name = self.course_sections[0].name
        self.current_section_focus = self.course_sections[0].focus
        start_pos = self.simulation.snapshot().position
        self.camera_focus = Vec3(start_pos)
        self.camera_look_target = Vec3(start_pos) + self.camera_forward * self.camera_look_ahead
        self.disableMouse()

        self._setup_window()
        self._setup_scene()
        self._build_world()
        self._build_ui()
        self._setup_input()
        self._load_level_progress()
        self._reset_run_state()
        if self.menu_disabled:
            self._start_selected_level()
        else:
            self._enter_menu_state()
        self.taskMgr.add(self._tick, "marble-ramp-tick")
        self._apply_state()

    @property
    def finish_line_distance(self) -> float:
        return shared_finish_line_distance(self.sim_config)

    @property
    def current_level(self) -> TrackLevel:
        return self.level_runtime.level

    @property
    def current_level_key(self) -> str:
        return self.level_runtime.key

    def _mode_label(self, mode: str) -> str:
        if mode == self.MODE_BOT_RACE:
            return "BOT RACE"
        return "TIME TRIAL"

    def _mode_description(self, mode: str) -> str:
        if mode == self.MODE_BOT_RACE:
            return "Race a pack of physical bot marbles. Collisions count."
        return "Race your saved ghost and chase the fastest clean lap."

    @property
    def selected_mode(self) -> str:
        return self.MODE_OPTIONS[self.selected_mode_index]

    def _find_level_index(self, level_ref: str | None) -> int | None:
        if level_ref is None:
            return None
        normalized = level_ref.strip().lower()
        for index, runtime in enumerate(self.levels):
            if normalized in {runtime.key, runtime.level.name.lower()}:
                return index
        return None

    def _is_level_unlocked(self, index: int) -> bool:
        runtime = self.levels[index]
        progress_entry = self.progress.levels.get(runtime.key)
        return runtime.level.unlock_index == 0 or (progress_entry is not None and progress_entry.unlocked)

    def _load_level_progress(self) -> None:
        progress_entry = self.progress.levels.get(self.current_level_key)
        self.session_best_time = progress_entry.best_time if progress_entry is not None else None
        if self.active_mode == self.MODE_TIME_TRIAL and progress_entry is not None:
            self._load_ghost(progress_entry.ghost)
        else:
            self._load_ghost(())

    def _load_runtime_level(self, index: int, *, reset_selection: bool = True) -> None:
        self.level_index = index
        if reset_selection:
            self.selected_level_index = index
        self.level_runtime = self.levels[index]
        self.sim_config = self.level_runtime.config
        self.simulation = MarbleRampSimulation(self.sim_config)
        self.simulation.settle_start_contact()
        self.course_sections = build_course_sections(self.sim_config.length)
        self.boost_pads = []
        self.pulse_nodes = []
        self.speed_lines = []
        self.prop_spinners = []
        self.moving_obstacle_visuals = []
        self._clear_bot_marbles()
        if hasattr(self, "scene_root") and not self.scene_root.isEmpty():
            self.scene_root.removeNode()
        self._build_world()
        if hasattr(self, "finish_glow_np") and not self.finish_glow_np.isEmpty():
            finish_point = ramp_surface_point(self.sim_config, self.finish_line_distance)
            self.finish_glow_np.setPos(finish_point.x, finish_point.y, finish_point.z + 3.2)
        self._spawn_bot_marbles()
        self._load_level_progress()
        self._reset()

    def _start_selected_level(self) -> None:
        self.active_mode = self.selected_mode
        self._load_runtime_level(self.selected_level_index)

    def _unlock_next_level(self) -> None:
        next_index = self.level_index + 1
        if next_index >= len(self.levels):
            return
        next_level = self.levels[next_index]
        self.progress = update_level_progress(self.progress, next_level.key, unlocked=True)
        save_progress(self.progress)

    def _enter_menu_state(self) -> None:
        self.race_phase = "menu"
        self.paused = False
        self.failure_reason = None
        self.selected_level_index = self.level_index
        self._show_event(
            "MODE SELECT",
            "Up/Down mode   Left/Right track   Enter start",
            Vec4(0.92, 0.94, 0.98, 1.0),
            hold=60.0,
        )

    def _enter_failed_state(self, reason: str, subtitle: str) -> None:
        self.race_phase = "failed"
        self.failure_reason = reason
        self.current_section_name = "Run Failed"
        self.current_section_focus = subtitle
        self._show_event(reason, subtitle, Vec4(0.96, 0.42, 0.18, 1.0), hold=60.0)
        self._trigger_flash(Vec4(0.96, 0.42, 0.18, 1.0), 0.22)

    def _setup_window(self) -> None:
        props = WindowProperties()
        props.setSize(1440, 900)
        props.setTitle("Marble Physics Run")
        if self.win and hasattr(self.win, "requestProperties"):
            self.win.requestProperties(props)
        self.setBackgroundColor(0.02, 0.03, 0.05, 1.0)
        self.render.setAntialias(AntialiasAttrib.MAuto)
        if patch_loader is not None:
            patch_loader(self.loader)
        if self.enable_fancy_rendering:
            self.render.setShaderAuto()
        if self.enable_fancy_rendering and simplepbr is not None and self.win is not None:
            simplepbr.init(enable_shadows=False, use_occlusion_maps=False, max_lights=8)
        self.filters = None

    def _setup_scene(self) -> None:
        fog = Fog("desk-fog")
        fog.setColor(0.04, 0.05, 0.08)
        fog.setExpDensity(0.0055)
        self.render.setFog(fog)

        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.26, 0.28, 0.32, 1.0))
        self.render.setLight(self.render.attachNewNode(ambient))

        sun = DirectionalLight("sun")
        sun.setColor(Vec4(0.96, 0.90, 0.78, 1.0))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(-34, -32, 0)
        self.render.setLight(sun_np)

        fill = DirectionalLight("fill")
        fill.setColor(Vec4(0.24, 0.30, 0.40, 1.0))
        fill_np = self.render.attachNewNode(fill)
        fill_np.setHpr(136, -18, 0)
        self.render.setLight(fill_np)

        finish_glow = PointLight("finish-glow")
        finish_glow.setColor(Vec4(0.32, 0.58, 0.82, 1.0))
        finish_point = ramp_surface_point(self.sim_config, self.finish_line_distance)
        self.finish_glow_np = self.render.attachNewNode(finish_glow)
        self.finish_glow_np.setPos(finish_point.x, finish_point.y, finish_point.z + 3.2)
        self.render.setLight(self.finish_glow_np)

    def _build_world(self) -> None:
        self.scene_root = self.render.attachNewNode("scene-root")
        self._build_backdrop()
        self._build_ground()
        self._build_ramp()
        self._build_rails()
        self._build_boost_pads()
        self._build_finish_gate()
        self._build_obstacles()
        self._build_start_zone()
        self._build_marble()
        self._setup_camera()

    def _build_backdrop(self) -> None:
        for index, (width, height, x, y, z, color, h) in enumerate(
            (
                (52.0, 20.0, self.sim_config.length + 16.0, 0.0, 9.0, Vec4(0.07, 0.09, 0.13, 1.0), 90.0),
                (34.0, 10.0, self.sim_config.length + 10.0, -9.2, 4.2, Vec4(0.06, 0.08, 0.11, 1.0), 115.0),
                (34.0, 10.0, self.sim_config.length + 10.0, 9.2, 4.2, Vec4(0.06, 0.08, 0.11, 1.0), 65.0),
            )
        ):
            cm = CardMaker(f"backdrop-{index}")
            cm.setFrame(-width * 0.5, width * 0.5, -height * 0.5, height * 0.5)
            card = self.scene_root.attachNewNode(cm.generate())
            card.setColor(color)
            card.setTransparency(False)
            card.setTexture(self._get_texture("backdrop"))
            card.setPos(x, y, z)
            card.setH(h)
            card.setTwoSided(True)

        for side in (-1, 1):
            horizon = self.scene_root.attachNewNode(f"horizon-{side}")
            self._attach_centered_box(
                horizon,
                size=Vec3(12.0, 0.12, 2.2),
                color=Vec4(0.08, 0.10, 0.16, 1.0),
                style="panel",
            )
            horizon.setPos(self.sim_config.length + 7.5, side * 7.8, 1.2)
            horizon.setH(90.0 - side * 24.0)

        aura = self.scene_root.attachNewNode("finish-aura")
        self._attach_centered_box(aura, size=Vec3(2.8, 4.8, 0.04), color=Vec4(0.15, 0.17, 0.20, 1.0))
        aura.setPos(self.sim_config.length - 1.2, 0.0, 0.08)

    def _build_ground(self) -> None:
        return

    def _build_ramp(self) -> None:
        for index, (segment_np, _, segment) in enumerate(self.simulation.ramp_nodes):
            center_distance = segment.start_distance + segment.length * 0.5
            lane_width = lane_surface_width(self.sim_config, center_distance)
            ramp = self.scene_root.attachNewNode(f"ramp-{index}")
            ramp.setHpr(segment.heading_deg, segment.bank_deg, self.sim_config.angle_deg)
            ramp.setPos(segment_np.getPos())

            surface = ramp.attachNewNode("surface")
            self._attach_centered_box(
                surface,
                size=Vec3(segment.length * 0.99, lane_width * 0.82, 0.028),
                color=Vec4(0.17, 0.18, 0.20, 1.0),
                style="track",
            )
            surface.setZ(self.sim_config.ramp_thickness * 0.46)

            lane = ramp.attachNewNode("lane")
            self._attach_centered_box(
                lane,
                size=Vec3(segment.length * 0.92, max(0.06, lane_width * 0.10), 0.014),
                color=Vec4(0.82, 0.84, 0.88, 1.0),
                style="screen",
            )
            lane.setZ(self.sim_config.ramp_thickness * 0.56)

            deck = ramp.attachNewNode("deck")
            self._attach_centered_box(
                deck,
                size=Vec3(segment.length * 0.90, lane_width * 0.42, 0.12),
                color=Vec4(0.08, 0.10, 0.12, 1.0),
                style="panel",
            )
            deck.setZ(-0.14)

            for side in (-1, 1):
                trim = ramp.attachNewNode(f"trim-{side}")
                self._attach_centered_box(
                    trim,
                    size=Vec3(segment.length * 0.96, 0.08, 0.10),
                    color=Vec4(0.24, 0.26, 0.30, 1.0),
                    style="screen",
                )
                trim.setPos(0.0, side * (lane_width * 0.41), self.sim_config.ramp_thickness * 0.43)

                skirt = ramp.attachNewNode(f"skirt-{side}")
                self._attach_centered_box(
                    skirt,
                    size=Vec3(segment.length * 0.92, 0.05, 0.18),
                    color=Vec4(0.10, 0.12, 0.15, 1.0),
                    style="panel",
                )
                skirt.setPos(0.0, side * (lane_width * 0.34), -0.04)

            support = ramp.attachNewNode("support")
            self._load_styled_model(
                support,
                "box",
                scale=Vec3(segment.length * 0.16, lane_width * 0.18, 0.64),
                color=Vec4(0.08, 0.10, 0.12, 1.0),
                pos=Vec3(0.0, 0.0, -0.70),
                style="panel",
            )

            underlight = support.attachNewNode("underlight")
            self._attach_centered_box(
                underlight,
                size=Vec3(segment.length * 0.10, lane_width * 0.36, 0.04),
                color=Vec4(0.12, 0.14, 0.18, 1.0),
                style="screen",
            )
            underlight.setPos(0.0, 0.0, -0.30)
        self._build_continuous_track_skin()

    def _sample_track_angles(self, distance: float) -> tuple[float, float]:
        segments = self.simulation.ramp_segments
        if not segments:
            return 0.0, 0.0
        segment_length = self.sim_config.length / len(segments)
        clamped = max(0.0, min(self.sim_config.length - 1e-6, distance))
        index = min(int(clamped / segment_length), len(segments) - 1)
        next_index = min(index + 1, len(segments) - 1)
        local_t = (clamped - segments[index].start_distance) / segment_length
        blend = local_t * local_t * (3.0 - 2.0 * local_t)
        heading = segments[index].heading_deg * (1.0 - blend) + segments[next_index].heading_deg * blend
        bank = segments[index].bank_deg * (1.0 - blend) + segments[next_index].bank_deg * blend
        return heading, bank

    def _build_continuous_track_skin(self) -> None:
        # Keep the continuity accents narrow so they do not z-fight with the main ramp mesh.
        sample_count = max(24, len(self.simulation.ramp_segments))
        slice_length = self.sim_config.length / sample_count
        for index in range(sample_count):
            distance = min(self.sim_config.length - slice_length * 0.5, (index + 0.5) * slice_length)
            heading_deg, bank_deg = self._sample_track_angles(distance)
            center = ramp_surface_point(self.sim_config, distance)
            normal = ramp_normal(self.sim_config, distance)
            lane_width = lane_surface_width(self.sim_config, distance)
            lane_offsets = lane_center_offsets(self.sim_config, distance)

            for lane_index, lane_offset in enumerate(lane_offsets):
                skin = self.scene_root.attachNewNode(f"track-skin-{index}-{lane_index}")
                skin.setPos(center + ramp_side(self.sim_config, distance) * lane_offset + normal * 0.016)
                skin.setHpr(heading_deg, bank_deg, self.sim_config.angle_deg)

                lane = skin.attachNewNode("lane-ribbon")
                self._attach_centered_box(
                    lane,
                    size=Vec3(slice_length * 0.96, max(0.05, lane_width * 0.09), 0.010),
                    color=Vec4(0.72, 0.74, 0.78, 1.0),
                    style="screen",
                )
                lane.setZ(0.004)

                for side in (-1, 1):
                    trim = skin.attachNewNode(f"trim-{side}")
                    self._attach_centered_box(
                        trim,
                        size=Vec3(slice_length * 0.98, 0.032, 0.030),
                        color=Vec4(0.38, 0.40, 0.44, 1.0),
                        style="screen",
                    )
                    trim.setPos(0.0, side * (lane_width * 0.418), 0.006)

            gap_width = split_gap_width(self.sim_config, distance)
            if gap_width > 0.0:
                gap_shadow = self.scene_root.attachNewNode(f"track-gap-{index}")
                gap_shadow.setPos(center + normal * -0.045)
                gap_shadow.setHpr(heading_deg, bank_deg, self.sim_config.angle_deg)
                self._attach_centered_box(
                    gap_shadow,
                    size=Vec3(slice_length * 0.92, gap_width * 0.92, 0.040),
                    color=Vec4(0.04, 0.05, 0.07, 1.0),
                    style="panel",
                )

    def _build_rails(self) -> None:
        for index, (rail_np, _, segment, lateral_offset) in enumerate(self.simulation.rail_nodes):
            rail = self.scene_root.attachNewNode(f"rail-{index}")
            rail.setHpr(segment.heading_deg, segment.bank_deg, self.sim_config.angle_deg)
            rail.setPos(rail_np.getPos())

            beam = rail.attachNewNode("beam")
            self._attach_centered_box(
                beam,
                size=Vec3(segment.length * 0.98, self.sim_config.rail_width * 0.88, self.sim_config.rail_height * 0.24),
                color=Vec4(0.72, 0.78, 0.88, 1.0),
                style="panel",
            )
            beam.setZ(self.sim_config.rail_height * 0.34)

            wall = rail.attachNewNode("wall")
            self._attach_centered_box(
                wall,
                size=Vec3(segment.length * 0.96, self.sim_config.rail_width * 0.26, self.sim_config.rail_height * 0.78),
                color=Vec4(0.10, 0.13, 0.18, 1.0),
                style="panel",
            )
            wall.setZ(self.sim_config.rail_height * 0.08)

            glow = rail.attachNewNode("glow")
            self._attach_centered_box(
                glow,
                size=Vec3(segment.length * 0.94, self.sim_config.rail_width * 0.18, self.sim_config.rail_height * 0.12),
                color=Vec4(0.58, 0.62, 0.70, 1.0),
                style="screen",
            )
            glow.setZ(self.sim_config.rail_height * 0.34)

            if index % 3 == 0:
                post_offsets = (0.0,)
            else:
                post_offsets = ()
            for post_offset in post_offsets:
                post = rail.attachNewNode("post")
                self._attach_centered_box(
                    post,
                    size=Vec3(0.08, 0.08, self.sim_config.rail_height * 1.04),
                    color=Vec4(0.18, 0.22, 0.28, 1.0),
                    style="panel",
                )
                post.setPos(post_offset, 0.0, 0.0)

    def _register_pulse(self, node: NodePath, *, amplitude: float, speed: float, phase: float = 0.0) -> None:
        self.pulse_nodes.append(
            PulseNode(
                node=node,
                base_color=Vec4(1.0, 1.0, 1.0, 1.0),
                amplitude=amplitude,
                speed=speed,
                phase=phase,
            )
        )

    def _section_for_distance(self, distance: float):
        for section in self.course_sections:
            if distance <= section.end_distance:
                return section
        return self.course_sections[-1]

    def _build_section_gates(self) -> None:
        for index, section in enumerate(self.course_sections[:-1]):
            distance = min(self.sim_config.length - 2.0, section.end_distance)
            segment = ramp_segment_at_distance(self.sim_config, distance)
            gate = self.scene_root.attachNewNode(f"section-gate-{index}")
            gate.setPos(ramp_surface_point(self.sim_config, distance))
            gate.setHpr(segment.heading_deg, segment.bank_deg, self.sim_config.angle_deg)

            for side in (-1, 1):
                post = gate.attachNewNode(f"post-{side}")
                self._attach_centered_box(
                    post,
                    size=Vec3(0.14, 0.14, 2.2),
                    color=Vec4(0.10, 0.12, 0.16, 1.0),
                    style="panel",
                )
                post.setPos(0.0, side * 1.55, 1.10)
                stripe = post.attachNewNode("stripe")
                self._attach_centered_box(
                    stripe,
                    size=Vec3(0.10, 0.10, 1.4),
                    color=Vec4(*section.color),
                    style="screen",
                )
                stripe.setZ(0.15)
                self._register_pulse(stripe, amplitude=0.30, speed=2.4, phase=index * 0.5 + side * 0.2)

            crown = gate.attachNewNode("crown")
            self._attach_centered_box(
                crown,
                size=Vec3(0.16, 3.3, 0.16),
                color=Vec4(0.08, 0.10, 0.14, 1.0),
                style="panel",
            )
            crown.setPos(0.0, 0.0, 2.32)
            beam = crown.attachNewNode("beam")
            self._attach_centered_box(
                beam,
                size=Vec3(0.10, 2.7, 0.08),
                color=Vec4(*section.color),
                style="screen",
            )
            beam.setPos(0.0, 0.0, 0.02)
            self._register_pulse(beam, amplitude=0.36, speed=2.0, phase=index * 0.62)

            side = ramp_side(self.sim_config, distance)
            accent_color = Vec4(*section.color)
            self._build_track_marker_wall(gate.getPos() + side * 3.8, segment.heading_deg + 90.0)
            self._build_track_marker_wall(gate.getPos() - side * 3.8, segment.heading_deg - 90.0)
            self._build_light_tower(gate.getPos() + side * 5.2 + Vec3(0.0, 0.0, 0.4), segment.heading_deg + 90.0, accent_color)
            self._build_light_tower(gate.getPos() - side * 5.2 + Vec3(0.0, 0.0, 0.4), segment.heading_deg - 90.0, accent_color)

    def _build_boost_pads(self) -> None:
        for index, section in enumerate(self.course_sections):
            config_pad = (
                self.sim_config.boost_pads[index]
                if index < len(self.sim_config.boost_pads)
                else None
            )
            distance = (
                min(self.sim_config.length - 1.8, config_pad.distance_along_ramp)
                if config_pad is not None
                else min(self.sim_config.length - 1.8, section.boost_distance)
            )
            segment = ramp_segment_at_distance(self.sim_config, distance)
            pad = self.scene_root.attachNewNode(f"boost-pad-{index}")
            pad.setPos(
                boost_pad_center_position(self.sim_config, config_pad)
                if config_pad is not None
                else ramp_surface_point(self.sim_config, distance) + ramp_normal(self.sim_config, distance) * 0.018
            )
            pad.setHpr(segment.heading_deg, segment.bank_deg, self.sim_config.angle_deg)

            frame = pad.attachNewNode("frame")
            self._attach_centered_box(
                frame,
                size=Vec3(
                    config_pad.length if config_pad is not None else 0.80,
                    config_pad.width if config_pad is not None else self.sim_config.width * 0.72,
                    0.018,
                ),
                color=Vec4(0.10, 0.12, 0.16, 1.0),
                style="panel",
            )
            glow = pad.attachNewNode("glow")
            self._attach_centered_box(
                glow,
                size=Vec3(
                    (config_pad.length * 0.78) if config_pad is not None else 0.56,
                    (config_pad.width * 0.78) if config_pad is not None else self.sim_config.width * 0.56,
                    0.016,
                ),
                color=Vec4(0.24, 0.70, 0.92, 1.0),
                style="screen",
            )
            glow.setZ(0.014)

            stripe = pad.attachNewNode("stripe")
            self._attach_centered_box(
                stripe,
                size=Vec3(
                    (config_pad.length * 0.68) if config_pad is not None else 0.56,
                    0.10,
                    0.012,
                ),
                color=Vec4(0.94, 0.96, 1.0, 1.0),
                style="screen",
            )
            stripe.setPos(0.0, 0.0, 0.022)

            self.boost_pads.append(
                BoostPadVisual(
                    distance=distance,
                    root=pad,
                    glow=glow,
                    phase=index * 0.7,
                )
            )

    def _build_finish_gate(self) -> None:
        gate = self.scene_root.attachNewNode("finish-gate")
        finish_distance = self.finish_line_distance
        finish_point = ramp_surface_point(self.sim_config, finish_distance)
        gate.setPos(finish_point)
        for side in (-1, 1):
            post = gate.attachNewNode(f"post-{side}")
            self._attach_centered_box(
                post,
                size=Vec3(0.16, 0.16, 2.8),
                color=Vec4(0.74, 0.78, 0.84, 1.0),
                style="panel",
            )
            post.setPos(0.0, side * 1.55, 1.4)

        top_bar = gate.attachNewNode("top-bar")
        self._attach_centered_box(top_bar, size=Vec3(0.18, 3.5, 0.18), color=Vec4(0.10, 0.12, 0.16, 1.0), style="panel")
        top_bar.setPos(0.0, 0.0, 2.92)

        stripe = gate.attachNewNode("stripe")
        self._attach_centered_box(stripe, size=Vec3(0.08, 2.6, 0.08), color=Vec4(0.24, 0.70, 0.92, 1.0), style="screen")
        stripe.setPos(0.0, 0.0, 2.92)
        self._register_pulse(stripe, amplitude=0.40, speed=2.6, phase=0.4)

        finish_ribbon = gate.attachNewNode("finish-ribbon")
        self._attach_centered_box(
            finish_ribbon,
            size=Vec3(0.06, self.sim_config.width * 0.96, 0.18),
            color=Vec4(0.90, 0.94, 0.98, 1.0),
            style="screen",
        )
        finish_ribbon.setPos(0.0, 0.0, 1.18)
        self._register_pulse(finish_ribbon, amplitude=0.32, speed=2.2, phase=0.0)

        for index, lateral in enumerate((-0.72, -0.36, 0.0, 0.36, 0.72)):
            marker = gate.attachNewNode(f"finish-marker-{index}")
            self._attach_centered_box(
                marker,
                size=Vec3(0.10, 0.22, 0.04),
                color=Vec4(0.18, 0.82, 0.96, 1.0) if index % 2 == 0 else Vec4(0.96, 0.96, 0.98, 1.0),
                style="screen",
            )
            marker.setPos(0.0, lateral, 0.10)
            self._register_pulse(marker, amplitude=0.22, speed=2.0, phase=index * 0.3)

        finish_pad = gate.attachNewNode("finish-pad")
        self._attach_centered_box(
            finish_pad,
            size=Vec3(0.92, self.sim_config.width * 0.98, 0.018),
            color=Vec4(0.14, 0.16, 0.18, 1.0),
            style="panel",
        )
        finish_pad.setPos(0.02, 0.0, 0.02)

        tile_length = 0.12
        row_offsets = (-0.07, 0.07)
        lateral_min = -self.sim_config.width * 0.42
        lateral_step = 0.24
        lateral_values: list[float] = []
        lateral = lateral_min
        while lateral <= self.sim_config.width * 0.42 + 1e-6:
            lateral_values.append(lateral)
            lateral += lateral_step
        for row_index, row_offset in enumerate(row_offsets):
            for column_index, lateral in enumerate(lateral_values):
                tile = gate.attachNewNode(f"finish-tile-{row_index}-{column_index}")
                self._attach_centered_box(
                    tile,
                    size=Vec3(tile_length, 0.20, 0.010),
                    color=Vec4(0.96, 0.96, 0.98, 1.0) if (row_index + column_index) % 2 == 0 else Vec4(0.10, 0.12, 0.14, 1.0),
                    depth_offset=1,
                )
                tile.setPos(row_offset, lateral, 0.034)

    def _build_obstacles(self) -> None:
        colors = {
            "block": Vec4(0.96, 0.42, 0.18, 1.0),
            "pencil": Vec4(0.18, 0.70, 0.94, 1.0),
            "eraser": Vec4(0.84, 0.40, 0.94, 1.0),
            "pen": Vec4(0.18, 0.70, 0.94, 1.0),
            "ruler": Vec4(0.96, 0.74, 0.20, 1.0),
            "pendulum": Vec4(0.96, 0.50, 0.22, 1.0),
            "sweeper": Vec4(0.22, 0.86, 0.96, 1.0),
            "fork-divider": Vec4(0.96, 0.82, 0.24, 1.0),
        }
        for obstacle in self.sim_config.obstacles:
            self._build_obstacle(obstacle, colors.get(obstacle.kind, Vec4(0.45, 0.45, 0.45, 1.0)))

    def _build_obstacle(self, obstacle: GuideObstacle, color: Vec4) -> None:
        marker = self._build_obstacle_marker(obstacle, color)
        obstacle_np = self.scene_root.attachNewNode("obstacle")
        arm = self._attach_obstacle_visual(obstacle_np, obstacle, color)
        segment = ramp_segment_at_distance(self.sim_config, obstacle.distance_along_ramp)
        if obstacle.motion_kind == "pendulum":
            surface = ramp_surface_point(self.sim_config, obstacle.distance_along_ramp)
            side = ramp_side(self.sim_config, obstacle.distance_along_ramp)
            normal = ramp_normal(self.sim_config, obstacle.distance_along_ramp)
            pivot_height = obstacle.pivot_height if obstacle.pivot_height > 0.0 else 1.05
            obstacle_np.setPos(surface + side * obstacle.lateral_offset + normal * pivot_height)
            obstacle_np.setHpr(segment.heading_deg, segment.bank_deg, self.sim_config.angle_deg)
        else:
            center, heading_deg = obstacle_pose(self.sim_config, obstacle, 0.0)
            obstacle_np.setHpr(segment.heading_deg + heading_deg, segment.bank_deg, self.sim_config.angle_deg)
            obstacle_np.setPos(center)
        if obstacle.motion_kind != "static":
            self.moving_obstacle_visuals.append(MovingObstacleVisual(obstacle=obstacle, root=obstacle_np, marker=marker, arm=arm))

    def _build_obstacle_marker(self, obstacle: GuideObstacle, color: Vec4) -> NodePath:
        marker_distance = max(0.9, obstacle.distance_along_ramp - max(0.9, obstacle.length * 1.05))
        segment = ramp_segment_at_distance(self.sim_config, marker_distance)
        marker = self.scene_root.attachNewNode("obstacle-marker")
        marker.setPos(
            ramp_surface_point(self.sim_config, marker_distance)
            + ramp_side(self.sim_config, marker_distance) * obstacle.lateral_offset
            + ramp_normal(self.sim_config, marker_distance) * 0.055
        )
        marker.setHpr(segment.heading_deg, segment.bank_deg, self.sim_config.angle_deg)

        plate_width = min(self.sim_config.width * 0.78, obstacle.width + 0.42)
        self._attach_centered_box(
            marker,
            size=Vec3(0.60, plate_width + 0.08, 0.024),
            color=Vec4(0.08, 0.10, 0.13, 1.0),
            style="panel",
            depth_offset=1,
        )
        self._attach_centered_box(
            marker,
            size=Vec3(0.52, plate_width, 0.012),
            color=Vec4(color.x * 0.95, color.y * 0.95, color.z * 0.95, 1.0),
            style="screen",
            depth_offset=2,
        )

        center_line = marker.attachNewNode("center-line")
        self._attach_centered_box(
            center_line,
            size=Vec3(0.36, min(plate_width * 0.34, 0.14), 0.014),
            color=Vec4(0.95, 0.96, 0.98, 1.0),
            style="screen",
            depth_offset=3,
        )
        center_line.setZ(0.018)
        return marker

    def _attach_obstacle_visual(self, parent: NodePath, obstacle: GuideObstacle, color: Vec4) -> NodePath | None:
        if obstacle.kind == "fork-divider":
            self._attach_centered_box(parent, size=Vec3(obstacle.length, obstacle.width, obstacle.height), color=color, style="panel")
            fin = parent.attachNewNode("divider-fin")
            self._attach_centered_box(
                fin,
                size=Vec3(obstacle.length * 0.74, obstacle.width * 0.34, obstacle.height * 1.55),
                color=Vec4(0.18, 0.20, 0.24, 1.0),
                style="panel",
            )
            fin.setPos(0.0, 0.0, obstacle.height * 0.52)
            return None

        if obstacle.kind == "pendulum":
            pivot_height = obstacle.pivot_height if obstacle.pivot_height > 0.0 else 1.05
            rod_length = max(obstacle.height * 0.5 + 0.16, pivot_height - obstacle.height * 0.5)
            arm = parent.attachNewNode("pendulum-arm")
            rod = arm.attachNewNode("pendulum-rod")
            self._attach_centered_box(
                rod,
                size=Vec3(0.08, 0.08, rod_length),
                color=Vec4(0.16, 0.18, 0.22, 1.0),
                style="panel",
            )
            rod.setPos(0.0, 0.0, -rod_length * 0.5)
            bob = arm.attachNewNode("pendulum-bob")
            self._attach_centered_box(bob, size=Vec3(obstacle.length, obstacle.width, obstacle.height), color=color, style="panel")
            bob.setZ(-rod_length)
            return arm

        if obstacle.kind == "sweeper":
            boom = parent.attachNewNode("sweeper-boom")
            self._attach_centered_box(
                boom,
                size=Vec3(obstacle.length * 1.08, obstacle.width, obstacle.height),
                color=color,
                style="panel",
            )
            boom.setZ(obstacle.height * 0.5)
            hub = parent.attachNewNode("sweeper-hub")
            self._attach_centered_box(
                hub,
                size=Vec3(0.18, 0.18, obstacle.height * 1.8),
                color=Vec4(0.12, 0.14, 0.18, 1.0),
                style="panel",
            )
            hub.setZ(obstacle.height * 0.36)
            return None

        size = Vec3(obstacle.length, obstacle.width, obstacle.height)
        self._attach_centered_box(parent, size=size, color=color, style="panel")

        inset = parent.attachNewNode("inset")
        self._attach_centered_box(
            inset,
            size=Vec3(size.x * 0.82, size.y * 0.58, size.z * 0.44),
            color=Vec4(0.18, 0.10, 0.08, 1.0),
            style="panel",
        )
        inset.setPos(0.0, 0.0, size.z * 0.02)

        cap = parent.attachNewNode("cap")
        self._attach_centered_box(
            cap,
            size=Vec3(size.x * 0.88, size.y * 0.88, size.z * 0.16),
            color=Vec4(0.95, 0.96, 0.98, 1.0),
            style="screen",
        )
        cap.setPos(0.0, 0.0, size.z * 0.42)

        stripe = parent.attachNewNode("stripe")
        self._attach_centered_box(
            stripe,
            size=Vec3(size.x * 0.18, size.y * 0.92, size.z * 0.18),
            color=Vec4(0.20, 0.12, 0.08, 1.0),
            style="panel",
        )
        stripe.setPos(0.0, 0.0, size.z * 0.18)
        return None

    def _build_desk_props(self) -> None:
        self._build_arena_shell()
        self._build_monitor_wall(Vec3(self.sim_config.length * 0.12, -9.2, 0.0), 12.0)
        self._build_monitor_wall(Vec3(self.sim_config.length * 0.58, 9.4, 0.0), 168.0)
        self._build_service_crates(Vec3(self.sim_config.length * 0.20, 9.1, 0.0), -24.0)
        self._build_service_crates(Vec3(self.sim_config.length * 0.72, -9.0, 0.0), 18.0)
        self._build_notebook(Vec3(self.sim_config.length * 0.35, 10.3, 0.25), -16.0)
        self._build_sheet(Vec3(self.sim_config.length * 0.43, -10.4, 0.10), Vec3(4.2, 3.0, 0.02), 14.0)
        self._build_sheet(Vec3(self.sim_config.length * 0.78, 10.0, 0.10), Vec3(3.6, 2.8, 0.02), -18.0)
        self._build_light_tower(Vec3(self.sim_config.length * 0.18, -10.2, 0.0), 8.0, Vec4(0.18, 0.70, 0.96, 1.0))
        self._build_light_tower(Vec3(self.sim_config.length * 0.82, 10.2, 0.0), 186.0, Vec4(0.96, 0.42, 0.18, 1.0))
        self._build_stationery_cup(Vec3(self.sim_config.length * 0.64, -10.6, 0.0))
        self._build_lamp(Vec3(self.sim_config.length + 5.8, -6.4, 0.0))
        self._build_bicycle_prop(Vec3(self.sim_config.length + 7.4, 7.0, 0.1), 208.0)

    def _build_start_zone(self) -> None:
        pad = self.scene_root.attachNewNode("start-pad")
        self._attach_centered_box(
            pad,
            size=Vec3(2.2, 1.4, 0.05),
            color=Vec4(0.12, 0.14, 0.18, 1.0),
            style="panel",
        )
        pad.setPos(0.8, 0.0, self.sim_config.height - 0.42)

        line = pad.attachNewNode("line")
        self._attach_centered_box(
            line,
            size=Vec3(1.8, 0.10, 0.02),
            color=Vec4(0.76, 0.80, 0.86, 1.0),
            style="screen",
        )
        line.setZ(0.04)

        glow = pad.attachNewNode("glow")
        self._attach_centered_box(
            glow,
            size=Vec3(1.92, 0.72, 0.018),
            color=Vec4(0.18, 0.74, 0.96, 1.0),
            style="screen",
            depth_offset=2,
        )
        glow.setPos(0.0, 0.0, 0.028)
        self._register_pulse(glow, amplitude=0.28, speed=2.4, phase=0.0)

        for index, x_offset in enumerate((-0.62, 0.0, 0.62)):
            chevron = pad.attachNewNode(f"chevron-{index}")
            self._attach_centered_box(
                chevron,
                size=Vec3(0.28, 0.22, 0.014),
                color=Vec4(0.94, 0.96, 1.0, 1.0),
                style="screen",
                depth_offset=3,
            )
            chevron.setPos(x_offset, 0.0, 0.042)
            chevron.setH(45.0)
            self._register_pulse(chevron, amplitude=0.22, speed=3.0, phase=index * 0.24)

        for side in (-1, 1):
            beacon = pad.attachNewNode(f"start-beacon-{side}")
            self._attach_centered_box(
                beacon,
                size=Vec3(0.12, 0.12, 0.52),
                color=Vec4(0.24, 0.82, 0.98, 1.0),
                style="screen",
                depth_offset=2,
            )
            beacon.setPos(0.0, side * 0.82, 0.26)
            self._register_pulse(beacon, amplitude=0.30, speed=2.7, phase=0.2 if side > 0 else 0.6)

    def _build_sheet(self, pos: Vec3, size: Vec3, heading: float) -> None:
        sheet = self.scene_root.attachNewNode("sheet")
        self._attach_centered_box(sheet, size=size, color=Vec4(0.97, 0.96, 0.92, 1.0))
        sheet.setPos(pos)
        sheet.setH(heading)
        for line_index in range(6):
            line = sheet.attachNewNode(f"line-{line_index}")
            self._attach_centered_box(line, size=Vec3(size.x * 0.88, 0.012, 0.004), color=Vec4(0.78, 0.86, 0.95, 1.0))
            line.setPos(0.0, -size.y * 0.26 + line_index * 0.26, 0.018)

    def _build_monitor_wall(self, pos: Vec3, heading: float) -> None:
        bank = self.scene_root.attachNewNode("monitor-wall")
        bank.setPos(pos)
        bank.setH(heading)
        base = bank.attachNewNode("base")
        self._attach_centered_box(base, size=Vec3(3.8, 1.1, 0.24), color=Vec4(0.10, 0.11, 0.14, 1.0))
        base.setZ(0.12)
        for index in range(3):
            panel = bank.attachNewNode(f"panel-{index}")
            self._attach_centered_box(panel, size=Vec3(1.08, 0.16, 1.22), color=Vec4(0.05, 0.06, 0.08, 1.0))
            panel.setPos(-1.18 + index * 1.18, 0.0, 1.02)
            glow = panel.attachNewNode("glow")
            self._attach_centered_box(
                glow,
                size=Vec3(0.92, 0.03, 1.02),
                color=Vec4(0.14 + index * 0.08, 0.28 + index * 0.12, 0.42 + index * 0.10, 1.0),
            )
            glow.setPos(0.0, -0.055, 0.0)
            self._register_pulse(glow, amplitude=0.26, speed=1.9, phase=index * 0.45)
        mast = bank.attachNewNode("mast")
        self._attach_centered_box(mast, size=Vec3(0.18, 0.18, 1.16), color=Vec4(0.16, 0.17, 0.20, 1.0))
        mast.setPos(0.0, 0.0, 0.6)

    def _build_arena_shell(self) -> None:
        for side in (-1, 1):
            wall = self.scene_root.attachNewNode(f"arena-wall-{side}")
            self._attach_centered_box(
                wall,
                size=Vec3(self.sim_config.length + 10.0, 0.40, 1.3),
                color=Vec4(0.08, 0.10, 0.13, 1.0),
                style="panel",
            )
            wall.setPos(self.sim_config.length * 0.5, side * 8.1, 0.72)

            cap = wall.attachNewNode("cap")
            self._attach_centered_box(
                cap,
                size=Vec3(self.sim_config.length + 8.2, 0.16, 0.10),
                color=Vec4(0.74, 0.80, 0.88, 1.0),
                style="screen",
            )
            cap.setPos(0.0, -side * 0.12, 0.74)

            light_strip = wall.attachNewNode("light-strip")
            self._attach_centered_box(
                light_strip,
                size=Vec3(self.sim_config.length + 7.4, 0.05, 0.12),
                color=Vec4(0.18 if side < 0 else 0.92, 0.66, 0.94 if side < 0 else 0.22, 1.0),
                style="screen",
            )
            light_strip.setPos(0.0, -side * 0.19, 0.04)
            self._register_pulse(light_strip, amplitude=0.24, speed=1.7, phase=side * 0.5)

            for index in range(5):
                pylon = self.scene_root.attachNewNode(f"arena-pylon-{side}-{index}")
                self._attach_centered_box(
                    pylon,
                    size=Vec3(0.30, 0.30, 3.9),
                    color=Vec4(0.10, 0.12, 0.16, 1.0),
                    style="panel",
                )
                pylon.setPos(2.8 + index * 5.0, side * 8.8, 1.95)
                beacon = pylon.attachNewNode("beacon")
                self._attach_centered_box(
                    beacon,
                    size=Vec3(0.54, 0.16, 0.54),
                    color=Vec4(0.20 if side < 0 else 0.96, 0.66 if side < 0 else 0.44, 0.96 if side < 0 else 0.18, 1.0),
                    style="screen",
                )
                beacon.setPos(0.0, -side * 0.10, 1.20)
                self._register_pulse(beacon, amplitude=0.30, speed=2.1, phase=index * 0.3 + side * 0.4)

    def _build_track_marker_wall(self, pos: Vec3, heading: float) -> None:
        wall = self.scene_root.attachNewNode("track-marker-wall")
        wall.setPos(pos)
        wall.setH(heading)
        self._attach_centered_box(wall, size=Vec3(3.4, 0.26, 1.9), color=Vec4(0.08, 0.10, 0.13, 1.0))
        for index, color in enumerate((Vec4(0.18, 0.68, 0.94, 1.0), Vec4(0.96, 0.42, 0.18, 1.0), Vec4(0.86, 0.88, 0.92, 1.0))):
            stripe = wall.attachNewNode(f"stripe-{index}")
            self._attach_centered_box(stripe, size=Vec3(2.8, 0.04, 0.14), color=color)
            stripe.setPos(0.0, -0.12, 0.46 - index * 0.34)
            self._register_pulse(stripe, amplitude=0.20, speed=2.8, phase=index * 0.33)

    def _build_light_tower(self, pos: Vec3, heading: float, color: Vec4) -> None:
        tower = self.scene_root.attachNewNode("light-tower")
        tower.setPos(pos)
        tower.setH(heading)
        self._attach_centered_box(tower, size=Vec3(0.9, 0.9, 0.12), color=Vec4(0.10, 0.11, 0.14, 1.0))
        stem = tower.attachNewNode("stem")
        self._attach_centered_box(stem, size=Vec3(0.16, 0.16, 2.8), color=Vec4(0.20, 0.22, 0.26, 1.0))
        stem.setZ(1.4)
        head = tower.attachNewNode("head")
        self._attach_centered_box(head, size=Vec3(1.2, 0.4, 0.34), color=Vec4(0.08, 0.10, 0.12, 1.0))
        head.setPos(0.28, 0.0, 2.72)
        beam = head.attachNewNode("beam")
        self._attach_centered_box(beam, size=Vec3(0.96, 0.10, 0.18), color=color)
        beam.setPos(0.08, -0.14, 0.0)
        self._register_pulse(beam, amplitude=0.34, speed=2.2, phase=heading * 0.01)

    def _build_service_crates(self, pos: Vec3, heading: float) -> None:
        cluster = self.scene_root.attachNewNode("crate-cluster")
        cluster.setPos(pos)
        cluster.setH(heading)
        for index, (offset, size, color) in enumerate(
            (
                (Vec3(-0.9, 0.0, 0.0), Vec3(0.9, 0.9, 0.6), Vec4(0.12, 0.14, 0.18, 1.0)),
                (Vec3(0.0, 0.5, 0.0), Vec3(0.7, 0.7, 0.42), Vec4(0.20, 0.22, 0.28, 1.0)),
                (Vec3(0.9, -0.1, 0.0), Vec3(0.6, 0.6, 0.36), Vec4(0.94, 0.42, 0.18, 1.0)),
            )
        ):
            crate = cluster.attachNewNode(f"crate-{index}")
            self._attach_centered_box(crate, size=size, color=color)
            crate.setPos(offset + Vec3(0.0, 0.0, size.z * 0.5))

    def _build_notebook(self, pos: Vec3, heading: float) -> None:
        notebook = self.scene_root.attachNewNode("notebook")
        self._attach_centered_box(notebook, size=Vec3(3.6, 2.3, 0.10), color=Vec4(0.12, 0.24, 0.52, 1.0))
        notebook.setPos(pos)
        notebook.setH(heading)
        for coil_index in range(6):
            coil = notebook.attachNewNode(f"coil-{coil_index}")
            self._load_styled_model(
                coil,
                "cylinder",
                scale=Vec3(0.10, 0.05, 0.05),
                color=Vec4(0.84, 0.86, 0.90, 1.0),
                pos=Vec3(-1.3 + coil_index * 0.52, 1.10, 0.08),
            )
            coil.setP(90)

    def _build_lamp(self, pos: Vec3) -> None:
        lamp = self.scene_root.attachNewNode("lamp")
        lamp.setPos(pos)
        self._attach_centered_box(lamp, size=Vec3(1.8, 1.1, 0.08), color=Vec4(0.12, 0.13, 0.16, 1.0))
        stem = lamp.attachNewNode("stem")
        self._load_styled_model(stem, "cylinder", scale=Vec3(1.4, 0.09, 0.09), color=Vec4(0.26, 0.28, 0.32, 1.0), pos=Vec3(0.0, 0.0, 1.28))
        stem.setP(90)
        shade = lamp.attachNewNode("shade")
        self._load_styled_model(shade, "cube", scale=Vec3(0.74, 0.62, 0.52), color=Vec4(0.92, 0.76, 0.42, 1.0), pos=Vec3(0.42, 0.0, 2.2))
        shade.setH(-24)
        bulb = lamp.attachNewNode("bulb")
        self._load_styled_model(bulb, "sphere", scale=Vec3(0.18, 0.18, 0.18), color=Vec4(1.0, 0.93, 0.72, 1.0), pos=Vec3(0.52, 0.0, 2.0))

    def _build_stationery_cup(self, pos: Vec3) -> None:
        cup = self.scene_root.attachNewNode("cup")
        cup.setPos(pos)
        self._load_styled_model(cup, "cylinder", scale=Vec3(0.92, 0.42, 0.42), color=Vec4(0.80, 0.83, 0.88, 1.0), pos=Vec3(0.0, 0.0, 0.74))
        cup.setP(90)
        for index, color in enumerate((Vec4(0.95, 0.79, 0.20, 1.0), Vec4(0.22, 0.50, 0.88, 1.0), Vec4(0.96, 0.50, 0.68, 1.0))):
            tool = cup.attachNewNode(f"tool-{index}")
            self._load_styled_model(
                tool,
                "cylinder",
                scale=Vec3(1.3, 0.05, 0.05),
                color=color,
                pos=Vec3(-0.18 + index * 0.18, 0.0, 1.3),
            )
            tool.setP(84 - index * 4)
            tool.setR(-6 + index * 7)

    def _build_bicycle_prop(self, pos: Vec3, heading: float) -> None:
        frame_path = resolve_model_path("frame")
        wheel_path = resolve_model_path("wheel")
        if frame_path is None or wheel_path is None:
            return
        bike = self.scene_root.attachNewNode("bike")
        bike.setPos(pos)
        bike.setH(heading)
        bike.setScale(0.006)

        frame = self.loader.loadModel(frame_path)
        frame.reparentTo(bike)
        frame.setColor(0.24, 0.50, 0.96, 1.0)
        frame.setTextureOff(1)

        rear = self.loader.loadModel(wheel_path)
        rear.reparentTo(bike)
        rear.setColor(0.12, 0.14, 0.18, 1.0)
        rear.setTextureOff(1)
        self.prop_spinners.append(rear)

        front = self.loader.loadModel(wheel_path)
        front.reparentTo(bike)
        front.setPos(130, 0, 0)
        front.setColor(0.12, 0.14, 0.18, 1.0)
        front.setTextureOff(1)
        self.prop_spinners.append(front)

    def _build_marble(self) -> None:
        self.marble_root = self.scene_root.attachNewNode("marble-root")
        self.marble_root.setPos(self.simulation.marble_np.getPos())

        self._load_styled_model(
            self.marble_root,
            "sphere",
            scale=Vec3(self.sim_config.marble_radius),
            color=Vec4(0.94, 0.97, 1.0, 1.0),
        )
        stripe = self.marble_root.attachNewNode("stripe")
        self._load_styled_model(
            stripe,
            "sphere",
            scale=Vec3(self.sim_config.marble_radius * 0.72),
            color=Vec4(0.14, 0.72, 0.88, 1.0),
            pos=Vec3(self.sim_config.marble_radius * 0.18, 0.0, 0.0),
        )
        marker = self.marble_root.attachNewNode("marker")
        self._load_styled_model(
            marker,
            "sphere",
            scale=Vec3(self.sim_config.marble_radius * 0.22),
            color=Vec4(0.08, 0.10, 0.14, 1.0),
            pos=Vec3(self.sim_config.marble_radius, 0.0, 0.0),
        )
        for index in range(2):
            line = self.marble_root.attachNewNode(f"speed-line-{index}")
            self._attach_centered_box(
                line,
                size=Vec3(self.sim_config.marble_radius * 1.5, 0.016, 0.016),
                color=Vec4(0.76, 0.80, 0.86, 1.0),
            )
            line.setPos(
                -self.sim_config.marble_radius * (1.1 + index * 0.45),
                0.0,
                -0.02 + index * 0.012,
            )
            self.speed_lines.append(line)
        self._build_ghost_marker()

    def _clear_bot_marbles(self) -> None:
        for bot in self.bot_marbles:
            if hasattr(self, "simulation") and self.simulation is not None:
                self.simulation.world.removeRigidBody(bot.body)
            if not bot.body_np.isEmpty():
                bot.body_np.removeNode()
            if not bot.visual_root.isEmpty():
                bot.visual_root.removeNode()
        self.bot_marbles = []

    def _spawn_bot_marbles(self) -> None:
        self._clear_bot_marbles()
        if self.active_mode != self.MODE_BOT_RACE:
            return
        bot_specs = [
            ("BOT-1", -0.24, -0.12, Vec4(0.96, 0.44, 0.22, 1.0)),
            ("BOT-2", 0.24, 0.10, Vec4(0.22, 0.78, 0.96, 1.0)),
        ]
        for index, (name, lane_offset, speed_bias, color) in enumerate(bot_specs):
            body = BulletRigidBodyNode(f"bot-marble-{index}")
            body.addShape(BulletSphereShape(self.sim_config.marble_radius))
            body.setMass(self.sim_config.marble_mass)
            body.setFriction(self.sim_config.marble_friction)
            body.setRestitution(self.sim_config.restitution)
            body.setLinearDamping(0.012)
            body.setAngularDamping(0.014)
            body.setCcdMotionThreshold(1e-7)
            body.setCcdSweptSphereRadius(self.sim_config.marble_radius * 0.98)
            body_np = self.simulation.root.attachNewNode(body)
            self.simulation.world.attachRigidBody(body)

            visual_root = self.scene_root.attachNewNode(f"bot-visual-{index}")
            self._load_styled_model(
                visual_root,
                "sphere",
                scale=Vec3(self.sim_config.marble_radius * 0.96),
                color=color,
            )
            stripe = visual_root.attachNewNode("stripe")
            self._load_styled_model(
                stripe,
                "sphere",
                scale=Vec3(self.sim_config.marble_radius * 0.64),
                color=Vec4(0.96, 0.98, 1.0, 1.0),
                pos=Vec3(self.sim_config.marble_radius * 0.16, 0.0, 0.0),
            )
            marker = visual_root.attachNewNode("marker")
            self._load_styled_model(
                marker,
                "sphere",
                scale=Vec3(self.sim_config.marble_radius * 0.18),
                color=Vec4(0.10, 0.12, 0.16, 1.0),
                pos=Vec3(self.sim_config.marble_radius * 0.94, 0.0, 0.0),
            )
            self.bot_marbles.append(
                BotMarble(
                    name=name,
                    body_np=body_np,
                    body=body,
                    visual_root=visual_root,
                    lane_offset=lane_offset,
                    speed_bias=speed_bias,
                )
            )
        self._reset_bot_marbles()

    def _reset_bot_body(self, bot: BotMarble, *, distance: float) -> None:
        spawn_distance = max(0.0, min(self.sim_config.length - 1.0, distance))
        surface = ramp_surface_point(self.sim_config, spawn_distance)
        side = ramp_side(self.sim_config, spawn_distance)
        normal = ramp_normal(self.sim_config, spawn_distance)
        start_pos = surface + side * bot.lane_offset + normal * (self.sim_config.marble_radius + 0.01)
        bot.body_np.setPos(start_pos)
        bot.body_np.setQuat(NodePath("identity").getQuat())
        bot.body.setLinearVelocity(Vec3(0.0, 0.0, 0.0))
        bot.body.setAngularVelocity(Vec3(0.0, 0.0, 0.0))
        bot.body.clearForces()
        bot.progress = spawn_distance
        bot.finish_time = None
        bot.finish_order = None
        bot.off_track_timer = 0.0
        bot.fall_timer = 0.0
        bot.reset_warning_second = None
        bot.previous_position = Vec3(start_pos)
        bot.previous_progress = spawn_distance
        bot.active_boost_pad_index = None
        bot.last_boost_pad_index = None

    def _reset_bot_marbles(self) -> None:
        spawn_distances = (0.0, 0.0)
        for index, bot in enumerate(self.bot_marbles):
            self._reset_bot_body(bot, distance=spawn_distances[min(index, len(spawn_distances) - 1)])
        self._update_bot_visuals()

    def _bot_lane_target_offset(self, distance: float, preferred_offset: float) -> float:
        lane_offsets = lane_center_offsets(self.sim_config, distance)
        if len(lane_offsets) == 1:
            return preferred_offset
        preferred_sign = -1.0 if preferred_offset < 0.0 else 1.0
        matching_lanes = [lane_offset for lane_offset in lane_offsets if lane_offset * preferred_sign > 0.0]
        if matching_lanes:
            return min(matching_lanes, key=lambda lane_offset: abs(lane_offset - preferred_offset))
        return min(lane_offsets, key=lambda lane_offset: abs(lane_offset - preferred_offset))

    def _settle_start_grid(self, *, settle_steps: int = 24, dt: float = 1.0 / 960.0) -> None:
        if settle_steps <= 0 or dt <= 0.0:
            return
        for _ in range(settle_steps):
            self.simulation.world.doPhysics(dt, 1, dt)
        self.simulation.marble_body.setLinearVelocity(Vec3(0.0, 0.0, 0.0))
        self.simulation.marble_body.setAngularVelocity(Vec3(0.0, 0.0, 0.0))
        self.simulation.marble_body.clearForces()
        for bot in self.bot_marbles:
            bot.body.setLinearVelocity(Vec3(0.0, 0.0, 0.0))
            bot.body.setAngularVelocity(Vec3(0.0, 0.0, 0.0))
            bot.body.clearForces()
            bot.previous_position = Vec3(bot.body_np.getPos())
            bot.previous_progress = bot.progress
        self._update_bot_visuals()

    def _update_bot_controllers(self, dt: float) -> None:
        if self.active_mode != self.MODE_BOT_RACE:
            return
        mass = self.sim_config.marble_mass
        for bot in self.bot_marbles:
            if bot.finish_time is not None:
                continue
            position = bot.body_np.getPos()
            bot.previous_position = Vec3(position)
            bot.previous_progress = bot.progress
            near_progress = path_distance_for_position_near(self.sim_config, position, bot.progress)
            absolute_progress = path_distance_for_position(self.sim_config, position)
            progress_candidate = near_progress
            if abs(absolute_progress - near_progress) <= 4.0:
                progress_candidate = max(progress_candidate, absolute_progress)
            bot.progress = max(bot.progress, progress_candidate)
            velocity = bot.body.getLinearVelocity()
            lookahead_distance = min(self.sim_config.length, bot.progress + 1.4)
            target_tangent = ramp_tangent(self.sim_config, lookahead_distance)
            if target_tangent.length_squared() <= 1e-9:
                continue
            target_tangent.normalize()
            target_side = ramp_side(self.sim_config, lookahead_distance)
            target_normal = ramp_normal(self.sim_config, lookahead_distance)
            target_lane_offset = self._bot_lane_target_offset(lookahead_distance, bot.lane_offset)
            target_point = (
                ramp_surface_point(self.sim_config, lookahead_distance)
                + target_side * target_lane_offset
                + target_normal * (self.sim_config.marble_radius + 0.01)
            )
            relative = target_point - position
            forward_speed = velocity.dot(target_tangent)
            lateral_speed = velocity.dot(target_side)
            vertical_speed = velocity.dot(target_normal)
            target_speed = 6.2 + bot.speed_bias + min(2.6, bot.progress / max(self.sim_config.length, 1e-6) * 2.8)
            forward_force = max(0.0, target_speed - forward_speed) * mass * 4.0
            lateral_force = (relative.dot(target_side) * 7.4 - lateral_speed * 2.2) * mass
            bot.body.applyCentralForce(target_tangent * forward_force + target_side * lateral_force)
            active_boost = self._boost_pad_at_position(position, airborne=False)
            bot.active_boost_pad_index = active_boost[0] if active_boost is not None else None
            if active_boost is not None:
                pad_index, boost_pad = active_boost
                bot.body.setLinearVelocity(velocity + target_tangent * (boost_pad.acceleration * dt))
                if pad_index != bot.last_boost_pad_index and pad_index < len(self.boost_pads):
                    self._pulse_boost_pad(self.boost_pads[pad_index])
            bot.last_boost_pad_index = bot.active_boost_pad_index
            if relative.dot(target_normal) < -0.02 or vertical_speed > 0.55:
                bot.body.applyCentralForce(-target_normal * mass * 2.2)

    def _update_bot_race_state(self) -> None:
        if self.active_mode != self.MODE_BOT_RACE:
            return
        for bot in self.bot_marbles:
            if bot.finish_time is not None:
                continue
            position = bot.body_np.getPos()
            absolute_progress = path_distance_for_position(self.sim_config, position)
            bot.progress = max(
                bot.progress,
                absolute_progress,
                path_distance_for_position_near(self.sim_config, position, bot.progress),
            )
            if bot.previous_position is not None:
                previous_state = type("BotState", (), {})()
                previous_state.position = bot.previous_position
                previous_state.path_distance = bot.previous_progress
                current_state = type("BotState", (), {})()
                current_state.position = position
                current_state.path_distance = bot.progress
                if self._should_finish_run(previous_state, current_state):
                    bot.finish_time = self.simulation.time
                    bot.finish_order = self._claim_finish_order()
                    continue

    def _update_bot_visuals(self) -> None:
        for bot in self.bot_marbles:
            if bot.visual_root.isEmpty():
                continue
            bot.visual_root.setPos(bot.body_np.getPos())
            bot.visual_root.setQuat(bot.body_np.getQuat())

    def _build_ghost_marker(self) -> None:
        if self.ghost_playback is not None and not self.ghost_playback.root.isEmpty():
            self.ghost_playback.root.removeNode()
        ghost_root = self.scene_root.attachNewNode("ghost-root")
        ghost_root.setTransparency(TransparencyAttrib.MAlpha)
        ghost_root.setColorScale(0.36, 0.82, 0.98, 0.60)
        ghost_root.setDepthWrite(False)
        ghost_root.setBin("transparent", 30)
        ghost_root.hide()
        self._load_styled_model(
            ghost_root,
            "sphere",
            scale=Vec3(self.sim_config.marble_radius * 0.96),
            color=Vec4(0.36, 0.82, 0.98, 0.62),
        )
        marker = ghost_root.attachNewNode("ghost-marker")
        self._load_styled_model(
            marker,
            "sphere",
            scale=Vec3(self.sim_config.marble_radius * 0.24),
            color=Vec4(0.98, 1.0, 1.0, 0.82),
            pos=Vec3(self.sim_config.marble_radius * 1.02, 0.0, self.sim_config.marble_radius * 0.12),
        )
        samples = self.ghost_playback.samples if self.ghost_playback is not None else ()
        self.ghost_playback = GhostPlayback(samples=samples, root=ghost_root, marker=marker)

    def _setup_camera(self) -> None:
        if self.camLens is None or self.camera is None:
            return
        self.camLens.setFov(48.0)
        self._update_camera(self.simulation.snapshot(), 1.0 / 60.0)

    def _build_ui(self) -> None:
        self.title = OnscreenText(
            text="MARBLE PHYSICS RUN",
            pos=(-1.30, 0.92),
            align=TextNode.ALeft,
            scale=0.052,
            fg=(0.92, 0.94, 0.98, 1.0),
            mayChange=True,
        )
        self.subtitle = OnscreenText(
            text="Long banked course. Sparse obstacles. Fast lines.",
            pos=(-1.30, 0.85),
            align=TextNode.ALeft,
            scale=0.030,
            fg=(0.66, 0.72, 0.80, 1.0),
            mayChange=True,
        )
        self.hud = OnscreenText(
            text="",
            pos=(-1.30, 0.76),
            align=TextNode.ALeft,
            scale=0.036,
            fg=(0.90, 0.93, 0.97, 1.0),
            mayChange=True,
        )
        self.status = OnscreenText(
            text="",
            pos=(1.30, 0.90),
            align=TextNode.ARight,
            scale=0.036,
            fg=(0.90, 0.94, 1.0, 1.0),
            mayChange=True,
        )
        self.center_message = OnscreenText(
            text="MARBLE PHYSICS RUN",
            pos=(0.0, 0.18),
            align=TextNode.ACenter,
            scale=0.16,
            fg=(0.92, 0.94, 0.98, 1.0),
            mayChange=True,
        )
        self.banner = OnscreenText(
            text="Press Space to start.",
            pos=(0.0, 0.07),
            align=TextNode.ACenter,
            scale=0.040,
            fg=(0.78, 0.84, 0.90, 1.0),
            mayChange=True,
        )
        self.result = OnscreenText(
            text="",
            pos=(1.30, 0.62),
            align=TextNode.ARight,
            scale=0.040,
            fg=(1.0, 0.82, 0.28, 1.0),
            mayChange=True,
        )
        self.progress_caption = OnscreenText(
            text="",
            pos=(0.0, -0.80),
            align=TextNode.ACenter,
            scale=0.028,
            fg=(0.68, 0.74, 0.82, 1.0),
            mayChange=True,
        )
        self.controls = OnscreenText(
            text="Up/Down mode   Left/Right track   Space start/pause   M menu   R restart   Esc quit",
            pos=(0.0, -0.94),
            scale=0.030,
            fg=(0.70, 0.74, 0.80, 1.0),
            mayChange=True,
        )
        self.debug_text = OnscreenText(
            text="",
            pos=(-1.30, -0.44),
            align=TextNode.ALeft,
            scale=0.030,
            fg=(0.74, 0.86, 0.98, 1.0),
            mayChange=True,
        )

        self.flash_card = self._create_ui_card("flash-overlay", -1.34, 1.34, -1.02, 1.02, Vec4(0.0, 0.0, 0.0, 0.0))
        self.flash_card.setBin("fixed", 0)

        self.progress_back = self._create_ui_card("progress-back", 0.0, 2.32, -0.016, 0.016, Vec4(0.06, 0.08, 0.12, 0.85))
        self.progress_back.reparentTo(self.aspect2d)
        self.progress_back.setPos(-1.16, 0.0, -0.87)
        self.progress_back.setBin("fixed", 20)

        self.progress_fill = self._create_ui_card("progress-fill", 0.0, 2.32, -0.011, 0.011, Vec4(0.18, 0.70, 0.96, 0.95))
        self.progress_fill.reparentTo(self.aspect2d)
        self.progress_fill.setPos(-1.16, 0.0, -0.87)
        self.progress_fill.setBin("fixed", 21)

    def _create_ui_card(
        self,
        name: str,
        left: float,
        right: float,
        bottom: float,
        top: float,
        color: Vec4,
    ) -> NodePath:
        cm = CardMaker(name)
        cm.setFrame(left, right, bottom, top)
        card = self.aspect2d.attachNewNode(cm.generate())
        card.setTransparency(TransparencyAttrib.MAlpha)
        card.setColor(color)
        return card

    def _setup_input(self) -> None:
        self.accept("arrow_left", self._set_steering, [1.0])
        self.accept("arrow_left-up", self._release_steering, [1.0])
        self.accept("arrow_right", self._set_steering, [-1.0])
        self.accept("arrow_right-up", self._release_steering, [-1.0])
        self.accept("arrow_up", self._on_up_press)
        self.accept("arrow_down", self._on_down_press)
        self.accept("arrow_down-up", self._set_brake, [0.0])
        self.accept("space", self._queue_primary_action)
        self.accept("raw-space", self._queue_primary_action)
        self.accept("r", self._handle_reset)
        self.accept("enter", self._queue_primary_action)
        self.accept("raw-enter", self._queue_primary_action)
        self.accept("m", self._enter_menu_state)
        self.accept("tab", self._toggle_debug_overlay)
        self.accept("escape", self.userExit)

    def _queue_primary_action(self) -> None:
        self.primary_action_pending = True

    def _handle_reset(self) -> None:
        if self.race_phase == "menu":
            self.progress = load_progress()
            self.selected_level_index = 0
            self._load_level_progress()
            self._show_event(
                "MODE SELECT",
                "Progress reset with --reset-progress only",
                Vec4(0.92, 0.94, 0.98, 1.0),
                hold=2.0,
            )
            return
        self._reset()

    def _set_steering(self, direction: float) -> None:
        if self.race_phase == "menu":
            self._menu_change_track(-1 if direction > 0.0 else 1)
            return
        if self.race_phase != "running":
            return
        if self.start_input_lock_timer > 0.0:
            return
        self.steering_input = direction
        self.simulation.set_steering_input(direction)

    def _release_steering(self, direction: float) -> None:
        if self.steering_input == direction:
            self.steering_input = 0.0
            self.simulation.set_steering_input(0.0)

    def _set_brake(self, amount: float) -> None:
        if self.race_phase != "running":
            return
        if self.start_input_lock_timer > 0.0:
            self.brake_input = 0.0
            self.simulation.set_brake_input(0.0)
            return
        self.brake_input = amount
        self.simulation.set_brake_input(amount)

    def _on_up_press(self) -> None:
        if self.race_phase == "menu":
            self._menu_change_mode(-1)

    def _on_down_press(self) -> None:
        if self.race_phase == "menu":
            self._menu_change_mode(1)
            return
        self._set_brake(1.0)

    def _menu_change_mode(self, delta: int) -> None:
        if self.race_phase != "menu":
            return
        self.selected_mode_index = (self.selected_mode_index + delta) % len(self.MODE_OPTIONS)

    def _menu_change_track(self, delta: int) -> None:
        if self.race_phase != "menu":
            return
        self.selected_level_index = (self.selected_level_index + delta) % len(self.levels)

    def _handle_space(self) -> None:
        if self.race_phase == "menu":
            if self._is_level_unlocked(self.selected_level_index):
                self._start_selected_level()
            return
        if self.race_phase in {"finished", "failed"}:
            self._reset()
            return
        if self.race_phase == "title":
            self._reset()
            return
        if self.race_phase == "running":
            self.paused = not self.paused
            return
        if self.paused:
            self.paused = False

    def _poll_primary_action_keys(self) -> None:
        watcher = getattr(self, "mouseWatcherNode", None)
        if watcher is None:
            return
        space_down = watcher.is_button_down(KeyboardButton.space())
        enter_down = watcher.is_button_down(KeyboardButton.enter())
        if (space_down and not self._space_key_down) or (enter_down and not self._enter_key_down):
            self.primary_action_pending = True
        self._space_key_down = space_down
        self._enter_key_down = enter_down

    def _toggle_debug_overlay(self) -> None:
        self.debug_overlay_visible = not self.debug_overlay_visible

    def _show_event(self, text: str, subtitle: str, color: Vec4, *, hold: float = 0.9) -> None:
        self.event_text = text
        self.event_subtitle = subtitle
        self.event_color = Vec4(color)
        self.event_timer = hold

    def _trigger_flash(self, color: Vec4, amount: float) -> None:
        self.flash_color = Vec4(color.x, color.y, color.z, max(self.flash_alpha, amount))
        self.flash_alpha = max(self.flash_alpha, amount)

    def _load_ghost(self, samples: tuple[GhostSample, ...]) -> None:
        if self.ghost_playback is None:
            self.ghost_playback = GhostPlayback(
                samples=samples,
                root=NodePath("ghost-placeholder"),
                marker=NodePath("ghost-placeholder-marker"),
            )
            return
        self.ghost_playback.samples = samples
        if samples and self.race_phase != "menu":
            self.ghost_playback.root.show()
        else:
            self.ghost_playback.root.hide()

    def _record_ghost_sample(self, time_s: float) -> None:
        if self.active_mode != self.MODE_TIME_TRIAL or self.race_phase != "running":
            return
        if self.pending_ghost_samples and time_s - self._last_ghost_sample_time < 0.05:
            return
        state = self.simulation.snapshot()
        self.pending_ghost_samples.append(
            GhostSample.from_vec3(time=time_s, path_distance=state.path_distance, position=state.position)
        )
        self._last_ghost_sample_time = time_s

    def _update_ghost_playback(self, time_s: float) -> None:
        if self.active_mode != self.MODE_TIME_TRIAL:
            if self.ghost_playback is not None:
                self.ghost_playback.root.hide()
            return
        if self.ghost_playback is None or not self.ghost_playback.samples:
            return
        if self.race_phase in {"menu", "failed"}:
            self.ghost_playback.root.hide()
            return
        self.ghost_playback.root.show()
        samples = self.ghost_playback.samples
        current = samples[-1]
        for sample in samples:
            current = sample
            if sample.time >= time_s:
                break
        ghost_position = Vec3(*current.position) + Vec3(0.0, 0.0, self.sim_config.marble_radius * 0.10)
        self.ghost_playback.root.setPos(ghost_position)

    def _enter_title_state(self) -> None:
        self.race_phase = "title"
        self.paused = False
        self.current_section_name = "Banked Course"
        self.current_section_focus = "Press Space to start"
        self._show_event(
            "MARBLE PHYSICS RUN",
            "Press Space to start",
            Vec4(0.92, 0.94, 0.98, 1.0),
            hold=60.0,
        )

    def _reset_run_state(self) -> None:
        self.race_phase = "countdown"
        self.countdown_timer = 3.1
        self.last_countdown_value = int(ceil(self.countdown_timer))
        self.finish_order_counter = 0
        self.last_finish_place = None
        self.finish_time = None
        self.finish_grade = None
        self.failure_reason = None
        self.major_impacts = 0
        self.section_major_impacts = 0
        self.run_stability_loss = 0.0
        self.section_stability_loss = 0.0
        self.clean_sections = 0
        self.section_boost_hits = 0
        self.boost_chain = 0
        self.best_boost_chain = 0
        self.off_track_timer = 0.0
        self.player_fall_timer = 0.0
        self.player_reset_warning_second = None
        self.last_supported_path_distance = 0.0
        self.impact_event_cooldown = 0.0
        self.next_section_index = 0
        self.section_results.clear()
        self.current_section_name = self.course_sections[0].name
        self.current_section_focus = self.course_sections[0].focus
        self.flash_alpha = 0.0
        self.boost_flash = 0.0
        self.finish_flash = 0.0
        self.start_burst = 0.0
        self.start_input_lock_timer = 0.0
        self.launch_assist_timer = 0.0
        self._clear_finish_confetti()
        self.pending_ghost_samples = []
        self._last_ghost_sample_time = 0.0
        self._stall_timer = 0.0
        self._show_event("3", "", Vec4(0.92, 0.94, 0.98, 1.0), hold=10.0)
        for pad in self.boost_pads:
            pad.pulse = 0.0

    def _reset(self) -> None:
        self.simulation.reset()
        self.simulation.settle_start_contact()
        self.paused = False
        self.steering_input = 0.0
        self.brake_input = 0.0
        self.start_input_lock_timer = 0.0
        self.launch_assist_timer = 0.0
        self.simulation.set_steering_input(0.0)
        self.simulation.set_brake_input(0.0)
        start_pos = self.simulation.snapshot().position
        self.camera_forward = ramp_tangent(self.sim_config, 0.0)
        self.camera_forward.z = 0.0
        self.camera_forward.normalize()
        self.camera_velocity_forward = Vec3(self.camera_forward)
        self.camera_track_distance = 0.0
        self.camera_collision_blend = 0.0
        self.camera_focus_lateral_offset = 0.0
        self.camera_focus_vertical_offset = 0.0
        self.camera_roll = 0.0
        self.camera_focus = Vec3(start_pos)
        self.camera_look_target = Vec3(start_pos) + self.camera_forward * self.camera_look_ahead
        self._reset_bot_marbles()
        self._settle_start_grid()
        self._reset_run_state()
        self._update_ghost_playback(0.0)
        self._apply_state(1.0 / 60.0)

    def _update_feedback_state(self, dt: float) -> None:
        self.flash_alpha = max(0.0, self.flash_alpha - dt * 1.7)
        self.boost_flash = max(0.0, self.boost_flash - dt * 1.8)
        self.finish_flash = max(0.0, self.finish_flash - dt * 0.75)
        self.start_burst = max(0.0, self.start_burst - dt * 1.35)
        self.impact_event_cooldown = max(0.0, self.impact_event_cooldown - dt)
        self.start_input_lock_timer = max(0.0, self.start_input_lock_timer - dt)
        self.launch_assist_timer = max(0.0, self.launch_assist_timer - dt)
        if self.event_timer > 0.0:
            self.event_timer = max(0.0, self.event_timer - dt)
        elif self.race_phase == "running":
            self.event_text = ""
            self.event_subtitle = ""
        for pad in self.boost_pads:
            pad.pulse = max(0.0, pad.pulse - dt * 1.4)

    def _handle_countdown(self, dt: float) -> None:
        self.countdown_timer = max(0.0, self.countdown_timer - dt)
        countdown_value = min(3, max(1, int(ceil(self.countdown_timer))))
        if self.countdown_timer > 0.0 and countdown_value != self.last_countdown_value:
            self.last_countdown_value = countdown_value
            if countdown_value == 2:
                self.start_burst = max(self.start_burst, 0.16)
                self._show_event("2", "HOLD", Vec4(0.98, 0.82, 0.34, 1.0), hold=10.0)
                self._trigger_flash(Vec4(0.98, 0.82, 0.34, 1.0), 0.12)
            elif countdown_value == 1:
                self.start_burst = max(self.start_burst, 0.24)
                self._show_event("1", "PUNCH IT", Vec4(0.96, 0.56, 0.22, 1.0), hold=10.0)
                self._trigger_flash(Vec4(0.96, 0.56, 0.22, 1.0), 0.16)
        if self.countdown_timer <= 0.0:
            self.race_phase = "running"
            self.simulation.marble_body.setLinearVelocity(Vec3(0.0, 0.0, 0.0))
            self.simulation.marble_body.setAngularVelocity(Vec3(0.0, 0.0, 0.0))
            self.simulation.marble_body.clearForces()
            self.simulation.add_forward_speed(3.20)
            self.start_input_lock_timer = 0.22
            self.launch_assist_timer = self.launch_assist_duration
            for index, bot in enumerate(self.bot_marbles):
                bot.body.setLinearVelocity(Vec3(0.0, 0.0, 0.0))
                bot.body.setAngularVelocity(Vec3(0.0, 0.0, 0.0))
                bot.body.clearForces()
                launch_speed = 1.10 + 0.08 * index + max(0.0, bot.speed_bias) * 0.14
                tangent = ramp_tangent(self.sim_config, max(0.0, bot.progress))
                if tangent.length_squared() > 1e-9:
                    tangent.normalize()
                    bot.body.setLinearVelocity(tangent * launch_speed)
            self.start_burst = 1.0
            self.boost_flash = max(self.boost_flash, 0.42)
            self._show_event("GO!", "SLINGSHOT", Vec4(0.28, 0.90, 0.98, 1.0), hold=0.7)
            self._trigger_flash(Vec4(0.28, 0.90, 0.98, 1.0), 0.24)

    def _update_launch_assist(self, dt: float) -> None:
        if self.race_phase != "running" or self.launch_assist_timer <= 0.0:
            return
        normalized = self.launch_assist_timer / max(self.launch_assist_duration, 1e-6)
        assist_strength = 5.2 * normalized * normalized + 0.80
        self.simulation.add_forward_speed(assist_strength * dt)
        if self.start_input_lock_timer > 0.0:
            self.steering_input = 0.0
            self.brake_input = 0.0
            self.simulation.set_steering_input(0.0)
            self.simulation.set_brake_input(0.0)

    def _finish_metrics(self, position: Vec3) -> tuple[float, float, float]:
        metrics = shared_finish_metrics(
            self.sim_config,
            position,
            target_distance=self.finish_line_distance,
        )
        return metrics.longitudinal_offset, metrics.lateral_offset, metrics.normal_offset

    def _has_crossed_finish(self, position: Vec3) -> bool:
        return shared_has_crossed_finish(
            self.sim_config,
            position,
            target_distance=self.finish_line_distance,
        )

    def _boost_pad_at_position(self, position: Vec3, *, airborne: bool = False) -> tuple[int, object] | None:
        activation_margin = self.sim_config.marble_radius
        for index, boost_pad in enumerate(self.sim_config.boost_pads):
            center = boost_pad_center_position(self.sim_config, boost_pad)
            tangent = ramp_tangent(self.sim_config, boost_pad.distance_along_ramp)
            if tangent.length_squared() <= 1e-9:
                continue
            tangent.normalize()
            side = ramp_side(self.sim_config, boost_pad.distance_along_ramp)
            normal = ramp_normal(self.sim_config, boost_pad.distance_along_ramp)
            relative = position - center
            longitudinal_offset = abs(relative.dot(tangent))
            lateral_offset = abs(relative.dot(side))
            normal_offset = relative.dot(normal)
            if longitudinal_offset > boost_pad.length * 0.5 + activation_margin:
                continue
            if lateral_offset > boost_pad.width * 0.5 + activation_margin:
                continue
            if normal_offset < -0.12 or normal_offset > activation_margin * 1.8:
                continue
            if airborne and normal_offset > activation_margin * 0.8:
                continue
            return index, boost_pad
        return None

    def _track_relative_offsets(self, position: Vec3, reference_distance: float) -> tuple[float, float, float]:
        return shared_track_relative_offsets(self.sim_config, position, reference_distance)

    def _is_definitely_off_track(self, position: Vec3, reference_distance: float) -> bool:
        return shared_is_definitely_off_track(self.sim_config, position, reference_distance)

    def _effective_track_distance(self, position: Vec3, reference_distance: float) -> float:
        return shared_effective_track_distance(self.sim_config, position, reference_distance)

    def _is_below_track_freefall(
        self,
        position: Vec3,
        *,
        reference_distance: float,
        contacts: int,
        velocity: Vec3,
    ) -> bool:
        nearest_distance = self._nearest_track_distance(position, reference_distance)
        surface = ramp_surface_point(self.sim_config, nearest_distance)
        lateral_offset, normal_offset, _ = self._track_relative_offsets(position, nearest_distance)
        below_surface = position.z < surface.z - 0.85 or normal_offset < -0.70
        far_from_lane = lateral_offset > self.sim_config.width * 0.5 + 0.25
        falling_fast = velocity.z < -7.0
        return contacts == 0 and below_surface and falling_fast and (far_from_lane or position.z < surface.z - 1.05)

    def _respawn_player_to_start(self) -> None:
        self.simulation.respawn_marble(distance=0.0)
        self.simulation.settle_start_contact(settle_steps=16)
        self.simulation.set_steering_input(0.0)
        self.simulation.set_brake_input(0.0)
        self.steering_input = 0.0
        self.brake_input = 0.0
        self.player_fall_timer = 0.0
        self.player_reset_warning_second = None
        self.off_track_timer = 0.0
        self._stall_timer = 0.0
        self.launch_assist_timer = 0.0
        self.start_input_lock_timer = 0.0
        start_pos = self.simulation.snapshot().position
        self.camera_track_distance = 0.0
        self.camera_focus = Vec3(start_pos)
        self.camera_look_target = Vec3(start_pos) + self.camera_forward * self.camera_look_ahead
        self._show_event("RESET", "BACK TO START", Vec4(0.98, 0.82, 0.34, 1.0), hold=0.9)

    def _update_player_fall_reset(self, previous_state, state, dt: float) -> bool:
        if self.race_phase != "running":
            self.player_fall_timer = 0.0
            self.player_reset_warning_second = None
            return False
        reference_distance = max(previous_state.path_distance, state.path_distance)
        if self._is_below_track_freefall(
            state.position,
            reference_distance=reference_distance,
            contacts=state.total_contacts,
            velocity=state.linear_velocity,
        ):
            self.player_fall_timer += dt
            remaining = max(0.0, self.fall_reset_delay - self.player_fall_timer)
            warning_second = max(1, int(ceil(remaining))) if remaining > 0.0 else 0
            if warning_second > 0 and warning_second != self.player_reset_warning_second:
                self.player_reset_warning_second = warning_second
                self._show_event(
                    f"RESET {warning_second}",
                    "FREEFALL DETECTED",
                    Vec4(0.96, 0.56, 0.22, 1.0),
                    hold=0.45,
                )
            if self.player_fall_timer >= self.fall_reset_delay:
                self._respawn_player_to_start()
                return True
        else:
            self.player_fall_timer = 0.0
            self.player_reset_warning_second = None
        return False

    def _update_bot_fall_resets(self, dt: float) -> None:
        if self.active_mode != self.MODE_BOT_RACE:
            return
        for bot in self.bot_marbles:
            if bot.finish_time is not None:
                continue
            position = bot.body_np.getPos()
            contacts = self.simulation.world.contactTest(bot.body).getNumContacts()
            if self._is_below_track_freefall(
                position,
                reference_distance=bot.progress,
                contacts=contacts,
                velocity=bot.body.getLinearVelocity(),
            ):
                bot.fall_timer += dt
                remaining = max(0.0, self.fall_reset_delay - bot.fall_timer)
                warning_second = max(1, int(ceil(remaining))) if remaining > 0.0 else 0
                if warning_second > 0 and warning_second != bot.reset_warning_second:
                    bot.reset_warning_second = warning_second
                    self._show_event(
                        f"{bot.name} {warning_second}",
                        "RESETTING",
                        Vec4(0.94, 0.68, 0.28, 1.0),
                        hold=0.35,
                    )
                if bot.fall_timer >= self.fall_reset_delay:
                    self._reset_bot_body(bot, distance=0.0)
                    bot.previous_position = Vec3(bot.body_np.getPos())
                    bot.previous_progress = bot.progress
            else:
                bot.fall_timer = 0.0
                bot.reset_warning_second = None

    def _nearest_track_distance(self, position: Vec3, reference_distance: float) -> float:
        clamped_reference = max(0.0, min(self.sim_config.length, reference_distance))
        return path_distance_for_position_near(
            self.sim_config,
            position,
            clamped_reference,
            search_radius_segments=32,
        )

    def _off_track_status(
        self,
        position: Vec3,
        *,
        reference_distance: float,
        contacts: int,
        airborne: bool,
        impact_severity: float,
        previous_contacts: int = 0,
        previous_impact_severity: float = 0.0,
    ) -> tuple[float, bool, bool]:
        nearest_distance = self._nearest_track_distance(position, reference_distance)
        lateral_offset, normal_offset, z_drop = self._track_relative_offsets(position, nearest_distance)
        unsupported = contacts == 0 and airborne
        recovering = (
            contacts > 0
            or previous_contacts > 0
            or impact_severity >= 0.34
            or previous_impact_severity >= 0.34
        )
        lane_half_width = self.sim_config.width * 0.5
        far_outside = lateral_offset > lane_half_width + max(1.15, self.sim_config.marble_radius * 5.0)
        outside_lane = lateral_offset > lane_half_width + max(0.48, self.sim_config.marble_radius * 2.2)
        deep_below_track = normal_offset < -1.30 or z_drop < -1.70 or position.z < -3.0
        below_track = normal_offset < -0.42 or z_drop < -0.58
        definitely_lost = unsupported and (deep_below_track or far_outside or (outside_lane and below_track))
        maybe_lost = unsupported and outside_lane and below_track and not recovering
        if recovering:
            severity = 0.0
        elif definitely_lost:
            severity = 2.4
        elif maybe_lost:
            severity = 1.0
        else:
            severity = 0.0
        return severity, recovering, definitely_lost

    def _should_finish_run(self, previous_state, state) -> bool:
        return shared_should_finish_run(
            self.sim_config,
            previous_state.position,
            previous_state.path_distance,
            state.position,
            state.path_distance,
            target_distance=self.finish_line_distance,
        )

    def _claim_finish_order(self) -> int:
        self.finish_order_counter += 1
        return self.finish_order_counter

    def _bot_race_placement(self) -> int:
        return self._claim_finish_order()

    def _race_standings(self, player_progress: float) -> list[tuple[str, tuple[float, float]]]:
        racers: list[tuple[str, tuple[float, float]]] = [("YOU", (0.0, player_progress))]
        for bot in self.bot_marbles:
            if bot.finish_order is not None:
                racers.append((bot.name, (1.0, -float(bot.finish_order))))
            else:
                racers.append((bot.name, (0.0, bot.progress)))
        racers.sort(key=lambda item: item[1], reverse=True)
        return racers

    def _format_bot_leaderboard(self, player_progress: float) -> str:
        standings = self._race_standings(player_progress)
        lines: list[str] = []
        for place, (name, ranking) in enumerate(standings, start=1):
            finished_flag, metric = ranking
            if finished_flag >= 1.0:
                gap_text = "FIN"
            else:
                leader_metric = standings[0][1][1]
                gap = leader_metric - metric
                gap_text = "LEAD" if place == 1 else f"-{gap:0.1f}m"
            lines.append(f"P{place} {name:<5} {gap_text:>5}")
        return "\n".join(lines)

    def _clear_finish_confetti(self) -> None:
        for piece in self.finish_confetti:
            if not piece.node.isEmpty():
                piece.node.removeNode()
        self.finish_confetti = []

    def _spawn_finish_confetti(self, anchor: Vec3) -> None:
        self._clear_finish_confetti()
        palette = (
            Vec4(0.98, 0.28, 0.34, 1.0),
            Vec4(0.98, 0.86, 0.22, 1.0),
            Vec4(0.22, 0.88, 0.98, 1.0),
            Vec4(0.96, 0.96, 0.98, 1.0),
        )
        burst_origin = anchor + Vec3(0.0, 0.0, 0.8)
        piece_count = 28
        for index in range(piece_count):
            piece_root = self.scene_root.attachNewNode(f"finish-confetti-{index}")
            piece_root.setPos(burst_origin)
            self._attach_centered_box(
                piece_root,
                size=Vec3(0.05, 0.012, 0.03),
                color=palette[index % len(palette)],
                style="screen",
            )
            angle = tau * (index / piece_count)
            radial_speed = 2.2 + 0.7 * ((index % 5) / 4.0)
            upward_speed = 2.8 + 1.4 * ((index % 7) / 6.0)
            velocity = Vec3(cos(angle) * radial_speed, sin(angle) * radial_speed, upward_speed)
            spin_hpr = Vec3(180.0 + index * 9.0, 220.0 + index * 11.0, 260.0 + index * 7.0)
            self.finish_confetti.append(
                ConfettiPiece(
                    node=piece_root,
                    velocity=velocity,
                    spin_hpr=spin_hpr,
                    lifetime=1.7 + 0.25 * (index % 3),
                )
            )

    def _pulse_boost_pad(self, pad: BoostPadVisual) -> None:
        pad.pulse = 1.0

    def _activate_boost_pad(self, pad: BoostPadVisual, index: int) -> None:
        self._pulse_boost_pad(pad)
        self.section_boost_hits += 1
        self.boost_chain += 1
        self.best_boost_chain = max(self.best_boost_chain, self.boost_chain)
        self.boost_flash = max(self.boost_flash, 0.60)
        self._trigger_flash(Vec4(0.26, 0.84, 0.98, 1.0), 0.10)

    def _complete_section(self, section_index: int) -> None:
        section = self.course_sections[section_index]
        section_result = rate_section(
            major_impacts=self.section_major_impacts,
            stability_loss=self.section_stability_loss,
            boost_hits=self.section_boost_hits,
        )
        if section_result.rating == "CLEAN":
            self.clean_sections += 1
        self.section_results.append(f"{section.name}: {section_result.rating}")
        self.section_major_impacts = 0
        self.section_stability_loss = 0.0
        self.section_boost_hits = 0
        self.next_section_index += 1
        if self.next_section_index < len(self.course_sections):
            self.current_section_name = self.course_sections[self.next_section_index].name
            self.current_section_focus = self.course_sections[self.next_section_index].focus
        else:
            self.current_section_name = "Finish Run"
            self.current_section_focus = "Carry the exit speed"

    def _finish_run(self, state) -> None:
        if self.race_phase == "finished":
            return
        if self.active_mode == self.MODE_TIME_TRIAL and (
            not self.pending_ghost_samples or self.pending_ghost_samples[-1].time < state.time
        ):
            self.pending_ghost_samples.append(
                GhostSample.from_vec3(time=state.time, path_distance=state.path_distance, position=state.position)
            )
        self.race_phase = "finished"
        self.current_section_name = "Finished"
        self.current_section_focus = ""
        self.finish_time = state.time
        self.last_finish_place = None
        previous_best = self.session_best_time
        if self.current_level.target_times is not None:
            self.finish_grade = grade_time_trial(state.time, self.current_level.target_times)
        else:
            self.finish_grade = grade_run(
                length=self.sim_config.length,
                obstacle_count=len(self.sim_config.obstacles),
                finish_time=state.time,
                major_impacts=self.major_impacts,
                clean_sections=self.clean_sections,
                stability_loss=self.run_stability_loss,
                boost_chain=self.best_boost_chain,
            )
        is_best = previous_best is None or state.time < previous_best
        if self.active_mode == self.MODE_TIME_TRIAL:
            if is_best:
                self.session_best_time = state.time
                self.progress = update_level_progress(
                    self.progress,
                    self.current_level_key,
                    unlocked=True,
                    best_time=state.time,
                    best_medal=self.finish_grade.medal,
                    ghost=tuple(self.pending_ghost_samples),
                )
                save_progress(self.progress)
            else:
                self.progress = update_level_progress(
                    self.progress,
                    self.current_level_key,
                    unlocked=True,
                    best_medal=self.finish_grade.medal,
                )
                save_progress(self.progress)
        else:
            self.progress = update_level_progress(
                self.progress,
                self.current_level_key,
                unlocked=True,
                best_medal=self.finish_grade.medal,
            )
            save_progress(self.progress)
        self._unlock_next_level()
        subtitle = format_seconds(state.time)
        if self.active_mode == self.MODE_BOT_RACE:
            placement = self._bot_race_placement()
            self.last_finish_place = placement
            subtitle += f"   P{placement}/{len(self.bot_marbles) + 1}"
        if self.active_mode == self.MODE_TIME_TRIAL and is_best:
            subtitle += "   NEW BEST"
        elif self.active_mode == self.MODE_TIME_TRIAL and previous_best is not None:
            subtitle += f"   {format_delta(state.time - previous_best)}"
        finish_label = "FINISH"
        if self.active_mode == self.MODE_BOT_RACE and self.last_finish_place is not None:
            suffix = "TH"
            if self.last_finish_place % 10 == 1 and self.last_finish_place % 100 != 11:
                suffix = "ST"
            elif self.last_finish_place % 10 == 2 and self.last_finish_place % 100 != 12:
                suffix = "ND"
            elif self.last_finish_place % 10 == 3 and self.last_finish_place % 100 != 13:
                suffix = "RD"
            finish_label = f"{self.last_finish_place}{suffix} PLACE"
        self._show_event(finish_label, subtitle, Vec4(0.92, 0.94, 0.98, 1.0), hold=60.0)
        self.finish_flash = 0.45
        self._spawn_finish_confetti(state.position)

    def _should_reset_for_off_track(self, previous_state, state, dt: float) -> bool:
        return False

    def _handle_running_state(self, previous_state, state, dt: float) -> None:
        if state.boost_active and not previous_state.boost_active:
            active_index = self.simulation.active_boost_pad_index
            if active_index is not None and active_index < len(self.boost_pads):
                self._activate_boost_pad(self.boost_pads[active_index], active_index)

        if self._should_finish_run(previous_state, state):
            self._finish_run(state)
            return

        if self._update_player_fall_reset(previous_state, state, dt):
            return

        progress_delta = max(0.0, state.path_distance - previous_state.path_distance)
        if progress_delta < 0.01 and state.speed < 0.35:
            self._stall_timer += dt
        else:
            self._stall_timer = max(0.0, self._stall_timer - dt * 2.0)
        if self._stall_timer >= 1.25:
            self._enter_failed_state("STALLED", "Press Space to retry   M for menu")
            return

        speed_drop = max(0.0, previous_state.speed - state.speed - self.brake_input * 0.45)
        stability_step = state.impact_severity * dt * 0.95 + speed_drop * 0.14 + (0.04 if state.airborne else 0.0)
        self.section_stability_loss += stability_step
        self.run_stability_loss += stability_step

        while self.next_section_index < len(self.course_sections):
            section = self.course_sections[self.next_section_index]
            if previous_state.path_distance < section.end_distance <= state.path_distance:
                self._complete_section(self.next_section_index)
                continue
            break

        if state.total_contacts > 0 and state.impact_severity >= 0.70 and self.impact_event_cooldown <= 0.0:
            self.major_impacts += 1
            self.section_major_impacts += 1
            self.impact_event_cooldown = 0.72
            self.boost_chain = 0
        self._record_ghost_sample(state.time)

    def _tick(self, task: Task) -> int:
        dt = clamp_frame_dt(globalClock.getDt())
        self._poll_primary_action_keys()
        if self.primary_action_pending:
            self.primary_action_pending = False
            self._handle_space()
        self._update_feedback_state(dt)
        if not self.paused:
            if self.race_phase == "countdown":
                self._handle_countdown(dt)
            elif self.race_phase == "running":
                self._update_launch_assist(dt)
                self._update_bot_controllers(dt)
                previous_state = self.simulation.snapshot()
                self.simulation.step(dt)
                self._update_bot_fall_resets(dt)
                self._update_bot_race_state()
                self._handle_running_state(previous_state, self.simulation.snapshot(), dt)
        self._animate_props(task.time, dt)
        self._update_ghost_playback(self.simulation.snapshot().time)
        self._apply_state(dt)
        return Task.cont

    def _animate_props(self, time_s: float, dt: float) -> None:
        state = self.simulation.snapshot()
        speed = state.speed
        for pulse in self.pulse_nodes:
            lift = 1.0 + pulse.amplitude * (0.5 + 0.5 * sin(time_s * pulse.speed + pulse.phase))
            lift += self.boost_flash * 0.04 + self.finish_flash * 0.06 + self.start_burst * 0.10
            pulse.node.setColorScale(
                min(1.08, pulse.base_color.x * lift),
                min(1.08, pulse.base_color.y * lift),
                min(1.08, pulse.base_color.z * lift),
                pulse.base_color.w,
            )

        for index, line in enumerate(self.speed_lines):
            speed_scale = 0.62 + min(0.52, speed * 0.040)
            speed_scale += self.boost_flash * 0.14 + self.start_burst * 0.36
            alpha = min(0.56, 0.08 + speed * 0.020 + self.boost_flash * 0.08 + self.start_burst * 0.14)
            line.setScale(speed_scale * (1.0 + index * 0.12), 1.0, 1.0)
            line.setColor(0.78, 0.82 + 0.04 * index, 0.88 + 0.05 * index, alpha * (1.0 - index * 0.18))

        for pad in self.boost_pads:
            pulse_scale = 1.0 + pad.pulse * 0.10
            ambient_scale = 0.92
            pad.root.setScale(pulse_scale)
            pad.root.setColorScale(
                ambient_scale + pad.pulse * 0.16,
                ambient_scale + pad.pulse * 0.24,
                ambient_scale + pad.pulse * 0.30,
                1.0,
            )
            pad.glow.setScale(1.0 + pad.pulse * 0.18)
            pad.glow.setColorScale(
                0.78 + pad.pulse * 0.28,
                0.88 + pad.pulse * 0.36,
                0.96 + pad.pulse * 0.42,
                1.0,
            )

        remaining_confetti: list[ConfettiPiece] = []
        for piece in self.finish_confetti:
            piece.age += dt
            if piece.age >= piece.lifetime or piece.node.isEmpty():
                if not piece.node.isEmpty():
                    piece.node.removeNode()
                continue
            piece.velocity.z -= 8.8 * dt
            piece.node.setPos(piece.node.getPos() + piece.velocity * dt)
            piece.node.setHpr(piece.node.getHpr() + piece.spin_hpr * dt)
            alpha = max(0.0, 1.0 - piece.age / piece.lifetime)
            piece.node.setColorScale(1.0, 1.0, 1.0, alpha)
            remaining_confetti.append(piece)
        self.finish_confetti = remaining_confetti
        self._animate_moving_obstacles(state.time)

    def _animate_moving_obstacles(self, time_s: float) -> None:
        for visual in self.moving_obstacle_visuals:
            if visual.root.isEmpty():
                continue
            segment = ramp_segment_at_distance(self.sim_config, visual.obstacle.distance_along_ramp)
            if visual.obstacle.motion_kind == "pendulum":
                surface = ramp_surface_point(self.sim_config, visual.obstacle.distance_along_ramp)
                side = ramp_side(self.sim_config, visual.obstacle.distance_along_ramp)
                normal = ramp_normal(self.sim_config, visual.obstacle.distance_along_ramp)
                pivot_height = visual.obstacle.pivot_height if visual.obstacle.pivot_height > 0.0 else 1.05
                phase = visual.obstacle.motion_phase + time_s * visual.obstacle.motion_speed
                swing_angle = sin(phase) * visual.obstacle.motion_amplitude
                visual.root.setPos(surface + side * visual.obstacle.lateral_offset + normal * pivot_height)
                visual.root.setHpr(segment.heading_deg, segment.bank_deg, self.sim_config.angle_deg)
                if visual.arm is not None and not visual.arm.isEmpty():
                    visual.arm.setHpr(0.0, degrees(swing_angle), 0.0)
            else:
                center, heading_deg = obstacle_pose(self.sim_config, visual.obstacle, time_s)
                visual.root.setPos(center)
                visual.root.setHpr(segment.heading_deg + heading_deg, segment.bank_deg, self.sim_config.angle_deg)
            if visual.marker is not None and visual.obstacle.motion_kind == "sweeper":
                phase = visual.obstacle.motion_phase + time_s * visual.obstacle.motion_speed
                visual.marker.setColorScale(1.0, 1.0, 1.0, 0.84 + 0.16 * sin(phase))

    def _apply_state(self, dt: float | None = None) -> None:
        state = self.simulation.snapshot()
        self.marble_root.setPos(state.position)
        self.marble_root.setQuat(self.simulation.marble_np.getQuat())
        self._update_bot_visuals()
        camera_dt = 1.0 / 60.0 if dt is None else min(max(dt, 0.0), 1.0 / 30.0)
        self._update_camera(state, camera_dt)
        progress = min(100.0, (state.path_distance / self.sim_config.length) * 100.0)
        best_text = "--" if self.session_best_time is None else format_seconds(self.session_best_time)
        medal_targets = self.current_level.target_times
        self.title.setText(self.current_level.name)
        subtitle = self.current_level.description or f"{self.current_level.difficulty.title()} {self.current_level.theme.title()} course."
        if self.race_phase == "menu":
            subtitle = self._mode_description(self.selected_mode)
        self.subtitle.setText(subtitle)
        self.hud.setText(
            (
                f"time   {format_seconds(state.time):>7}\n"
                f"speed  {state.speed:5.2f} m/s\n"
                f"track  {progress:5.1f}%"
            )
        )
        if medal_targets is not None:
            medal_text = (
                f"gold   {format_seconds(medal_targets.gold)}\n"
                f"silver {format_seconds(medal_targets.silver)}\n"
                f"bronze {format_seconds(medal_targets.bronze)}"
            )
        else:
            medal_text = ""
        if self.active_mode == self.MODE_BOT_RACE and self.race_phase not in {"menu"}:
            medal_text = self._format_bot_leaderboard(state.path_distance)
        self.status.setText(medal_text)
        if self.finish_grade is not None and self.finish_time is not None:
            result_text = (
                f"run    {format_seconds(self.finish_time)}\n"
                f"best   {best_text}\n"
                f"medal  {self.finish_grade.medal}"
            )
            if self.active_mode == self.MODE_BOT_RACE and self.last_finish_place is not None:
                result_text = (
                    f"place  P{self.last_finish_place}/{len(self.bot_marbles) + 1}\n"
                    f"time   {format_seconds(self.finish_time)}\n"
                    f"medal  {self.finish_grade.medal}"
                )
            self.result.setText(result_text)
            self.result.setFg((self.finish_grade.color[0], self.finish_grade.color[1], self.finish_grade.color[2], 1.0))
        else:
            self.result.setText("")

        center_text = self.event_text
        center_subtitle = self.event_subtitle
        if self.race_phase == "menu":
            selected = self.levels[self.selected_level_index]
            selected_unlocked = self._is_level_unlocked(self.selected_level_index)
            center_text = self._mode_label(self.selected_mode)
            lock_text = "" if selected_unlocked else "LOCKED   "
            center_subtitle = (
                f"{lock_text}{selected.level.name}   "
                f"{selected.level.difficulty.title()} / {selected.level.theme.title()}"
            )
        if self.paused and self.race_phase == "running":
            center_text = "PAUSED"
            center_subtitle = "Space to resume   R to restart"
        if self.race_phase == "running" and self.event_timer <= 0.0 and not self.paused:
            center_text = ""
            center_subtitle = ""
        self.center_message.setText(center_text)
        self.banner.setText(center_subtitle)
        self.center_message.setFg((self.event_color.x, self.event_color.y, self.event_color.z, 1.0))
        self.banner.setFg((self.event_color.x * 0.88, self.event_color.y * 0.92, self.event_color.z, 1.0))
        if self.race_phase == "menu":
            selected = self.levels[self.selected_level_index]
            progress_entry = self.progress.levels.get(selected.key)
            best_time = format_seconds(progress_entry.best_time) if progress_entry is not None and progress_entry.best_time is not None else "--"
            self.progress_caption.setText(
                f"Mode: {self._mode_label(self.selected_mode)}   Track: {selected.level.name}   Best: {best_time}"
            )
        elif self.race_phase in {"finished", "failed"}:
            action_text = "Space retry   M menu"
            if self.level_index + 1 < len(self.levels) and self._is_level_unlocked(min(self.level_index + 1, len(self.levels) - 1)):
                action_text += "   Enter retry"
            self.progress_caption.setText(action_text)
        else:
            section_text = f"{self.current_section_name}   {self.current_section_focus}"
            if self.active_mode == self.MODE_BOT_RACE and self.bot_marbles:
                racers = self._race_standings(state.path_distance)
                place = next(index for index, racer in enumerate(racers, start=1) if racer[0] == "YOU")
                leader_name = racers[0][0]
                section_text = f"P{place}/{len(racers)}   Leader {leader_name}   {self.current_section_name}"
            self.progress_caption.setText(section_text)
        self.progress_fill.setScale(max(0.001, progress / 100.0), 1.0, 1.0)
        self.progress_fill.setColor(0.82, 0.84, 0.88, 0.95)
        if self.race_phase == "menu":
            self.controls.setText("Up/Down mode   Left/Right track   Enter start   R reset   Esc quit")
        else:
            self.controls.setText("Left/Right steer   Down brake   Space pause   M menu   R restart   Esc quit")
        if self.debug_overlay_visible:
            self.debug_text.setText(
                (
                    f"dbg  speed {state.speed:5.2f} m/s\n"
                    f"     lateral {state.lateral_speed:5.2f} m/s\n"
                    f"     impact {state.impact_severity:4.2f}\n"
                    f"     rail contacts {state.rail_contacts:2d}\n"
                    f"     airborne {'yes' if state.airborne else 'no'}\n"
                    f"     boost idx {state.boost_pad_index if state.boost_pad_index is not None else '--'}"
                )
            )
        else:
            self.debug_text.setText("")

        flash_alpha = max(self.flash_alpha, self.boost_flash * 0.10, self.finish_flash * 0.16, self.start_burst * 0.18)
        flash_color = Vec4(self.flash_color)
        if self.start_burst * 0.18 >= flash_alpha - 1e-6:
            flash_color = Vec4(0.24, 0.86, 0.98, flash_alpha)
        if self.boost_flash * 0.10 >= flash_alpha - 1e-6:
            flash_color = Vec4(0.18, 0.82, 0.96, flash_alpha)
        if self.finish_flash * 0.16 >= flash_alpha - 1e-6 and self.finish_grade is not None:
            flash_color = Vec4(*self.finish_grade.color[:3], flash_alpha)
        self.flash_card.setColor(flash_color.x, flash_color.y, flash_color.z, flash_alpha)

    def _attach_centered_box(
        self,
        parent: NodePath,
        size: Vec3,
        color: Vec4,
        style: str | None = None,
        depth_offset: int = 0,
    ) -> None:
        box = self.loader.loadModel("models/box")
        box.reparentTo(parent)
        box.setPos(-size.x * 0.5, -size.y * 0.5, -size.z * 0.5)
        box.setScale(size)
        if style is None:
            box.setTextureOff(1)
        else:
            box.setTexture(self._get_texture(style), 1)
        box.setColor(color)
        self._apply_render_finish(box, style, depth_offset=depth_offset)

    def _load_styled_model(
        self,
        parent: NodePath,
        model_name: str,
        *,
        scale: Vec3,
        color: Vec4,
        pos: Vec3 | None = None,
        style: str | None = None,
        depth_offset: int = 0,
    ) -> NodePath:
        model_path = resolve_model_path(model_name)
        if model_path is None:
            model = self.loader.loadModel("models/box")
            model.reparentTo(parent)
            model.setPos(-scale.x * 0.5, -scale.y * 0.5, -scale.z * 0.5)
            model.setScale(scale)
        else:
            model = self.loader.loadModel(model_path)
            model.reparentTo(parent)
            model.setScale(scale)
        if pos is not None:
            model.setPos(pos)
        if style is None:
            model.setTextureOff(1)
        else:
            model.setTexture(self._get_texture(style), 1)
        model.setColor(color)
        self._apply_render_finish(model, style, depth_offset=depth_offset)
        return model

    def _apply_render_finish(self, node: NodePath, style: str | None, *, depth_offset: int = 0) -> None:
        node.clearColorScale()
        node.setShaderOff(1)
        node.setAntialias(AntialiasAttrib.MAuto)
        if style == "screen":
            node.setDepthOffset(max(1, depth_offset))
        elif depth_offset != 0:
            node.setDepthOffset(depth_offset)

    def _get_texture(self, style: str) -> Texture:
        cached = self._texture_cache.get(style)
        if cached is not None:
            return cached

        image = PNMImage(256, 256, 4)
        image.fill(0.0, 0.0, 0.0)
        image.alpha_fill(1.0)
        if style == "backdrop":
            for y in range(256):
                t = y / 255.0
                base_r = 0.04 + 0.05 * (1.0 - t)
                base_g = 0.05 + 0.03 * (1.0 - t)
                base_b = 0.09 + 0.10 * (1.0 - t)
                for x in range(256):
                    scan = 0.01 if y % 18 == 0 else 0.0
                    city = 0.05 if y > 160 and x % 29 == 0 and (x + y) % 47 < 3 else 0.0
                    image.set_xel_a(
                        x,
                        y,
                        min(1.0, base_r + scan + city),
                        min(1.0, base_g + scan * 0.8 + city),
                        min(1.0, base_b + scan * 0.4 + city * 0.6),
                        1.0,
                    )
        elif style == "runway":
            for y in range(256):
                for x in range(256):
                    center_falloff = abs((x / 255.0) - 0.5)
                    v = 0.10 + 0.05 * (1.0 - center_falloff * 2.0)
                    if x in range(18, 28) or x in range(228, 238):
                        v += 0.22
                    if y % 40 < 2:
                        v += 0.10
                    image.set_xel_a(x, y, v * 0.74, v * 0.78, v * 0.82, 1.0)
        elif style == "track":
            for y in range(256):
                for x in range(256):
                    pattern = 0.010 if (x + y) % 36 < 2 or (x - y) % 44 < 2 else 0.0
                    seam = 0.035 if y % 96 < 2 else 0.0
                    grain = (((x * 11) + (y * 7)) % 23) / 900.0
                    v = 0.12 + pattern + seam + grain
                    image.set_xel_a(x, y, v * 0.94, v * 0.96, v * 0.98, 1.0)
        elif style == "screen":
            for y in range(256):
                t = y / 255.0
                for x in range(256):
                    stripe = 0.035 if x % 56 < 3 else 0.0
                    grid = 0.018 if y % 44 < 2 else 0.0
                    image.set_xel_a(
                        x,
                        y,
                        0.20 + 0.10 * (1.0 - t) + stripe,
                        0.34 + 0.14 * (1.0 - t) + stripe * 0.55 + grid,
                        0.40 + 0.16 * (1.0 - t) + stripe * 0.40 + grid * 0.8,
                        1.0,
                    )
        else:
            for y in range(256):
                for x in range(256):
                    seam = 0.04 if x % 88 < 2 or y % 88 < 2 else 0.0
                    grain = (((x * 13) + (y * 7)) % 19) / 420.0
                    v = 0.10 + seam + grain
                    image.set_xel_a(x, y, v * 0.92, v * 0.95, v * 0.98, 1.0)

        texture = Texture(style)
        texture.load(image)
        texture.setWrapU(SamplerState.WM_repeat)
        texture.setWrapV(SamplerState.WM_repeat)
        texture.setMinfilter(SamplerState.FT_linear_mipmap_linear)
        texture.setMagfilter(SamplerState.FT_linear)
        self._texture_cache[style] = texture
        return texture

    def _sample_track_forward(self, distance: float) -> Vec3:
        sample_offsets = (-1.2, -0.6, 0.0, 0.6, 1.2)
        blended_forward = Vec3(0.0, 0.0, 0.0)
        for offset in sample_offsets:
            sample_distance = max(0.0, min(self.sim_config.length, distance + offset))
            tangent = ramp_tangent(self.sim_config, sample_distance)
            tangent.z = 0.0
            if tangent.length_squared() <= 1e-9:
                continue
            tangent.normalize()
            blended_forward += tangent
        if blended_forward.length_squared() <= 1e-9:
            return Vec3(self.camera_forward)
        blended_forward.normalize()
        return blended_forward

    def _update_camera(self, state, dt: float) -> None:
        if self.camera is None or self.camLens is None:
            return
        self.camera_collision_blend = smooth_binary_state(
            self.camera_collision_blend,
            state.total_contacts > 0,
            dt,
            attack_response=self.camera_collision_attack,
            release_response=self.camera_collision_release,
        )

        target_track_distance = max(0.0, state.path_distance - self.camera_track_follow_offset)
        self.camera_track_distance = max(
            target_track_distance - self.camera_track_max_lag,
            min(state.path_distance, target_track_distance),
        )

        track_forward = self._sample_track_forward(state.path_distance + 0.55)
        velocity_forward = Vec3(state.linear_velocity.x, state.linear_velocity.y, 0.0)
        target_forward = Vec3(track_forward)
        if velocity_forward.length_squared() > 0.64:
            velocity_forward.normalize()
            alignment = velocity_forward.dot(track_forward)
            if alignment > 0.6:
                velocity_mix = min(0.12, (alignment - 0.6) * 0.30)
                target_forward = track_forward * (1.0 - velocity_mix) + velocity_forward * velocity_mix
                if target_forward.length_squared() > 1e-9:
                    target_forward.normalize()

        heading_alpha = smoothing_alpha(self.camera_heading_response, dt)
        self.camera_forward = self.camera_forward * (1.0 - heading_alpha) + target_forward * heading_alpha
        self.camera_forward.normalize()

        track_point = ramp_surface_point(self.sim_config, state.path_distance)
        track_side = ramp_side(self.sim_config, state.path_distance)
        track_normal = ramp_normal(self.sim_config, state.path_distance)
        base_focus = state.position + track_normal * 0.20
        lateral_error = max(
            -self.sim_config.width * 0.10,
            min(
                self.sim_config.width * 0.10,
                (state.position - track_point).dot(track_side),
            ),
        )
        vertical_error = max(-0.02, min(0.12, state.position.z - track_point.z))
        self.camera_focus_lateral_offset = smooth_value(
            self.camera_focus_lateral_offset,
            lateral_error * (1.0 - 0.45 * self.camera_collision_blend),
            self.camera_focus_lateral_lag,
            dt,
        )
        self.camera_focus_vertical_offset = smooth_value(
            self.camera_focus_vertical_offset,
            vertical_error * (1.0 - 0.35 * self.camera_collision_blend),
            self.camera_focus_vertical_lag,
            dt,
        )
        desired_focus = (
            base_focus
            + track_side * self.camera_focus_lateral_offset
            + Vec3(0.0, 0.0, self.camera_focus_vertical_offset)
        )
        self.camera_focus = smooth_vec3(self.camera_focus, desired_focus, self.camera_focus_lag, dt)

        side_vector = Vec3(-self.camera_forward.y, self.camera_forward.x, 0.0)
        if side_vector.length_squared() > 1e-9:
            side_vector.normalize()
        desired_camera_pos = self.camera_focus - self.camera_forward * self.camera_follow_distance + Vec3(
            0.0,
            0.0,
            self.camera_height,
        ) + side_vector * self.camera_side_offset
        desired_camera_pos.z = max(desired_camera_pos.z, self.camera_focus.z + 2.2)
        desired_look_target = self.camera_focus + self.camera_forward * self.camera_look_ahead
        desired_look_target.z -= self.camera_target_drop

        pos_alpha = smoothing_alpha(self.camera_position_lag, dt)
        look_alpha = smoothing_alpha(self.camera_look_lag, dt)
        camera_pos = self.camera.getPos() * (1.0 - pos_alpha) + desired_camera_pos * pos_alpha
        self.camera_look_target = (
            self.camera_look_target * (1.0 - look_alpha) + desired_look_target * look_alpha
        )

        self.camera.setPos(camera_pos)
        self.camera.lookAt(self.camera_look_target)
        target_roll = -self.steering_input * min(6.4, state.speed * 1.55) * (1.0 - 0.35 * self.camera_collision_blend)
        self.camera_roll = smooth_value(self.camera_roll, target_roll, self.camera_roll_lag, dt)
        self.camera.setR(self.camera_roll)
        self.camLens.setFov(52.0 + min(10.0, state.speed * 1.10) + self.boost_flash * 4.0 + self.start_burst * 3.0)
