from __future__ import annotations

from dataclasses import dataclass
from math import ceil, exp, sin
from pathlib import Path

from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import (
    AmbientLight,
    AntialiasAttrib,
    CardMaker,
    DirectionalLight,
    Fog,
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
    grade_run,
    rate_section,
)
from .physics import (
    GuideObstacle,
    MarbleRampSimulation,
    SimulationConfig,
    boost_pad_center_position,
    is_position_off_track,
    obstacle_center_position,
    ramp_normal,
    ramp_segment_at_distance,
    ramp_side,
    ramp_surface_point,
    ramp_tangent,
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
    spent: bool = False
    pulse: float = 0.0


class MarbleRampApp(ShowBase):
    def __init__(self, config: SimulationConfig | None = None) -> None:
        super().__init__()
        self.sim_config = config or SimulationConfig()
        self.simulation = MarbleRampSimulation(self.sim_config)
        self.paused = False
        self.camera_follow_distance = 7.4
        self.camera_height = 5.2
        self.camera_look_ahead = 3.2
        self.camera_side_offset = -0.25
        self.camera_target_drop = 0.45
        self.camera_focus_lateral_lag = 1.6
        self.camera_focus_vertical_lag = 1.9
        self.camera_position_lag = 4.8
        self.camera_look_lag = 5.2
        self.camera_track_lag = 4.4
        self.camera_heading_response = 2.0
        self.camera_velocity_heading_response = 2.8
        self.camera_collision_attack = 10.0
        self.camera_collision_release = 2.2
        self.camera_roll_lag = 5.5
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
        self.major_impacts = 0
        self.section_major_impacts = 0
        self.run_stability_loss = 0.0
        self.section_stability_loss = 0.0
        self.clean_sections = 0
        self.section_boost_hits = 0
        self.boost_chain = 0
        self.best_boost_chain = 0
        self.off_track_timer = 0.0
        self.off_track_grace = 0.18
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
        self._reset_run_state()
        self._enter_title_state()
        self.taskMgr.add(self._tick, "marble-ramp-tick")
        self._apply_state()

    def _setup_window(self) -> None:
        props = WindowProperties()
        props.setSize(1440, 900)
        props.setTitle("Marble Physics Run")
        if self.win and hasattr(self.win, "requestProperties"):
            self.win.requestProperties(props)
        self.setBackgroundColor(0.02, 0.03, 0.05, 1.0)
        self.render.setAntialias(AntialiasAttrib.MAuto)
        self.render.setShaderAuto()
        if patch_loader is not None:
            patch_loader(self.loader)
        if simplepbr is not None and self.win is not None:
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
        finish_glow_np = self.render.attachNewNode(finish_glow)
        finish_glow_np.setPos(self.sim_config.length - 1.5, 0.0, 3.2)
        self.render.setLight(finish_glow_np)

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
        self._attach_centered_box(aura, size=Vec3(2.8, 4.8, 0.04), color=Vec4(0.18, 0.22, 0.28, 1.0))
        aura.setPos(self.sim_config.length - 1.2, 0.0, 0.08)

    def _build_ground(self) -> None:
        plinth = self.scene_root.attachNewNode("plinth")
        self._attach_centered_box(
            plinth,
            size=Vec3(self.sim_config.length + 11.0, 16.5, 0.9),
            color=Vec4(0.08, 0.09, 0.11, 1.0),
            style="panel",
        )
        plinth.setPos(self.sim_config.length * 0.5, 0.0, -0.46)

        top = self.scene_root.attachNewNode("plinth-top")
        self._attach_centered_box(
            top,
            size=Vec3(self.sim_config.length + 9.2, 14.2, 0.10),
            color=Vec4(0.18, 0.19, 0.22, 1.0),
            style="panel",
        )
        top.setPos(self.sim_config.length * 0.5, 0.0, 0.02)

        runway = self.scene_root.attachNewNode("runway")
        self._attach_centered_box(
            runway,
            size=Vec3(self.sim_config.length + 6.8, 4.9, 0.04),
            color=Vec4(0.10, 0.13, 0.18, 1.0),
            style="runway",
        )
        runway.setPos(self.sim_config.length * 0.5, 0.0, 0.08)

        for lane_index in range(6):
            lane_light = self.scene_root.attachNewNode(f"runway-line-{lane_index}")
            self._attach_centered_box(
                lane_light,
                size=Vec3(0.75, 0.08, 0.012),
                color=Vec4(0.18, 0.70, 0.94, 1.0),
            )
            lane_light.setPos(2.4 + lane_index * 6.0, 0.0, 0.105)

    def _build_ramp(self) -> None:
        for index, (segment_np, _, segment) in enumerate(self.simulation.ramp_nodes):
            ramp = self.scene_root.attachNewNode(f"ramp-{index}")
            ramp.setH(segment.heading_deg)
            ramp.setR(self.sim_config.angle_deg)
            ramp.setPos(segment_np.getPos())

            surface = ramp.attachNewNode("surface")
            self._attach_centered_box(
                surface,
                size=Vec3(segment.length * 0.99, self.sim_config.width * 0.82, 0.028),
                color=Vec4(0.17, 0.18, 0.20, 1.0),
                style="track",
            )
            surface.setZ(self.sim_config.ramp_thickness * 0.46)

            lane = ramp.attachNewNode("lane")
            self._attach_centered_box(
                lane,
                size=Vec3(segment.length * 0.92, self.sim_config.width * 0.08, 0.014),
                color=Vec4(0.82, 0.84, 0.88, 1.0),
                style="screen",
            )
            lane.setZ(self.sim_config.ramp_thickness * 0.56)

            deck = ramp.attachNewNode("deck")
            self._attach_centered_box(
                deck,
                size=Vec3(segment.length * 0.90, self.sim_config.width * 0.42, 0.12),
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
                trim.setPos(0.0, side * (self.sim_config.width * 0.41), self.sim_config.ramp_thickness * 0.43)

                skirt = ramp.attachNewNode(f"skirt-{side}")
                self._attach_centered_box(
                    skirt,
                    size=Vec3(segment.length * 0.92, 0.05, 0.18),
                    color=Vec4(0.10, 0.12, 0.15, 1.0),
                    style="panel",
                )
                skirt.setPos(0.0, side * (self.sim_config.width * 0.34), -0.04)

            support = ramp.attachNewNode("support")
            self._load_styled_model(
                support,
                "box",
                scale=Vec3(segment.length * 0.16, self.sim_config.width * 0.18, 0.64),
                color=Vec4(0.08, 0.10, 0.12, 1.0),
                pos=Vec3(0.0, 0.0, -0.70),
                style="panel",
            )

            underlight = support.attachNewNode("underlight")
            self._attach_centered_box(
                underlight,
                size=Vec3(segment.length * 0.10, self.sim_config.width * 0.36, 0.04),
                color=Vec4(0.12, 0.14, 0.18, 1.0),
                style="screen",
            )
            underlight.setPos(0.0, 0.0, -0.30)

    def _build_rails(self) -> None:
        for index, (rail_np, _, segment, lateral_offset) in enumerate(self.simulation.rail_nodes):
            rail = self.scene_root.attachNewNode(f"rail-{index}")
            rail.setH(segment.heading_deg)
            rail.setR(self.sim_config.angle_deg)
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

            for post_offset in (-segment.length * 0.32, 0.0, segment.length * 0.32):
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
            gate.setH(segment.heading_deg)
            gate.setR(self.sim_config.angle_deg)

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
            pad.setH(segment.heading_deg)
            pad.setR(self.sim_config.angle_deg)

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
        gate.setPos(self.sim_config.length - 1.0, 0.0, 0.0)
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

        pad = gate.attachNewNode("finish-pad")
        self._attach_centered_box(pad, size=Vec3(0.24, self.sim_config.width * 0.82, 0.02), color=Vec4(0.24, 0.70, 0.92, 1.0))
        pad.setPos(0.05, 0.0, 0.08)

    def _build_obstacles(self) -> None:
        colors = {
            "block": Vec4(0.96, 0.42, 0.18, 1.0),
            "pencil": Vec4(0.18, 0.70, 0.94, 1.0),
            "eraser": Vec4(0.84, 0.40, 0.94, 1.0),
            "pen": Vec4(0.18, 0.70, 0.94, 1.0),
            "ruler": Vec4(0.96, 0.74, 0.20, 1.0),
        }
        for obstacle in self.sim_config.obstacles:
            self._build_obstacle(obstacle, colors.get(obstacle.kind, Vec4(0.45, 0.45, 0.45, 1.0)))

    def _build_obstacle(self, obstacle: GuideObstacle, color: Vec4) -> None:
        self._build_obstacle_marker(obstacle, color)
        obstacle_np = self.scene_root.attachNewNode("obstacle")
        self._attach_obstacle_visual(obstacle_np, obstacle, color)
        obstacle_np.setH(obstacle.heading_deg)
        obstacle_np.setR(self.sim_config.angle_deg)
        obstacle_np.setPos(obstacle_center_position(self.sim_config, obstacle))

    def _build_obstacle_marker(self, obstacle: GuideObstacle, color: Vec4) -> None:
        marker_distance = max(0.9, obstacle.distance_along_ramp - max(0.9, obstacle.length * 1.05))
        segment = ramp_segment_at_distance(self.sim_config, marker_distance)
        marker = self.scene_root.attachNewNode("obstacle-marker")
        marker.setPos(
            ramp_surface_point(self.sim_config, marker_distance)
            + ramp_side(self.sim_config, marker_distance) * obstacle.lateral_offset
            + ramp_normal(self.sim_config, marker_distance) * 0.02
        )
        marker.setH(segment.heading_deg)
        marker.setR(self.sim_config.angle_deg)

        plate_width = min(self.sim_config.width * 0.78, obstacle.width + 0.42)
        self._attach_centered_box(
            marker,
            size=Vec3(0.52, plate_width, 0.012),
            color=Vec4(color.x * 0.95, color.y * 0.95, color.z * 0.95, 1.0),
            style="screen",
        )

        center_line = marker.attachNewNode("center-line")
        self._attach_centered_box(
            center_line,
            size=Vec3(0.36, min(plate_width * 0.34, 0.14), 0.014),
            color=Vec4(0.95, 0.96, 0.98, 1.0),
            style="screen",
        )
        center_line.setZ(0.01)

    def _attach_obstacle_visual(self, parent: NodePath, obstacle: GuideObstacle, color: Vec4) -> None:
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
        for index in range(1):
            line = self.marble_root.attachNewNode(f"speed-line-{index}")
            self._attach_centered_box(
                line,
                size=Vec3(self.sim_config.marble_radius * 1.5, 0.016, 0.016),
                color=Vec4(0.76, 0.80, 0.86, 1.0),
            )
            line.setPos(-self.sim_config.marble_radius * 1.1, 0.0, -0.02)
            self.speed_lines.append(line)

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
            mayChange=False,
        )
        self.subtitle = OnscreenText(
            text="Simple long track. Sparse obstacles. Clean resets.",
            pos=(-1.30, 0.85),
            align=TextNode.ALeft,
            scale=0.030,
            fg=(0.66, 0.72, 0.80, 1.0),
            mayChange=False,
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
            text="Left/Right steer   Down brake   Space start/pause   R restart   Esc quit",
            pos=(0.0, -0.94),
            scale=0.030,
            fg=(0.70, 0.74, 0.80, 1.0),
            mayChange=False,
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
        self.accept("arrow_down", self._set_brake, [1.0])
        self.accept("arrow_down-up", self._set_brake, [0.0])
        self.accept("space", self._handle_space)
        self.accept("r", self._reset)
        self.accept("enter", self._handle_space)
        self.accept("tab", self._toggle_debug_overlay)
        self.accept("escape", self.userExit)

    def _set_steering(self, direction: float) -> None:
        self.steering_input = direction
        self.simulation.set_steering_input(direction)

    def _release_steering(self, direction: float) -> None:
        if self.steering_input == direction:
            self.steering_input = 0.0
            self.simulation.set_steering_input(0.0)

    def _set_brake(self, amount: float) -> None:
        self.brake_input = amount
        self.simulation.set_brake_input(amount)

    def _handle_space(self) -> None:
        if self.race_phase in {"title", "finished"}:
            self._reset()
            return
        if self.race_phase == "running":
            self.paused = not self.paused
            return
        if self.paused:
            self.paused = False

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

    def _enter_title_state(self) -> None:
        self.race_phase = "title"
        self.paused = False
        self.current_section_name = "Long Course"
        self.current_section_focus = "Press Space to start"
        self._show_event(
            "MARBLE PHYSICS RUN",
            "Press Space to start",
            Vec4(0.92, 0.94, 0.98, 1.0),
            hold=60.0,
        )

    def _reset_run_state(self) -> None:
        self.race_phase = "countdown"
        self.countdown_timer = 3.6
        self.last_countdown_value = int(ceil(self.countdown_timer))
        self.finish_time = None
        self.finish_grade = None
        self.major_impacts = 0
        self.section_major_impacts = 0
        self.run_stability_loss = 0.0
        self.section_stability_loss = 0.0
        self.clean_sections = 0
        self.section_boost_hits = 0
        self.boost_chain = 0
        self.best_boost_chain = 0
        self.off_track_timer = 0.0
        self.last_supported_path_distance = 0.0
        self.impact_event_cooldown = 0.0
        self.next_section_index = 0
        self.section_results.clear()
        self.current_section_name = self.course_sections[0].name
        self.current_section_focus = self.course_sections[0].focus
        self.flash_alpha = 0.0
        self.boost_flash = 0.0
        self.finish_flash = 0.0
        self._show_event("3", "", Vec4(0.92, 0.94, 0.98, 1.0), hold=10.0)
        for pad in self.boost_pads:
            pad.spent = False
            pad.pulse = 0.0

    def _reset(self) -> None:
        self.simulation.reset()
        self.paused = False
        self.steering_input = 0.0
        self.brake_input = 0.0
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
        self._reset_run_state()
        self._apply_state(1.0 / 60.0)

    def _update_feedback_state(self, dt: float) -> None:
        self.flash_alpha = max(0.0, self.flash_alpha - dt * 1.7)
        self.boost_flash = max(0.0, self.boost_flash - dt * 1.8)
        self.finish_flash = max(0.0, self.finish_flash - dt * 0.75)
        self.impact_event_cooldown = max(0.0, self.impact_event_cooldown - dt)
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
            self._show_event(
                str(countdown_value),
                "",
                Vec4(0.92, 0.94, 0.98, 1.0),
                hold=10.0,
            )
        if self.countdown_timer <= 0.0:
            self.race_phase = "running"
            self._show_event("GO!", "", Vec4(0.92, 0.94, 0.98, 1.0), hold=0.9)

    def _activate_boost_pad(self, pad: BoostPadVisual, index: int) -> None:
        if pad.spent:
            return
        pad.spent = True
        pad.pulse = 1.0
        self.section_boost_hits += 1
        self.boost_chain += 1
        self.best_boost_chain = max(self.best_boost_chain, self.boost_chain)
        self.boost_flash = max(self.boost_flash, 0.35)

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
        self.race_phase = "finished"
        self.current_section_name = "Finished"
        self.current_section_focus = ""
        self.finish_time = state.time
        previous_best = self.session_best_time
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
        if is_best:
            self.session_best_time = state.time
        subtitle = format_seconds(state.time)
        if is_best:
            subtitle += "   NEW BEST"
        elif previous_best is not None:
            subtitle += f"   {format_delta(state.time - previous_best)}"
        self._show_event("FINISH", subtitle, Vec4(0.92, 0.94, 0.98, 1.0), hold=60.0)
        self.finish_flash = 0.25

    def _should_reset_for_off_track(self, previous_state, state, dt: float) -> bool:
        reference_distance = max(self.last_supported_path_distance, previous_state.path_distance)
        surface_point = ramp_surface_point(self.sim_config, reference_distance)
        relative = state.position - surface_point
        lateral_offset = abs(relative.dot(ramp_side(self.sim_config, reference_distance)))
        normal_offset = relative.dot(ramp_normal(self.sim_config, reference_distance))
        vertical_drop = surface_point.z - state.position.z

        if (
            lateral_offset <= self.sim_config.width * 0.5 + 0.08
            and normal_offset >= -0.06
            and vertical_drop <= 0.35
        ):
            self.last_supported_path_distance = max(self.last_supported_path_distance, state.path_distance)

        off_track_candidate = is_position_off_track(
            self.sim_config,
            state.position,
            reference_distance=reference_distance,
        )
        airborne_recovery = (
            state.airborne
            and lateral_offset <= self.sim_config.width * 0.5 + 0.18
            and normal_offset > -0.06
            and vertical_drop < 0.65
        )
        hard_off_track = (
            lateral_offset > self.sim_config.width * 0.5 + 0.42
            or normal_offset < -0.52
            or vertical_drop > 1.35
            or state.position.z < -2.0
        )

        if off_track_candidate and not airborne_recovery:
            self.off_track_timer += dt * (3.0 if hard_off_track else 1.0)
        else:
            self.off_track_timer = max(0.0, self.off_track_timer - dt * 2.5)

        return self.off_track_timer >= self.off_track_grace

    def _handle_running_state(self, previous_state, state, dt: float) -> None:
        if state.boost_active and not previous_state.boost_active:
            active_index = self.simulation.active_boost_pad_index
            if active_index is not None and active_index < len(self.boost_pads):
                self._activate_boost_pad(self.boost_pads[active_index], active_index)

        if state.path_distance >= self.sim_config.length - self.sim_config.marble_radius:
            self._finish_run(state)
            return

        if self._should_reset_for_off_track(previous_state, state, dt):
            self._reset()
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

    def _tick(self, task: Task) -> int:
        dt = min(globalClock.getDt(), 1.0 / 120.0)
        self._update_feedback_state(dt)
        if not self.paused:
            if self.race_phase == "countdown":
                self._handle_countdown(dt)
            elif self.race_phase == "running":
                previous_state = self.simulation.snapshot()
                self.simulation.step(dt)
                self._handle_running_state(previous_state, self.simulation.snapshot(), dt)
        self._animate_props(task.time, dt)
        self._apply_state(dt)
        return Task.cont

    def _animate_props(self, time_s: float, dt: float) -> None:
        state = self.simulation.snapshot()
        speed = state.speed
        for pulse in self.pulse_nodes:
            lift = 1.0 + pulse.amplitude * (0.5 + 0.5 * sin(time_s * pulse.speed + pulse.phase))
            lift += self.boost_flash * 0.04 + self.finish_flash * 0.06
            pulse.node.setColorScale(
                min(1.08, pulse.base_color.x * lift),
                min(1.08, pulse.base_color.y * lift),
                min(1.08, pulse.base_color.z * lift),
                pulse.base_color.w,
            )

        for index, line in enumerate(self.speed_lines):
            speed_scale = 0.55 + min(0.26, speed * 0.02)
            speed_scale += self.boost_flash * 0.08
            alpha = min(0.20, 0.04 + speed * 0.012 + self.boost_flash * 0.05)
            line.setScale(speed_scale, 1.0, 1.0)
            line.setColor(0.78, 0.82, 0.88, alpha)

        for pad in self.boost_pads:
            pulse_scale = 1.0 + pad.pulse * 0.04
            ambient_scale = 0.94
            if pad.spent:
                ambient_scale *= 0.82
            pad.root.setScale(pulse_scale)
            pad.root.setColorScale(
                ambient_scale + pad.pulse * 0.08,
                ambient_scale + pad.pulse * 0.08,
                ambient_scale + pad.pulse * 0.10,
                1.0,
            )

    def _apply_state(self, dt: float | None = None) -> None:
        state = self.simulation.snapshot()
        self.marble_root.setPos(state.position)
        self.marble_root.setQuat(self.simulation.marble_np.getQuat())
        camera_dt = 1.0 / 60.0 if dt is None else min(max(dt, 0.0), 1.0 / 30.0)
        self._update_camera(state, camera_dt)
        progress = min(100.0, (state.path_distance / self.sim_config.length) * 100.0)
        best_text = "--" if self.session_best_time is None else format_seconds(self.session_best_time)
        self.hud.setText(
            (
                f"time   {format_seconds(state.time):>7}\n"
                f"speed  {state.speed:5.2f} m/s\n"
                f"track  {progress:5.1f}%"
            )
        )
        self.status.setText("")
        if self.finish_grade is not None and self.finish_time is not None:
            self.result.setText(
                (
                    f"run   {format_seconds(self.finish_time)}\n"
                    f"best  {best_text}"
                )
            )
            self.result.setFg((0.92, 0.94, 0.98, 1.0))
        else:
            self.result.setText("")

        center_text = self.event_text
        center_subtitle = self.event_subtitle
        if self.race_phase == "title":
            center_text = self.event_text
            center_subtitle = self.event_subtitle
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
        self.progress_caption.setText("")
        self.progress_fill.setScale(max(0.001, progress / 100.0), 1.0, 1.0)
        self.progress_fill.setColor(0.82, 0.84, 0.88, 0.95)
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

        flash_alpha = max(self.flash_alpha, self.boost_flash * 0.10, self.finish_flash * 0.16)
        flash_color = Vec4(self.flash_color)
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

    def _load_styled_model(
        self,
        parent: NodePath,
        model_name: str,
        *,
        scale: Vec3,
        color: Vec4,
        pos: Vec3 | None = None,
        style: str | None = None,
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
        return model

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
                    image.set_xel_a(x, y, v * 0.6, v * 0.75, v, 1.0)
        elif style == "track":
            for y in range(256):
                for x in range(256):
                    pattern = 0.02 if (x + y) % 24 < 2 or (x - y) % 28 < 2 else 0.0
                    seam = 0.08 if y % 64 < 2 else 0.0
                    v = 0.12 + pattern + seam
                    image.set_xel_a(x, y, v * 0.95, v, v * 1.1, 1.0)
        elif style == "screen":
            for y in range(256):
                t = y / 255.0
                for x in range(256):
                    pulse = 0.18 if x % 32 < 4 else 0.0
                    image.set_xel_a(
                        x,
                        y,
                        0.22 + 0.18 * (1.0 - t) + pulse,
                        0.42 + 0.22 * (1.0 - t) + pulse * 0.6,
                        0.58 + 0.24 * (1.0 - t),
                        1.0,
                    )
        else:
            for y in range(256):
                for x in range(256):
                    seam = 0.10 if x % 64 < 2 or y % 64 < 2 else 0.0
                    grain = (((x * 13) + (y * 7)) % 19) / 255.0
                    v = 0.10 + seam + grain
                    image.set_xel_a(x, y, v * 0.9, v * 0.95, v, 1.0)

        texture = Texture(style)
        texture.load(image)
        texture.setWrapU(SamplerState.WM_repeat)
        texture.setWrapV(SamplerState.WM_repeat)
        texture.setMinfilter(SamplerState.FT_linear_mipmap_linear)
        texture.setMagfilter(SamplerState.FT_linear)
        self._texture_cache[style] = texture
        return texture

    def _update_camera(self, state, dt: float) -> None:
        if self.camera is None or self.camLens is None:
            return
        track_response = self.camera_track_lag * (1.0 - 0.45 * self.camera_collision_blend)
        track_alpha = smoothing_alpha(track_response, dt)
        target_track_distance = max(self.camera_track_distance, max(0.0, state.path_distance - 0.40))
        self.camera_track_distance += (target_track_distance - self.camera_track_distance) * track_alpha

        track_forward = ramp_tangent(self.sim_config, self.camera_track_distance)
        track_forward.z = 0.0
        if track_forward.length_squared() > 1e-9:
            track_forward.normalize()

        desired_velocity_forward = Vec3(track_forward)
        velocity_forward = Vec3(state.linear_velocity.x, state.linear_velocity.y, 0.0)
        if velocity_forward.length_squared() > 0.16:
            velocity_forward.normalize()
            desired_velocity_forward = velocity_forward
        velocity_response = self.camera_velocity_heading_response * (1.0 - 0.55 * self.camera_collision_blend)
        velocity_alpha = smoothing_alpha(velocity_response, dt)
        self.camera_velocity_forward = (
            self.camera_velocity_forward * (1.0 - velocity_alpha) + desired_velocity_forward * velocity_alpha
        )
        if self.camera_velocity_forward.length_squared() > 1e-9:
            self.camera_velocity_forward.normalize()

        self.camera_collision_blend = smooth_binary_state(
            self.camera_collision_blend,
            state.total_contacts > 0,
            dt,
            attack_response=self.camera_collision_attack,
            release_response=self.camera_collision_release,
        )
        velocity_mix = min(0.18, state.speed * 0.025) * (1.0 - 0.92 * self.camera_collision_blend)
        target_forward = track_forward * (1.0 - velocity_mix) + self.camera_velocity_forward * velocity_mix
        if target_forward.length_squared() > 1e-9:
            target_forward.normalize()

        heading_alpha = smoothing_alpha(self.camera_heading_response, dt)
        self.camera_forward = self.camera_forward * (1.0 - heading_alpha) + target_forward * heading_alpha
        self.camera_forward.normalize()

        track_point = ramp_surface_point(self.sim_config, self.camera_track_distance)
        track_side = ramp_side(self.sim_config, self.camera_track_distance)
        track_normal = ramp_normal(self.sim_config, self.camera_track_distance)
        track_focus = track_point + track_normal * (self.sim_config.marble_radius + 0.05)
        collision_damping = 1.0 - 0.82 * self.camera_collision_blend
        lateral_error = max(
            -self.sim_config.width * 0.16,
            min(
                self.sim_config.width * 0.16,
                (state.position - track_focus).dot(track_side),
            ),
        ) * collision_damping
        vertical_error = max(
            -0.04,
            min(
                0.18,
                state.position.z - track_focus.z,
            ),
        ) * collision_damping

        focus_response_scale = 1.0 - 0.60 * self.camera_collision_blend
        lateral_alpha = smoothing_alpha(self.camera_focus_lateral_lag * focus_response_scale, dt)
        vertical_alpha = smoothing_alpha(self.camera_focus_vertical_lag * focus_response_scale, dt)
        self.camera_focus_lateral_offset += (
            lateral_error - self.camera_focus_lateral_offset
        ) * lateral_alpha
        self.camera_focus_vertical_offset += (
            vertical_error - self.camera_focus_vertical_offset
        ) * vertical_alpha

        desired_focus = (
            track_focus
            + track_side * self.camera_focus_lateral_offset
            + Vec3(0.0, 0.0, self.camera_focus_vertical_offset)
        )
        self.camera_focus = Vec3(desired_focus)
        speed_boost = min(1.2, state.speed * 0.06)
        side_vector = self.camera_forward.cross(Vec3(0.0, 0.0, 1.0))
        if side_vector.length_squared() > 1e-9:
            side_vector.normalize()
        desired_camera_pos = desired_focus - self.camera_forward * (self.camera_follow_distance + speed_boost) + Vec3(
            0.0,
            0.0,
            self.camera_height + speed_boost * 0.25,
        ) + side_vector * self.camera_side_offset
        desired_camera_pos.z = max(desired_camera_pos.z, desired_focus.z + 2.2)
        desired_look_target = desired_focus + self.camera_forward * (self.camera_look_ahead + speed_boost * 0.45)
        desired_look_target.z -= self.camera_target_drop

        pos_alpha = smoothing_alpha(self.camera_position_lag, dt)
        look_alpha = smoothing_alpha(self.camera_look_lag, dt)
        camera_pos = self.camera.getPos() * (1.0 - pos_alpha) + desired_camera_pos * pos_alpha
        self.camera_look_target = (
            self.camera_look_target * (1.0 - look_alpha) + desired_look_target * look_alpha
        )

        self.camera.setPos(camera_pos)
        self.camera.lookAt(self.camera_look_target)
        target_roll = -self.steering_input * min(4.0, state.speed * 1.2) * (1.0 - 0.35 * self.camera_collision_blend)
        self.camera_roll = smooth_value(self.camera_roll, target_roll, self.camera_roll_lag, dt)
        self.camera.setR(self.camera_roll)
        self.camLens.setFov(48.0 + min(3.5, state.speed * 0.55))
