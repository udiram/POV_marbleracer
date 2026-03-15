from __future__ import annotations

from math import exp, sin
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
    Vec3,
    Vec4,
    WindowProperties,
    loadPrcFileData,
)

from .physics import (
    GuideObstacle,
    MarbleRampSimulation,
    SimulationConfig,
    obstacle_center_position,
    ramp_normal,
    ramp_side,
    ramp_surface_point,
    ramp_tangent,
    required_static_friction,
)

try:
    import simplepbr
except ImportError:  # pragma: no cover - optional dependency
    simplepbr = None

try:
    from gltf import patch_loader
except ImportError:  # pragma: no cover - optional dependency
    patch_loader = None


loadPrcFileData("", "window-title Neon Desk Marble Run")
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
        self._texture_cache: dict[str, Texture] = {}
        start_pos = self.simulation.snapshot().position
        self.camera_focus = Vec3(start_pos)
        self.camera_look_target = Vec3(start_pos) + self.camera_forward * self.camera_look_ahead
        self.disableMouse()

        self._setup_window()
        self._setup_scene()
        self._build_world()
        self._build_ui()
        self._setup_input()
        self.taskMgr.add(self._tick, "marble-ramp-tick")
        self._apply_state()

    def _setup_window(self) -> None:
        props = WindowProperties()
        props.setSize(1440, 900)
        props.setTitle("Neon Desk Marble Run")
        if self.win and hasattr(self.win, "requestProperties"):
            self.win.requestProperties(props)
        self.setBackgroundColor(0.02, 0.03, 0.05, 1.0)
        self.render.setAntialias(AntialiasAttrib.MAuto)
        self.render.setShaderAuto()
        if patch_loader is not None:
            patch_loader(self.loader)
        if simplepbr is not None:
            simplepbr.init(enable_shadows=False, use_occlusion_maps=False, max_lights=8)

    def _setup_scene(self) -> None:
        fog = Fog("desk-fog")
        fog.setColor(0.04, 0.05, 0.08)
        fog.setExpDensity(0.0075)
        self.render.setFog(fog)

        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.22, 0.24, 0.30, 1.0))
        self.render.setLight(self.render.attachNewNode(ambient))

        sun = DirectionalLight("sun")
        sun.setColor(Vec4(1.00, 0.82, 0.58, 1.0))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(-38, -34, 0)
        self.render.setLight(sun_np)

        fill = DirectionalLight("fill")
        fill.setColor(Vec4(0.16, 0.28, 0.44, 1.0))
        fill_np = self.render.attachNewNode(fill)
        fill_np.setHpr(128, -18, 0)
        self.render.setLight(fill_np)

        rim = PointLight("rim")
        rim.setColor(Vec4(0.22, 0.76, 0.96, 1.0))
        rim_np = self.render.attachNewNode(rim)
        rim_np.setPos(self.sim_config.length * 0.82, -7.0, 6.0)
        self.render.setLight(rim_np)

        finish_glow = PointLight("finish-glow")
        finish_glow.setColor(Vec4(1.0, 0.42, 0.18, 1.0))
        finish_glow_np = self.render.attachNewNode(finish_glow)
        finish_glow_np.setPos(self.sim_config.length - 1.5, 0.0, 4.2)
        self.render.setLight(finish_glow_np)

        overhead = PointLight("overhead")
        overhead.setColor(Vec4(0.60, 0.66, 0.84, 1.0))
        overhead_np = self.render.attachNewNode(overhead)
        overhead_np.setPos(self.sim_config.length * 0.42, 0.0, 14.0)
        self.render.setLight(overhead_np)

    def _build_world(self) -> None:
        self.scene_root = self.render.attachNewNode("scene-root")
        self._build_backdrop()
        self._build_ground()
        self._build_ramp()
        self._build_rails()
        self._build_finish_gate()
        self._build_obstacles()
        self._build_desk_props()
        self._build_start_zone()
        self._build_marble()
        self._setup_camera()

    def _build_backdrop(self) -> None:
        for index, (width, height, x, y, z, color, h) in enumerate(
            (
                (44.0, 18.0, self.sim_config.length + 14.0, 0.0, 8.5, Vec4(0.07, 0.09, 0.14, 1.0), 90.0),
                (30.0, 10.0, self.sim_config.length + 8.0, -10.0, 4.5, Vec4(0.05, 0.07, 0.12, 1.0), 120.0),
                (30.0, 10.0, self.sim_config.length + 8.0, 10.0, 4.5, Vec4(0.05, 0.07, 0.12, 1.0), 60.0),
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
        self._attach_centered_box(aura, size=Vec3(3.2, 6.4, 0.08), color=Vec4(0.95, 0.36, 0.16, 1.0))
        aura.setPos(self.sim_config.length - 1.2, 0.0, 0.12)

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

        for lane_index in range(18):
            lane_light = self.scene_root.attachNewNode(f"runway-line-{lane_index}")
            self._attach_centered_box(
                lane_light,
                size=Vec3(0.75, 0.08, 0.012),
                color=Vec4(0.96, 0.42, 0.18 if lane_index % 2 == 0 else 0.44, 1.0),
            )
            lane_light.setPos(1.2 + lane_index * 1.35, -2.0, 0.105)
            mirror = self.scene_root.attachNewNode(f"runway-line-mirror-{lane_index}")
            self._attach_centered_box(
                mirror,
                size=Vec3(0.75, 0.08, 0.012),
                color=Vec4(0.18, 0.70, 0.94, 1.0),
            )
            mirror.setPos(1.2 + lane_index * 1.35, 2.0, 0.105)

        for rib_index in range(9):
            rib = self.scene_root.attachNewNode(f"plinth-rib-{rib_index}")
            self._attach_centered_box(
                rib,
                size=Vec3(0.14, 14.0, 0.24),
                color=Vec4(0.05, 0.06, 0.08, 1.0),
                style="panel",
            )
            rib.setPos(1.5 + rib_index * 3.0, 0.0, -0.2)

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
                color=Vec4(0.14, 0.16, 0.18, 1.0),
                style="track",
            )
            surface.setZ(self.sim_config.ramp_thickness * 0.46)

            lane = ramp.attachNewNode("lane")
            self._attach_centered_box(
                lane,
                size=Vec3(segment.length * 0.92, self.sim_config.width * 0.18, 0.016),
                color=Vec4(0.92, 0.46, 0.20, 1.0),
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
                    color=Vec4(0.14, 0.68 if side > 0 else 0.52, 0.92, 1.0),
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
                color=Vec4(0.16, 0.62, 0.88, 1.0),
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
                color=Vec4(0.18, 0.68 if lateral_offset > 0 else 0.48, 0.96, 1.0),
                style="screen",
            )
            glow.setZ(self.sim_config.rail_height * 0.34)

            banner = rail.attachNewNode("banner")
            self._attach_centered_box(
                banner,
                size=Vec3(segment.length * 0.12, 0.05, self.sim_config.rail_height * 0.44),
                color=Vec4(0.95, 0.42 if lateral_offset < 0 else 0.66, 0.18, 1.0),
                style="screen",
            )
            banner.setPos(0.0, 0.0, self.sim_config.rail_height * 0.12)

            for post_offset in (-segment.length * 0.32, 0.0, segment.length * 0.32):
                post = rail.attachNewNode("post")
                self._attach_centered_box(
                    post,
                    size=Vec3(0.08, 0.08, self.sim_config.rail_height * 1.04),
                    color=Vec4(0.18, 0.22, 0.28, 1.0),
                    style="panel",
                )
                post.setPos(post_offset, 0.0, 0.0)

    def _build_finish_gate(self) -> None:
        gate = self.scene_root.attachNewNode("finish-gate")
        gate.setPos(self.sim_config.length - 1.0, 0.0, 0.0)
        for side in (-1, 1):
            post = gate.attachNewNode(f"post-{side}")
            self._attach_centered_box(
                post,
                size=Vec3(0.18, 0.18, 3.4),
                color=Vec4(0.80, 0.84, 0.92, 1.0),
                style="panel",
            )
            post.setPos(0.0, side * 1.7, 1.7)

        top_bar = gate.attachNewNode("top-bar")
        self._attach_centered_box(top_bar, size=Vec3(0.22, 4.1, 0.28), color=Vec4(0.10, 0.12, 0.16, 1.0), style="panel")
        top_bar.setPos(0.0, 0.0, 3.55)

        crest = gate.attachNewNode("crest")
        self._attach_centered_box(crest, size=Vec3(0.10, 3.0, 0.14), color=Vec4(0.94, 0.42, 0.18, 1.0), style="screen")
        crest.setPos(0.0, 0.0, 3.55)

        header = gate.attachNewNode("header")
        self._attach_centered_box(header, size=Vec3(0.10, 2.2, 0.42), color=Vec4(0.18, 0.70, 0.96, 1.0), style="screen")
        header.setPos(0.0, 0.0, 4.2)

        for side in (-1, 1):
            spinner = gate.attachNewNode(f"spinner-{side}")
            self._attach_centered_box(
                spinner,
                size=Vec3(0.10, 0.42, 0.10),
                color=Vec4(0.20, 0.72 if side > 0 else 0.42, 0.94, 1.0),
                style="screen",
            )
            spinner.setPos(0.0, side * 2.3, 3.0)
            self.prop_spinners.append(spinner)

        pad = gate.attachNewNode("finish-pad")
        self._attach_centered_box(pad, size=Vec3(0.3, self.sim_config.width * 0.88, 0.03), color=Vec4(0.95, 0.42, 0.18, 1.0))
        pad.setPos(0.05, 0.0, 0.10)

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
        obstacle_np = self.scene_root.attachNewNode("obstacle")
        self._attach_obstacle_visual(obstacle_np, obstacle, color)
        obstacle_np.setH(obstacle.heading_deg)
        obstacle_np.setR(self.sim_config.angle_deg)
        obstacle_np.setPos(obstacle_center_position(self.sim_config, obstacle))

    def _attach_obstacle_visual(self, parent: NodePath, obstacle: GuideObstacle, color: Vec4) -> None:
        size = Vec3(obstacle.length, obstacle.width, obstacle.height)
        if obstacle.kind == "pencil":
            self._attach_centered_box(
                parent,
                size=Vec3(size.x * 0.92, size.y * 0.22, size.z * 0.34),
                color=Vec4(0.16, 0.18, 0.22, 1.0),
                style="panel",
            )
            for side in (-1, 1):
                cap = parent.attachNewNode(f"cap-{side}")
                self._attach_centered_box(
                    cap,
                    size=Vec3(size.x * 0.12, size.y * 0.28, size.z * 0.40),
                    color=color,
                    style="screen",
                )
                cap.setPos(side * size.x * 0.40, 0.0, 0.0)
            spine = parent.attachNewNode("spine")
            self._attach_centered_box(spine, size=Vec3(size.x * 0.68, size.y * 0.24, size.z * 0.22), color=color, style="screen")
            spine.setPos(0.0, 0.0, size.z * 0.34)
            return
        if obstacle.kind == "pen":
            self._attach_centered_box(
                parent,
                size=Vec3(size.x * 0.88, size.y * 0.18, size.z * 0.28),
                color=Vec4(0.10, 0.12, 0.16, 1.0),
                style="panel",
            )
            collar = parent.attachNewNode("collar")
            self._attach_centered_box(collar, size=Vec3(size.x * 0.18, size.y * 0.88, size.z * 0.88), color=color, style="screen")
            collar.setX(-size.x * 0.28)
            fin = parent.attachNewNode("fin")
            self._attach_centered_box(fin, size=Vec3(size.x * 0.58, size.y * 0.18, size.z * 0.24), color=color, style="screen")
            fin.setPos(size.x * 0.08, 0.0, size.z * 0.30)
            return
        if obstacle.kind == "ruler":
            self._attach_centered_box(parent, size=size, color=Vec4(0.10, 0.11, 0.15, 1.0), style="panel")
            for mark_index in range(5):
                mark = parent.attachNewNode(f"mark-{mark_index}")
                self._attach_centered_box(
                    mark,
                    size=Vec3(size.x * 0.10, size.y * 0.16, size.z * 0.70),
                    color=color,
                    style="screen",
                )
                mark.setPos(-size.x * 0.36 + mark_index * size.x * 0.18, 0.0, size.z * 0.10)
            return
        self._attach_centered_box(parent, size=size, color=Vec4(0.12, 0.14, 0.18, 1.0), style="panel")
        crown = parent.attachNewNode("crown")
        self._attach_centered_box(
            crown,
            size=Vec3(size.x * 0.84, size.y * 0.36, size.z * 0.18),
            color=color,
            style="screen",
        )
        crown.setPos(0.0, 0.0, size.z * 0.42)
        for side in (-1, 1):
            strip = parent.attachNewNode(f"strip-{side}")
            self._attach_centered_box(
                strip,
                size=Vec3(size.x * 0.72, size.y * 0.10, size.z * 0.14),
                color=Vec4(0.74, 0.80, 0.88, 1.0),
                style="screen",
            )
            strip.setPos(0.0, side * size.y * 0.28, -size.z * 0.18)

    def _build_desk_props(self) -> None:
        self._build_arena_shell()

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
            color=Vec4(0.18, 0.70, 0.96, 1.0),
            style="screen",
        )
        line.setZ(0.04)

        for side in (-1, 1):
            cap = self.scene_root.attachNewNode(f"start-cap-{side}")
            self._attach_centered_box(
                cap,
                size=Vec3(0.20, 0.08, 0.08),
                color=Vec4(0.96, 0.42 if side < 0 else 0.66, 0.18, 1.0),
                style="screen",
            )
            cap.setPos(0.85, side * 0.78, self.sim_config.height - 0.24)

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

    def _build_track_marker_wall(self, pos: Vec3, heading: float) -> None:
        wall = self.scene_root.attachNewNode("track-marker-wall")
        wall.setPos(pos)
        wall.setH(heading)
        self._attach_centered_box(wall, size=Vec3(3.4, 0.26, 1.9), color=Vec4(0.08, 0.10, 0.13, 1.0))
        for index, color in enumerate((Vec4(0.18, 0.68, 0.94, 1.0), Vec4(0.96, 0.42, 0.18, 1.0), Vec4(0.86, 0.88, 0.92, 1.0))):
            stripe = wall.attachNewNode(f"stripe-{index}")
            self._attach_centered_box(stripe, size=Vec3(2.8, 0.04, 0.14), color=color)
            stripe.setPos(0.0, -0.12, 0.46 - index * 0.34)

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
        for index in range(3):
            line = self.marble_root.attachNewNode(f"speed-line-{index}")
            self._attach_centered_box(
                line,
                size=Vec3(self.sim_config.marble_radius * 1.8, 0.02, 0.02),
                color=Vec4(0.18, 0.74, 0.94, 1.0),
            )
            line.setPos(-self.sim_config.marble_radius * (1.2 + 0.5 * index), 0.0, -0.05 + index * 0.05)
            self.speed_lines.append(line)

    def _setup_camera(self) -> None:
        self.camLens.setFov(48.0)
        self._update_camera(self.simulation.snapshot(), 1.0 / 60.0)

    def _build_ui(self) -> None:
        self.title = OnscreenText(
            text="NEON DESK MARBLE RUN",
            pos=(-1.30, 0.92),
            align=TextNode.ALeft,
            scale=0.060,
            fg=(1.00, 0.62, 0.24, 1.0),
            mayChange=False,
        )
        self.subtitle = OnscreenText(
            text="Arcade chase cam through a neon hazard run",
            pos=(-1.30, 0.85),
            align=TextNode.ALeft,
            scale=0.030,
            fg=(0.62, 0.76, 0.92, 1.0),
            mayChange=False,
        )
        self.hud = OnscreenText(
            text="",
            pos=(-1.30, 0.70),
            align=TextNode.ALeft,
            scale=0.042,
            fg=(0.90, 0.93, 0.97, 1.0),
            mayChange=True,
        )
        self.controls = OnscreenText(
            text="Left/Right steer   Down brake   Space pause   R reset   Esc quit",
            pos=(0.0, -0.94),
            scale=0.034,
            fg=(0.78, 0.84, 0.92, 1.0),
            mayChange=False,
        )

    def _setup_input(self) -> None:
        self.accept("arrow_left", self._set_steering, [1.0])
        self.accept("arrow_left-up", self._release_steering, [1.0])
        self.accept("arrow_right", self._set_steering, [-1.0])
        self.accept("arrow_right-up", self._release_steering, [-1.0])
        self.accept("arrow_down", self._set_brake, [1.0])
        self.accept("arrow_down-up", self._set_brake, [0.0])
        self.accept("space", self._toggle_pause)
        self.accept("r", self._reset)
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

    def _toggle_pause(self) -> None:
        self.paused = not self.paused

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
        self._apply_state(1.0 / 60.0)

    def _tick(self, task: Task) -> int:
        dt = min(globalClock.getDt(), 1.0 / 120.0)
        if not self.paused:
            self.simulation.step(dt)
        self._animate_props(task.time, dt)
        self._apply_state(dt)
        return Task.cont

    def _animate_props(self, time_s: float, dt: float) -> None:
        for index, spinner in enumerate(self.prop_spinners):
            spinner.setR(spinner.getR() + dt * (90.0 + index * 12.0))
        pulse = 0.70 + 0.30 * sin(time_s * 3.6)
        for index, line in enumerate(self.speed_lines):
            line.setScale(1.0 + pulse * (0.35 + index * 0.08), 1.0, 1.0)

    def _apply_state(self, dt: float | None = None) -> None:
        state = self.simulation.snapshot()
        self.marble_root.setPos(state.position)
        self.marble_root.setQuat(self.simulation.marble_np.getQuat())
        camera_dt = 1.0 / 60.0 if dt is None else min(max(dt, 0.0), 1.0 / 30.0)
        self._update_camera(state, camera_dt)
        progress = min(100.0, (state.path_distance / self.sim_config.length) * 100.0)
        self.hud.setText(
            (
                f"time   {state.time:4.2f} s\n"
                f"speed  {state.speed:5.2f} m/s\n"
                f"track  {progress:5.1f}%\n"
                f"hazrd  {len(self.sim_config.obstacles):2d}\n"
                f"hits   {state.total_contacts:2d}\n"
                f"steer  {self.steering_input:+3.0f}\n"
                f"brake  {self.brake_input:3.0f}\n"
                f"mu_s   {required_static_friction(self.sim_config):4.2f}"
            )
        )

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
