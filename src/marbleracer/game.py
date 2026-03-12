from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.bullet import BulletSphereShape, BulletWorld, BulletRigidBodyNode
from panda3d.core import (
    AmbientLight,
    DirectionalLight,
    KeyboardButton,
    NodePath,
    Point3,
    Vec3,
    Vec4,
    WindowProperties,
    loadPrcFileData,
)

from .config import MarbleTuning, TrackConfig, default_track_path, load_track_config
from .track import TrackBuilder


loadPrcFileData("", "window-title POV Marble Racer")
loadPrcFileData("", "sync-video true")


class RacePhase(str, Enum):
    MENU = "menu"
    COUNTDOWN = "countdown"
    RACING = "racing"
    PAUSED = "paused"
    FINISHED = "finished"


class CameraMode(str, Enum):
    POV = "POV"
    CHASE = "CHASE"


@dataclass(slots=True)
class RacerState:
    name: str
    body_np: NodePath
    body: BulletRigidBodyNode
    checkpoint_index: int = 0
    last_checkpoint_index: int = 0
    finished: bool = False
    finish_time: float = 0.0
    stuck_timer: float = 0.0

    @property
    def pos(self) -> Point3:
        return self.body_np.getPos()

    @property
    def speed(self) -> float:
        return self.body.getLinearVelocity().length()


class MarbleRacerGame(ShowBase):
    def __init__(self, track_path: str | Path | None = None) -> None:
        super().__init__()
        self.disableMouse()
        self.track_config = load_track_config(track_path or default_track_path())
        self.phase = RacePhase.MENU
        self.camera_mode = CameraMode.POV
        self.elapsed_race_time = 0.0
        self.countdown_remaining = 3.0
        self.pause_text = ""
        self.keys: set[str] = set()

        self._setup_window()
        self._setup_scene()
        self._setup_input()
        self._build_world()
        self._setup_ui()
        self.taskMgr.add(self._tick, "game_tick")

    def _setup_window(self) -> None:
        props = WindowProperties()
        props.setSize(1400, 900)
        props.setTitle("POV Marble Racer")
        if self.win and hasattr(self.win, "requestProperties"):
            self.win.requestProperties(props)
        self.setBackgroundColor(0.08, 0.11, 0.16, 1.0)

    def _setup_scene(self) -> None:
        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.5, 0.52, 0.58, 1.0))
        ambient_np = render.attachNewNode(ambient)
        render.setLight(ambient_np)

        sunlight = DirectionalLight("sunlight")
        sunlight.setColor(Vec4(0.92, 0.92, 0.85, 1.0))
        sun_np = render.attachNewNode(sunlight)
        sun_np.setHpr(-25, -45, 0)
        render.setLight(sun_np)

    def _setup_input(self) -> None:
        for key in ["w", "a", "s", "d", "space", "c", "r", "escape"]:
            self.accept(key, self._on_key, [key, True])
            self.accept(f"{key}-up", self._on_key, [key, False])
        self.accept("shift-r", self._return_to_menu)

    def _on_key(self, key: str, is_down: bool) -> None:
        if is_down:
            self.keys.add(key)
            if key == "space" and self.phase == RacePhase.MENU:
                self.start_countdown()
            elif key == "c":
                self.camera_mode = (
                    CameraMode.CHASE if self.camera_mode == CameraMode.POV else CameraMode.POV
                )
            elif key == "r" and self.phase in {RacePhase.RACING, RacePhase.COUNTDOWN, RacePhase.PAUSED}:
                self.reset_racer(self.player, self.player.last_checkpoint_index)
            elif key == "escape":
                self._handle_escape()
        else:
            self.keys.discard(key)

    def _handle_escape(self) -> None:
        if self.phase == RacePhase.RACING:
            self.phase = RacePhase.PAUSED
        elif self.phase == RacePhase.PAUSED:
            self.phase = RacePhase.RACING
        elif self.phase == RacePhase.FINISHED:
            self._return_to_menu()

    def _build_world(self) -> None:
        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -31))
        self.scene_root = render.attachNewNode("scene_root")
        self.track_builder = TrackBuilder(self, self.world, self.scene_root, self.track_config)
        self.track_builder.build()
        self.player = self._spawn_racer("Player", Vec4(0.95, 0.97, 1.0, 1.0), self.track_config.start_positions[0])
        self.ai_racers = [
            self._spawn_racer(f"AI {index + 1}", color, self.track_config.start_positions[index + 1])
            for index, color in enumerate(
                [
                    Vec4(0.95, 0.45, 0.38, 1.0),
                    Vec4(0.32, 0.75, 0.48, 1.0),
                    Vec4(0.35, 0.66, 0.95, 1.0),
                ]
            )
        ]
        self.all_racers = [self.player, *self.ai_racers]

    def _spawn_racer(self, name: str, color: Vec4, spawn_pos: Vec3) -> RacerState:
        tuning = self.track_config.marble_tuning
        body = BulletRigidBodyNode(name)
        body.setMass(tuning.mass)
        body.addShape(BulletSphereShape(tuning.radius))
        body.setFriction(1.15)
        body.setRestitution(0.08)
        body.setLinearDamping(tuning.linear_damping)
        body.setAngularDamping(tuning.angular_damping)
        body_np = self.scene_root.attachNewNode(body)
        body_np.setPos(spawn_pos)
        visual = self.loader.loadModel("models/smiley")
        visual.reparentTo(body_np)
        scale = tuning.radius / 0.48
        visual.setScale(scale)
        visual.setColorScale(color)
        self.world.attachRigidBody(body)
        return RacerState(name=name, body_np=body_np, body=body)

    def _setup_ui(self) -> None:
        self.title_text = OnscreenText(
            text="POV Marble Racer\nPress Space to Launch",
            pos=(0, 0.25),
            scale=0.08,
            fg=(0.95, 0.98, 1.0, 1.0),
            mayChange=True,
        )
        self.subtitle_text = OnscreenText(
            text="WASD steer  |  C toggle camera  |  R reset  |  Esc pause",
            pos=(0, 0.08),
            scale=0.045,
            fg=(0.72, 0.8, 0.94, 1.0),
            mayChange=True,
        )
        self.hud_text = OnscreenText(
            text="",
            pos=(-1.28, 0.92),
            align=0,
            scale=0.05,
            fg=(0.95, 0.98, 1.0, 1.0),
            mayChange=True,
        )
        self.center_text = OnscreenText(
            text="",
            pos=(0, 0),
            scale=0.12,
            fg=(1.0, 0.94, 0.6, 1.0),
            mayChange=True,
        )

    def start_countdown(self) -> None:
        self._reset_race()
        self.phase = RacePhase.COUNTDOWN
        self.countdown_remaining = 3.0
        self.title_text.setText("")
        self.subtitle_text.setText("")

    def _reset_race(self) -> None:
        self.elapsed_race_time = 0.0
        for index, racer in enumerate(self.all_racers):
            racer.checkpoint_index = 0
            racer.last_checkpoint_index = 0
            racer.finished = False
            racer.finish_time = 0.0
            racer.stuck_timer = 0.0
            self.reset_racer(racer, 0, spawn_index=index)

    def reset_racer(self, racer: RacerState, checkpoint_index: int, spawn_index: int = 0) -> None:
        if checkpoint_index <= 0:
            pos = self.track_config.start_positions[min(spawn_index, len(self.track_config.start_positions) - 1)]
        else:
            checkpoint = self.track_config.checkpoints[checkpoint_index - 1]
            pos = checkpoint.pos + Vec3(0, -1.5, 1.6)
        racer.body_np.setPos(pos)
        racer.body_np.setHpr(0, 0, 0)
        racer.body.setLinearVelocity(Vec3(0, 0, 0))
        racer.body.setAngularVelocity(Vec3(0, 0, 0))
        racer.last_checkpoint_index = checkpoint_index
        racer.checkpoint_index = checkpoint_index

    def _tick(self, task: Task) -> int:
        dt = min(globalClock.getDt(), 1 / 30)
        self._update_hazards(task.time)

        if self.phase == RacePhase.COUNTDOWN:
            self.countdown_remaining -= dt
            countdown = max(0, int(self.countdown_remaining) + 1)
            self.center_text.setText("GO!" if self.countdown_remaining <= 0 else str(countdown))
            if self.countdown_remaining <= 0:
                self.phase = RacePhase.RACING
                self.center_text.setText("GO!")
        elif self.phase == RacePhase.RACING:
            self.elapsed_race_time += dt
            self.center_text.setText("")
            self._update_player_controls(dt)
            for index, racer in enumerate(self.ai_racers):
                self._update_ai(racer, dt, index)
        elif self.phase == RacePhase.PAUSED:
            self.center_text.setText("PAUSED")
            self._update_camera(dt)
            self._update_hud()
            return Task.cont
        elif self.phase == RacePhase.MENU:
            self._update_camera(dt, attract_mode=True)
            self._update_hud()
            return Task.cont
        elif self.phase == RacePhase.FINISHED:
            self._update_camera(dt)

        self.world.doPhysics(dt, 8, 1 / 240.0)
        self._enforce_speed_caps()
        self._update_race_progress(dt)
        self._update_camera(dt)
        self._update_hud()
        return Task.cont

    def _update_hazards(self, elapsed: float) -> None:
        for hazard in self.track_builder.hazards:
            hazard.update(elapsed)

    def _movement_basis(self) -> tuple[Vec3, Vec3]:
        if self.camera_mode == CameraMode.CHASE:
            forward = self.camera.getQuat(render).getForward()
            forward.setZ(0)
            if forward.length_squared() < 0.001:
                forward = Vec3(0, 1, 0)
            forward.normalize()
            right = forward.cross(Vec3(0, 0, 1))
            right.normalize()
            return forward, right
        return Vec3(0, 1, 0), Vec3(1, 0, 0)

    def _update_player_controls(self, dt: float) -> None:
        x = float("d" in self.keys) - float("a" in self.keys)
        y = float("w" in self.keys) - float("s" in self.keys)
        direction = Vec3(x, y, 0)
        self._drive_racer(self.player, direction, dt, self.track_config.marble_tuning)

    def _drive_racer(self, racer: RacerState, input_vec: Vec3, dt: float, tuning: MarbleTuning) -> None:
        if input_vec.length_squared() <= 0:
            return
        forward, right = self._movement_basis() if racer is self.player else (Vec3(0, 1, 0), Vec3(1, 0, 0))
        move = forward * input_vec.y + right * input_vec.x
        move.setZ(0)
        if move.length_squared() < 0.01:
            return
        move.normalize()
        air_scale = tuning.air_control if abs(racer.body.getLinearVelocity().z) > 1.8 else 1.0
        torque_axis = Vec3(0, 0, 1).cross(move)
        torque = torque_axis * tuning.steering_torque * air_scale
        if racer.speed < tuning.max_speed or racer.body.getLinearVelocity().dot(move) < 0:
            racer.body.applyTorque(torque)
        velocity = racer.body.getLinearVelocity()
        throttle_window = max(4.0, tuning.max_speed - racer.speed)
        drive_force = move * tuning.grip_assist * throttle_window * tuning.mass * 8.0
        lateral = Vec3(velocity.x, velocity.y, 0) - move * Vec3(velocity.x, velocity.y, 0).dot(move)
        stability_force = -lateral * tuning.mass * 12.0
        drive_force.setZ(0)
        racer.body.applyCentralForce(drive_force + stability_force)
        horizontal = Vec3(velocity.x, velocity.y, 0)
        if horizontal.length() > tuning.max_speed:
            horizontal.normalize()
            velocity.x = horizontal.x * tuning.max_speed
            velocity.y = horizontal.y * tuning.max_speed
            racer.body.setLinearVelocity(velocity)

    def _update_ai(self, racer: RacerState, dt: float, index: int) -> None:
        target_idx = min(racer.checkpoint_index, len(self.track_config.checkpoints) - 1)
        target = self.track_config.checkpoints[target_idx].pos
        delta = target - racer.pos
        input_vec = Vec3(delta.x * 0.45, delta.y, 0)
        if input_vec.length_squared() > 0.01:
            input_vec.normalize()
        tuning = self.track_config.marble_tuning
        profile_scale = 0.87 + 0.07 * index
        ai_tuning = MarbleTuning(
            radius=tuning.radius,
            mass=tuning.mass,
            steering_torque=tuning.steering_torque * profile_scale,
            max_speed=tuning.max_speed * (0.93 + 0.03 * index),
            linear_damping=tuning.linear_damping,
            angular_damping=tuning.angular_damping,
            grip_assist=tuning.grip_assist * profile_scale,
            air_control=tuning.air_control,
            recovery_height=tuning.recovery_height,
        )
        self._drive_racer(racer, input_vec, dt, ai_tuning)

    def _update_race_progress(self, dt: float) -> None:
        for index, racer in enumerate(self.all_racers):
            if racer.finished:
                continue
            self._advance_checkpoint(racer)
            if racer.pos.z < self.track_config.marble_tuning.recovery_height:
                self.reset_racer(racer, racer.last_checkpoint_index, spawn_index=index)
                continue
            if racer.speed < 0.85:
                racer.stuck_timer += dt
                if racer.stuck_timer > 2.5:
                    self.reset_racer(racer, racer.last_checkpoint_index, spawn_index=index)
            else:
                racer.stuck_timer = 0.0
        if self.player.finished and self.phase != RacePhase.FINISHED:
            self.phase = RacePhase.FINISHED
            self.center_text.setText(f"Finished! {self.player.finish_time:0.2f}s")
            self.subtitle_text.setText("Press Esc for title or R to retry from last checkpoint")

    def _advance_checkpoint(self, racer: RacerState) -> None:
        if racer.checkpoint_index >= len(self.track_config.checkpoints):
            return
        checkpoint = self.track_config.checkpoints[racer.checkpoint_index]
        delta = racer.pos - checkpoint.pos
        horizontal_delta = Vec3(delta.x, delta.y, 0)
        if horizontal_delta.length() <= checkpoint.radius and abs(delta.z) <= checkpoint.radius * 0.85:
            racer.checkpoint_index += 1
            racer.last_checkpoint_index = racer.checkpoint_index
            if racer.checkpoint_index >= len(self.track_config.checkpoints):
                racer.finished = True
                racer.finish_time = self.elapsed_race_time

    def _progress_score(self, racer: RacerState) -> float:
        checkpoints = self.track_config.checkpoints
        if racer.finished:
            return float(len(checkpoints) + 1) - racer.finish_time * 0.001
        index = min(racer.checkpoint_index, len(checkpoints) - 1)
        target = checkpoints[index].pos
        distance = (target - racer.pos).length()
        fraction = max(0.0, 1.0 - distance / 14.0)
        return racer.checkpoint_index + fraction

    def _enforce_speed_caps(self) -> None:
        max_speed = self.track_config.marble_tuning.max_speed
        for racer in self.all_racers:
            velocity = racer.body.getLinearVelocity()
            horizontal = Vec3(velocity.x, velocity.y, 0)
            if horizontal.length() > max_speed:
                horizontal.normalize()
                velocity.x = horizontal.x * max_speed
                velocity.y = horizontal.y * max_speed
                racer.body.setLinearVelocity(velocity)

    def _ranking(self) -> list[RacerState]:
        return sorted(self.all_racers, key=self._progress_score, reverse=True)

    def _update_camera(self, dt: float, attract_mode: bool = False) -> None:
        if attract_mode:
            focus = self.player.pos + Vec3(0, 20, 6)
            cam_pos = Point3(12, -12, 11)
            self.camera.setPos(self.camera.getPos() + (cam_pos - self.camera.getPos()) * min(1.0, dt * 2.0))
            self.camera.lookAt(focus)
            return

        marble_pos = self.player.pos
        velocity = self.player.body.getLinearVelocity()
        forward = velocity if velocity.length_squared() > 0.4 else Vec3(0, 1, 0)
        forward.normalize()
        if self.camera_mode == CameraMode.POV:
            eye = marble_pos + Vec3(0, 0, self.track_config.marble_tuning.radius * 0.55)
            look_target = eye + forward * 14 + Vec3(0, 0, 0.6)
            self.camera.setPos(self.camera.getPos() + (eye - self.camera.getPos()) * min(1.0, dt * 12.0))
            self.camera.lookAt(look_target)
        else:
            chase_target = marble_pos - forward * 8 + Vec3(0, 0, 3.2)
            self.camera.setPos(self.camera.getPos() + (chase_target - self.camera.getPos()) * min(1.0, dt * 6.0))
            self.camera.lookAt(marble_pos + Vec3(0, 0, 1.3))

    def _update_hud(self) -> None:
        ranking = self._ranking()
        player_place = ranking.index(self.player) + 1
        checkpoint_total = len(self.track_config.checkpoints)
        checkpoint_value = min(self.player.checkpoint_index, checkpoint_total)
        if self.phase == RacePhase.MENU:
            self.hud_text.setText(f"Track: {self.track_config.name}\nMode: Tabletop Sprint")
            return
        hud = (
            f"Time  {self.elapsed_race_time:05.2f}\n"
            f"Place {player_place}/{len(self.all_racers)}\n"
            f"Speed {self.player.speed:04.1f}\n"
            f"Gate  {checkpoint_value}/{checkpoint_total}\n"
            f"Cam   {self.camera_mode.value}"
        )
        self.hud_text.setText(hud)
        if self.phase == RacePhase.COUNTDOWN:
            self.subtitle_text.setText("Hold your line. The launch starts fast.")
        elif self.phase == RacePhase.RACING:
            self.subtitle_text.setText("Reach each glowing gate. Reset with R if you drop.")
        elif self.phase == RacePhase.PAUSED:
            self.subtitle_text.setText("Esc resume  |  C toggle camera  |  Shift+R title")

    def _return_to_menu(self) -> None:
        self.phase = RacePhase.MENU
        self.center_text.setText("")
        self.title_text.setText("POV Marble Racer\nPress Space to Launch")
        self.subtitle_text.setText("WASD steer  |  C toggle camera  |  R reset  |  Esc pause")
        self._reset_race()


def run_game(track_path: str | Path | None = None) -> None:
    game = MarbleRacerGame(track_path=track_path)
    game.run()
