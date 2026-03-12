from __future__ import annotations

import math
from dataclasses import dataclass

from panda3d.bullet import BulletBoxShape, BulletRigidBodyNode
from panda3d.core import CardMaker, NodePath, Point3, TransparencyAttrib, Vec3, Vec4

from .config import Checkpoint, HazardConfig, PieceConfig, TrackConfig


@dataclass(slots=True)
class HazardRuntime:
    config: HazardConfig
    node: NodePath
    body: BulletRigidBodyNode

    def update(self, elapsed: float) -> None:
        offset = self.config.axis * math.sin(elapsed * self.config.speed) * self.config.amplitude
        pos = self.config.base_pos + offset
        self.node.setPos(pos)
        self.body.setTransform(self.node.getTransform())


class TrackBuilder:
    def __init__(self, base, world, root: NodePath, track_config: TrackConfig) -> None:
        self.base = base
        self.world = world
        self.root = root
        self.track_config = track_config
        self.hazards: list[HazardRuntime] = []
        self.checkpoint_markers: list[NodePath] = []

    def build(self) -> None:
        self._add_table()
        for piece in self.track_config.pieces:
            self._add_static_box(piece)
        for rail in self.track_config.rails:
            self._add_static_box(rail)
        for checkpoint in self.track_config.checkpoints:
            self._add_checkpoint_marker(checkpoint)
        for hazard in self.track_config.hazards:
            self._add_hazard(hazard)

    def _add_table(self) -> None:
        cfg = self.track_config
        self._add_static_box(
            PieceConfig(
                name="table",
                kind="box",
                size=cfg.table_size,
                pos=cfg.table_pos,
                hpr=Vec3(0, 0, 0),
                color=cfg.table_color,
            )
        )

    def _box_visual(self, size: Vec3, color: Vec4, parent: NodePath) -> NodePath:
        model = self.base.loader.loadModel("models/box")
        model.reparentTo(parent)
        model.setScale(size.x, size.y, size.z)
        model.setColorScale(color)
        return model

    def _add_static_box(self, piece: PieceConfig) -> None:
        node = BulletRigidBodyNode(piece.name)
        node.addShape(BulletBoxShape(piece.size * 0.5))
        node_np = self.root.attachNewNode(node)
        node_np.setPosHpr(piece.pos, piece.hpr)
        self._box_visual(piece.size, piece.color, node_np)
        self.world.attachRigidBody(node)

    def _add_checkpoint_marker(self, checkpoint: Checkpoint) -> None:
        ring = self.root.attachNewNode(f"checkpoint_{checkpoint.name}")
        ring.setPos(checkpoint.pos)
        card = CardMaker(f"checkpoint_card_{checkpoint.name}")
        card.setFrame(
            -checkpoint.radius,
            checkpoint.radius,
            -checkpoint.radius,
            checkpoint.radius,
        )
        marker = ring.attachNewNode(card.generate())
        marker.lookAt(Point3(0, 0, ring.getZ() + 1))
        marker.setP(90)
        marker.setColor(Vec4(0.2, 0.9, 0.95, 0.16))
        marker.setTransparency(TransparencyAttrib.M_alpha)
        marker.setTwoSided(True)
        self.checkpoint_markers.append(ring)

    def _add_hazard(self, hazard: HazardConfig) -> None:
        body = BulletRigidBodyNode(hazard.name)
        body.addShape(BulletBoxShape(hazard.size * 0.5))
        body.setKinematic(True)
        node = self.root.attachNewNode(body)
        node.setPos(hazard.base_pos)
        self._box_visual(hazard.size, hazard.color, node)
        self.world.attachRigidBody(body)
        self.hazards.append(HazardRuntime(config=hazard, node=node, body=body))
