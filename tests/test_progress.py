from __future__ import annotations

from pathlib import Path

from panda3d.core import Vec3

from marbleracer.progress import (
    GhostSample,
    LevelProgress,
    ProgressData,
    load_progress,
    reset_progress,
    save_progress,
    update_level_progress,
)


def test_progress_round_trip_preserves_unlocks_medals_and_ghosts(tmp_path: Path) -> None:
    progress = ProgressData(
        levels={
            "default-showcase": LevelProgress(
                unlocked=True,
                best_time=24.8,
                best_medal="GOLD",
                ghost=(GhostSample.from_vec3(time=0.0, path_distance=0.0, position=Vec3(1.0, 2.0, 3.0)),),
            )
        }
    )

    path = save_progress(progress, tmp_path / "progress.json")
    restored = load_progress(path)

    assert restored.levels["default-showcase"].unlocked
    assert restored.levels["default-showcase"].best_time == 24.8
    assert restored.levels["default-showcase"].best_medal == "GOLD"
    assert restored.levels["default-showcase"].ghost[0].position == (1.0, 2.0, 3.0)


def test_progress_updates_keep_the_best_medal_and_fastest_time() -> None:
    progress = ProgressData()
    progress = update_level_progress(progress, "default-showcase", unlocked=True, best_time=25.0, best_medal="SILVER")
    progress = update_level_progress(progress, "default-showcase", best_time=26.0, best_medal="BRONZE")
    progress = update_level_progress(progress, "default-showcase", best_time=24.5, best_medal="GOLD")

    entry = progress.levels["default-showcase"]
    assert entry.unlocked
    assert entry.best_time == 24.5
    assert entry.best_medal == "GOLD"


def test_reset_progress_removes_the_save_file(tmp_path: Path) -> None:
    path = save_progress(ProgressData(), tmp_path / "progress.json")
    assert path.exists()

    reset_progress(path)

    assert not path.exists()
