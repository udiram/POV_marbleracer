from marbleracer.config import default_track_path, load_track_config


def test_default_track_loads() -> None:
    track = load_track_config(default_track_path())
    assert track.name
    assert len(track.start_positions) >= 4
    assert len(track.checkpoints) >= 3
    assert track.marble_tuning.max_speed > 0
