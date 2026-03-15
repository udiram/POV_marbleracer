from marbleracer.app import smooth_binary_state, smoothing_alpha


def test_smoothing_alpha_is_zero_without_time_or_response() -> None:
    assert smoothing_alpha(0.0, 1.0 / 60.0) == 0.0
    assert smoothing_alpha(8.0, 0.0) == 0.0


def test_collision_blend_filters_single_frame_contact_dropouts() -> None:
    dt = 1.0 / 60.0
    blend = 0.0
    for _ in range(12):
        blend = smooth_binary_state(
            blend,
            True,
            dt,
            attack_response=12.0,
            release_response=4.0,
        )

    dropped = smooth_binary_state(
        blend,
        False,
        dt,
        attack_response=12.0,
        release_response=4.0,
    )
    assert blend > 0.85
    assert dropped > 0.80
    assert blend - dropped < 0.08


def test_collision_blend_eventually_releases_after_sustained_clearance() -> None:
    dt = 1.0 / 60.0
    blend = 1.0
    for _ in range(30):
        blend = smooth_binary_state(
            blend,
            False,
            dt,
            attack_response=12.0,
            release_response=4.0,
        )

    assert blend < 0.2
