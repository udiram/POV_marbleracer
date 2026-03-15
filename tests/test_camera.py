from panda3d.core import Vec3

from marbleracer.app import clamp_frame_dt, smooth_binary_state, smooth_vec3, smoothing_alpha


def test_smoothing_alpha_is_zero_without_time_or_response() -> None:
    assert smoothing_alpha(0.0, 1.0 / 60.0) == 0.0
    assert smoothing_alpha(8.0, 0.0) == 0.0


def test_clamp_frame_dt_caps_spikes_without_slowing_normal_frames() -> None:
    assert clamp_frame_dt(1.0 / 60.0) == 1.0 / 60.0
    assert clamp_frame_dt(1.0 / 10.0) == 1.0 / 30.0
    assert clamp_frame_dt(-1.0) == 0.0


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


def test_smooth_vec3_is_stable_with_zero_dt() -> None:
    current = Vec3(1.0, 2.0, 3.0)
    target = Vec3(4.0, 5.0, 6.0)

    assert smooth_vec3(current, target, 8.0, 0.0) == current


def test_smooth_vec3_moves_toward_target_without_overshoot() -> None:
    current = Vec3(0.0, 0.0, 0.0)
    target = Vec3(10.0, -4.0, 2.0)

    result = smooth_vec3(current, target, 6.0, 1.0 / 60.0)

    assert 0.0 < result.x < target.x
    assert target.y < result.y < 0.0
    assert 0.0 < result.z < target.z
