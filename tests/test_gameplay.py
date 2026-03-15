from marbleracer.gameplay import (
    build_course_sections,
    format_delta,
    format_seconds,
    grade_run,
    rate_section,
)


def test_sections_are_sorted_varied_and_within_course_length() -> None:
    sections = build_course_sections(90.0)
    assert len(sections) == 4
    assert all(section.focus for section in sections)
    assert all(0.0 < section.boost_distance < section.end_distance < 90.0 for section in sections)
    assert [section.end_distance for section in sections] == sorted(section.end_distance for section in sections)


def test_grade_run_rewards_cleaner_faster_more_stable_runs() -> None:
    rough_run = grade_run(
        length=90.0,
        obstacle_count=8,
        finish_time=31.0,
        major_impacts=4,
        clean_sections=0,
        stability_loss=3.8,
        boost_chain=1,
    )
    clean_run = grade_run(
        length=90.0,
        obstacle_count=8,
        finish_time=27.4,
        major_impacts=1,
        clean_sections=3,
        stability_loss=1.0,
        boost_chain=3,
    )

    assert rough_run.adjusted_time > clean_run.adjusted_time
    assert rough_run.medal in {"SILVER", "BRONZE"}
    assert clean_run.medal in {"PLATINUM", "GOLD", "SILVER"}


def test_rate_section_distinguishes_clean_and_rough_runs() -> None:
    clean = rate_section(major_impacts=0, stability_loss=0.1, boost_hits=1)
    rough = rate_section(major_impacts=2, stability_loss=2.6, boost_hits=0)

    assert clean.rating == "CLEAN"
    assert rough.rating in {"SCRAPPY", "ROUGH"}
    assert clean.score < rough.score


def test_time_formatters_handle_short_long_and_delta_times() -> None:
    assert format_seconds(9.876) == "9.88s"
    assert format_seconds(71.2) == "1:11.20"
    assert format_delta(0.42) == "+0.42s"
    assert format_delta(-0.42) == "-0.42s"
