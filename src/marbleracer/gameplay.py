from __future__ import annotations

from dataclasses import dataclass


SECTION_TEMPLATES: tuple[
    tuple[str, str, float, float, tuple[float, float, float, float]],
    ...,
] = (
    ("Launch", "Build speed cleanly", 0.22, 0.12, (0.24, 0.70, 0.92, 1.0)),
    ("Long Bend", "Carry the sweep", 0.50, 0.38, (0.24, 0.70, 0.92, 1.0)),
    ("Recovery Straight", "Set up the finish", 0.76, 0.62, (0.24, 0.70, 0.92, 1.0)),
    ("Final Run", "Stay smooth to the line", 0.97, 0.86, (0.24, 0.70, 0.92, 1.0)),
)

MEDAL_COLORS: dict[str, tuple[float, float, float, float]] = {
    "PLATINUM": (0.82, 0.96, 1.00, 1.0),
    "GOLD": (1.00, 0.78, 0.24, 1.0),
    "SILVER": (0.82, 0.86, 0.94, 1.0),
    "BRONZE": (0.90, 0.58, 0.32, 1.0),
}

SECTION_RATING_COLORS: dict[str, tuple[float, float, float, float]] = {
    "CLEAN": (0.20, 0.92, 0.78, 1.0),
    "STABLE": (0.20, 0.76, 0.96, 1.0),
    "SCRAPPY": (1.00, 0.78, 0.24, 1.0),
    "ROUGH": (0.96, 0.42, 0.18, 1.0),
}


@dataclass(frozen=True, slots=True)
class CourseSection:
    name: str
    focus: str
    end_distance: float
    boost_distance: float
    color: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class SectionResult:
    rating: str
    summary: str
    score: float
    color: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class RunGrade:
    medal: str
    adjusted_time: float
    target_time: float
    color: tuple[float, float, float, float]


def build_course_sections(length: float) -> tuple[CourseSection, ...]:
    if length <= 0.0:
        raise ValueError("Course length must be positive.")

    return tuple(
        CourseSection(
            name=name,
            focus=focus,
            end_distance=length * end_ratio,
            boost_distance=length * boost_ratio,
            color=color,
        )
        for name, focus, end_ratio, boost_ratio, color in SECTION_TEMPLATES
    )


def estimate_target_time(length: float, obstacle_count: int) -> float:
    if length <= 0.0:
        raise ValueError("Course length must be positive.")
    if obstacle_count < 0:
        raise ValueError("Obstacle count must be non-negative.")
    return max(18.0, length * 0.215 + obstacle_count * 0.16 + 4.6)


def rate_section(
    *,
    major_impacts: int,
    stability_loss: float,
    boost_hits: int,
) -> SectionResult:
    if major_impacts < 0 or stability_loss < 0.0 or boost_hits < 0:
        raise ValueError("Section metrics must be non-negative.")

    score = major_impacts * 1.2 + stability_loss * 0.72 - min(boost_hits, 2) * 0.22
    if score <= 0.30:
        return SectionResult("CLEAN", "Clean line, speed carried.", score, SECTION_RATING_COLORS["CLEAN"])
    if score <= 1.20:
        return SectionResult("STABLE", "Small scrub, good control.", score, SECTION_RATING_COLORS["STABLE"])
    if score <= 2.30:
        return SectionResult("SCRAPPY", "Speed lost through contact.", score, SECTION_RATING_COLORS["SCRAPPY"])
    return SectionResult("ROUGH", "Heavy contact, full recovery needed.", score, SECTION_RATING_COLORS["ROUGH"])


def grade_run(
    *,
    length: float,
    obstacle_count: int,
    finish_time: float,
    major_impacts: int,
    clean_sections: int,
    stability_loss: float,
    boost_chain: int,
) -> RunGrade:
    if finish_time < 0.0:
        raise ValueError("Finish time must be non-negative.")
    if major_impacts < 0 or clean_sections < 0 or boost_chain < 0:
        raise ValueError("Run metrics must be non-negative.")
    if stability_loss < 0.0:
        raise ValueError("Stability loss must be non-negative.")

    target_time = estimate_target_time(length, obstacle_count)
    adjusted_time = max(
        0.0,
        finish_time
        + major_impacts * 0.70
        + stability_loss * 0.26
        - clean_sections * 0.20
        - min(boost_chain, 4) * 0.08,
    )
    thresholds = (
        ("PLATINUM", target_time * 0.92),
        ("GOLD", target_time),
        ("SILVER", target_time * 1.11),
        ("BRONZE", target_time * 1.23),
    )
    for medal, limit in thresholds:
        if adjusted_time <= limit:
            return RunGrade(
                medal=medal,
                adjusted_time=adjusted_time,
                target_time=limit,
                color=MEDAL_COLORS[medal],
            )
    return RunGrade(
        medal="BRONZE",
        adjusted_time=adjusted_time,
        target_time=thresholds[-1][1],
        color=MEDAL_COLORS["BRONZE"],
    )


def format_seconds(seconds: float) -> str:
    if seconds < 0.0:
        raise ValueError("Seconds must be non-negative.")
    if seconds < 60.0:
        return f"{seconds:0.2f}s"
    minutes = int(seconds // 60.0)
    remainder = seconds - minutes * 60.0
    return f"{minutes}:{remainder:05.2f}"


def format_delta(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    sign = "+" if seconds >= 0.0 else "-"
    return f"{sign}{abs(seconds):0.2f}s"
