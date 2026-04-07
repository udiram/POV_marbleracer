from __future__ import annotations

import csv
import json
import math
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from marbleracer.bot_controller import ExportedBotPolicy


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "artifacts/overnight_default_2026-04-05/report"
RUN_DIR = ROOT / "artifacts/overnight_default_2026-04-05/live_latest"
FIGURES_DIR = ROOT / "figures"
OUTPUT_DIR = ROOT / "output/pdf"
OUTPUT_PATH = OUTPUT_DIR / "rl_bot_supplementary.pdf"


@dataclass(frozen=True)
class CheckpointRow:
    phase: str
    generation: int
    timesteps: int
    checkpoint_path: str
    finished: bool
    finish_time: float | None
    max_progress: float
    reward: float
    steps: int
    resets: int
    termination_reason: str


def _load_rows(path: Path) -> list[CheckpointRow]:
    rows: list[CheckpointRow] = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                CheckpointRow(
                    phase=str(row["phase"]),
                    generation=int(row["generation"]),
                    timesteps=int(row["timesteps"]),
                    checkpoint_path=str(row["checkpoint_path"]),
                    finished=str(row["finished"]).lower() == "true",
                    finish_time=None if row["finish_time"] == "" else float(row["finish_time"]),
                    max_progress=float(row["max_progress"]),
                    reward=float(row["reward"]),
                    steps=int(row["steps"]),
                    resets=int(row["resets"]),
                    termination_reason=str(row["termination_reason"]),
                )
            )
    return rows


def _font(size: int, *, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _copy_replay_frame() -> Path:
    src = RUN_DIR / "replay_3d_latest/training_ghosts_latest_active.png"
    dst = FIGURES_DIR / "training_monitor_latest_frame.png"
    if src.exists():
        shutil.copyfile(src, dst)
    return dst


def _make_phase_figure() -> Path:
    out = FIGURES_DIR / "rl_bot_phase_overview.png"
    img = PILImage.new("RGB", (1600, 900), "#f5f0e7")
    draw = ImageDraw.Draw(img)
    title_font = _font(54, bold=True)
    section_font = _font(30, bold=True)
    body_font = _font(24)
    small_font = _font(22)

    draw.rounded_rectangle((60, 70, 1540, 830), radius=36, fill="#fffaf3", outline="#d9c9b7", width=4)
    draw.text((100, 105), "RL Bot Training Pipeline", fill="#1f2933", font=title_font)
    draw.text((100, 170), "Two-stage PPO curriculum used for the marble bot project", fill="#55606d", font=body_font)

    boxes = [
        (
            (110, 260, 470, 710),
            "#dceefb",
            "Stage 1",
            [
                "Track: time-trial-short",
                "Obstacles disabled",
                "Goal: basic steering, braking, and finish completion",
                "Training interface: 23-D observation -> 2-D action",
            ],
        ),
        (
            (520, 260, 1080, 710),
            "#e6f6e8",
            "Stage 2",
            [
                "Track: default full course",
                "Obstacles and boosts enabled",
                "Spawn curriculum at 0, 18, 36, 54, 68, 82, 96, 110 m",
                "Checkpointed every 100k steps with replay snapshots",
            ],
        ),
        (
            (1130, 260, 1490, 710),
            "#fce8da",
            "Outputs",
            [
                "SB3 checkpoint archives",
                "checkpoint_metrics.csv / json",
                "live monitor plots and replay mp4",
                "exported policy manifest + weights",
            ],
        ),
    ]

    for x0, y0, x1, y1 in [item[0] for item in boxes]:
        draw.rounded_rectangle((x0, y0, x1, y1), radius=28, fill="#ffffff", outline="#c9b8a8", width=3)
    for rect, accent, header, bullets in boxes:
        x0, y0, x1, y1 = rect
        draw.rounded_rectangle((x0, y0, x1, y0 + 70), radius=28, fill=accent, outline=accent)
        draw.rectangle((x0, y0 + 40, x1, y0 + 70), fill=accent)
        draw.text((x0 + 24, y0 + 16), header, fill="#1f2933", font=section_font)
        y = y0 + 110
        for bullet in bullets:
            draw.ellipse((x0 + 24, y + 10, x0 + 36, y + 22), fill="#9b6b3d")
            draw.text((x0 + 50, y), bullet, fill="#39424e", font=body_font)
            y += 88

    draw.line((470, 485, 520, 485), fill="#9b6b3d", width=8)
    draw.polygon([(520, 485), (495, 470), (495, 500)], fill="#9b6b3d")
    draw.line((1080, 485, 1130, 485), fill="#9b6b3d", width=8)
    draw.polygon([(1130, 485), (1105, 470), (1105, 500)], fill="#9b6b3d")

    draw.text((100, 760), "Runtime note: the current exported policy head is 2-D, while the residual runtime controller expects 3 outputs.", fill="#7a2e1f", font=small_font)
    img.save(out)
    return out


def _make_checkpoint_summary(rows: list[CheckpointRow]) -> Path:
    out = FIGURES_DIR / "rl_bot_checkpoint_summary.png"
    width = 1700
    height = 960
    img = PILImage.new("RGB", (width, height), "#fcfaf6")
    draw = ImageDraw.Draw(img)
    title_font = _font(50, bold=True)
    label_font = _font(26, bold=True)
    body_font = _font(22)
    small_font = _font(18)

    draw.text((80, 70), "Checkpoint Outcome Summary", fill="#1f2933", font=title_font)
    draw.text((80, 135), "Late-stage overnight artifact slice (generations 17-27)", fill="#5b6673", font=body_font)

    chart_left = 120
    chart_top = 250
    chart_right = 1540
    chart_bottom = 780
    draw.rectangle((chart_left, chart_top, chart_right, chart_bottom), outline="#d7ccbe", width=3)

    max_progress = max(row.max_progress for row in rows)
    statuses = {"finish": "#2f855a", "off_track": "#c05621", "stall": "#b7791f", "timeout": "#805ad5"}

    for i, row in enumerate(rows):
        y = chart_top + 28 + i * 46
        draw.text((30, y - 6), f"G{row.generation}", fill="#1f2933", font=small_font)
        bar_end = chart_left + int((row.max_progress / max_progress) * (chart_right - chart_left - 20))
        color = statuses.get(row.termination_reason, "#4a5568")
        draw.rounded_rectangle((chart_left, y, bar_end, y + 26), radius=12, fill=color)
        draw.text((bar_end + 14, y - 2), f"{row.max_progress:.2f} m | {row.termination_reason}", fill="#39424e", font=small_font)

    for tick in range(0, math.ceil(max_progress / 20.0) + 1):
        x = chart_left + int((tick * 20.0 / max_progress) * (chart_right - chart_left - 20))
        draw.line((x, chart_top, x, chart_bottom), fill="#eee5da", width=1)
        draw.text((x - 10, chart_bottom + 18), f"{tick * 20}", fill="#5b6673", font=small_font)

    draw.text((120, 825), "Reward / finish notes", fill="#1f2933", font=label_font)
    y = 865
    highlights = [
        f"Best reward: generation {max(rows, key=lambda row: row.reward).generation} ({max(rows, key=lambda row: row.reward).reward:.2f})",
        f"Best raw progress: generation {max(rows, key=lambda row: row.max_progress).generation} ({max(rows, key=lambda row: row.max_progress).max_progress:.2f} m)",
        f"Fastest successful finish: generation {min((row for row in rows if row.finish_time is not None), key=lambda row: row.finish_time).generation} ({min((row for row in rows if row.finish_time is not None), key=lambda row: row.finish_time).finish_time:.2f} s)",
    ]
    for item in highlights:
        draw.ellipse((120, y + 8, 132, y + 20), fill="#9b6b3d")
        draw.text((145, y), item, fill="#39424e", font=body_font)
        y += 34

    img.save(out)
    return out


def _make_runtime_note_figure(policy: ExportedBotPolicy) -> Path:
    out = FIGURES_DIR / "rl_bot_runtime_compatibility.png"
    img = PILImage.new("RGB", (1400, 700), "#f8f6f2")
    draw = ImageDraw.Draw(img)
    title_font = _font(46, bold=True)
    label_font = _font(28, bold=True)
    body_font = _font(24)

    draw.text((70, 70), "Current Export / Runtime Compatibility", fill="#1f2933", font=title_font)
    draw.rounded_rectangle((90, 180, 620, 560), radius=28, fill="#dceefb", outline="#9ec5e5", width=3)
    draw.rounded_rectangle((780, 180, 1310, 560), radius=28, fill="#fce8da", outline="#ebb999", width=3)
    draw.text((130, 220), "Exported PPO policy", fill="#1f2933", font=label_font)
    draw.text((130, 290), f"Observation size: {policy.observation_size}", fill="#39424e", font=body_font)
    draw.text((130, 340), f"Action size: {policy.action_size}", fill="#39424e", font=body_font)
    draw.text((130, 390), "Meaning in training env:", fill="#39424e", font=body_font)
    draw.text((160, 435), "steer, brake", fill="#39424e", font=body_font)

    draw.text((820, 220), "Runtime bot wrapper", fill="#1f2933", font=label_font)
    draw.text((820, 290), "Expected residual action size: 3", fill="#39424e", font=body_font)
    draw.text((820, 340), "Meaning at runtime:", fill="#39424e", font=body_font)
    draw.text((850, 385), "lane delta, speed delta, brake", fill="#39424e", font=body_font)
    draw.text((820, 470), "Result in current code:", fill="#7a2e1f", font=body_font)
    draw.text((850, 515), "learned policy is marked incompatible and runtime falls back to heuristic", fill="#7a2e1f", font=body_font)

    draw.line((620, 370, 780, 370), fill="#7a2e1f", width=8)
    draw.polygon([(780, 370), (750, 350), (750, 390)], fill="#7a2e1f")
    img.save(out)
    return out


def _image(path: Path, width: float) -> Image:
    img = Image(str(path))
    img.drawWidth = width
    img.drawHeight = width * img.imageHeight / img.imageWidth
    return img


def _add_page_number(canvas, doc):
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#55606d"))
    canvas.drawRightString(doc.pagesize[0] - 54, 24, f"Page {doc.page}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    report_summary = json.loads((REPORT_DIR / "summary.json").read_text())
    rows = _load_rows(REPORT_DIR / "checkpoint_metrics.csv")
    policy = ExportedBotPolicy.load()
    run_script = (ROOT / "artifacts/overnight_default_2026-04-05/run_overnight.sh").read_text()
    overnight_requests_long_run = "--phase2-steps 2000000" in run_script and "--phase1-steps 80000" in run_script
    replay_frame = _copy_replay_frame()
    phase_figure = _make_phase_figure()
    checkpoint_figure = _make_checkpoint_summary(rows)
    compat_figure = _make_runtime_note_figure(policy)

    latest = rows[-1]
    best_progress = max(rows, key=lambda row: row.max_progress)
    best_reward = max(rows, key=lambda row: row.reward)
    fastest_finish = min((row for row in rows if row.finish_time is not None), key=lambda row: row.finish_time)
    finished_rows = [row for row in rows if row.finished]
    checkpoint_termination_counts = dict(Counter(row.termination_reason for row in rows))
    mean_finish_time = mean(row.finish_time for row in finished_rows if row.finish_time is not None)
    mean_progress = mean(row.max_progress for row in rows)
    mean_reward = mean(row.reward for row in rows)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SmallBody", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13, textColor=colors.HexColor("#24313d")))
    styles.add(ParagraphStyle(name="SectionTitle", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=15, leading=18, textColor=colors.HexColor("#13202b"), spaceAfter=6))
    styles["Title"].textColor = colors.HexColor("#13202b")

    story = []
    story.append(Paragraph("RL Bot Supplementary Walkthrough", styles["Title"]))
    story.append(Spacer(1, 0.12 * inch))
    story.append(
        Paragraph(
            "This supplement covers only the reinforcement-learning bot work in the marble racer project: the training environment, PPO setup, artifact pipeline, checkpointed results, and the current deployment status of the exported policy.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    summary_rows = [
        ["Artifact slice", "Overnight run dated 2026-04-05, checkpoint generations 17 through 27"],
        ["Observed total timesteps", f"{latest.timesteps:,} by the latest recorded checkpoint"],
        ["Recorded checkpoint outcomes", f"{len(rows)} checkpoints, {len(finished_rows)} finishes, {sum(row.termination_reason == 'off_track' for row in rows)} off-track, {sum(row.termination_reason == 'stall' for row in rows)} stall, {sum(row.termination_reason == 'timeout' for row in rows)} timeout"],
        ["Best successful checkpoint", f"Generation {best_reward.generation} with reward {best_reward.reward:.2f} and finish time {best_reward.finish_time:.2f} s"],
        ["Fastest successful finish", f"Generation {fastest_finish.generation} in {fastest_finish.finish_time:.2f} s"],
        ["Runtime export status", f"Exported policy shape is 23 -> 64 -> 64 -> {policy.action_size}; runtime compatibility flag = {policy.is_runtime_compatible}"],
    ]
    summary_table = Table(summary_rows, colWidths=[1.8 * inch, 5.25 * inch])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7f2eb")),
                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#d3c4b3")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e3d8cb")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEADING", (0, 0), (-1, -1), 12),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 0.16 * inch))
    story.append(_image(phase_figure, 7.0 * inch))
    story.append(Spacer(1, 0.1 * inch))
    story.append(
        Paragraph(
            "The RL work used a direct-control training environment. The observation vector contains 23 normalized features grouped around progress and velocity, lane error and contact state, boost and obstacle context, and near-term track geometry. The PPO policy therefore learned to steer and brake directly in simulation rather than only nudging a scripted planner.",
            styles["BodyText"],
        )
    )

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Method", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "The code implements a two-stage curriculum in <b>scripts/train_bot_policy.py</b>. Stage 1 trains on the short time-trial course without obstacles to stabilize basic motion. Stage 2 moves to the full default track with boosts and moving hazards, and resets spawn positions along a curriculum of distances from 0 m through 110 m so the policy repeatedly rehearses late-course sections instead of always relearning the opening ramp.",
            styles["BodyText"],
        )
    )
    story.append(
        Paragraph(
            "The PPO configuration is consistent and compact: an MLP policy with hidden layers [64, 64], n_steps = 1024, batch_size = 256, gamma = 0.995, gae_lambda = 0.95, entropy coefficient = 0.01, learning rate = 3e-4, and clip range = 0.2. Checkpoints are saved every 100k timesteps, after which the code runs an evaluation rollout and refreshes the live monitor snapshot assets.",
            styles["BodyText"],
        )
    )
    if overnight_requests_long_run:
        story.append(
            Paragraph(
                "The overnight launcher script requested a longer run than the metadata preserved in <b>report/summary.json</b>: it asks for 80k phase-1 steps and 2.0M phase-2 steps. Because the report summary still lists 40k and 400k, the most reliable evidence of actual progress is the checkpoint record itself, which reaches generation 27 at 2,924,544 total timesteps.",
                styles["SmallBody"],
            )
        )

    story.append(PageBreak())
    story.append(Paragraph("Observed Outcomes", styles["SectionTitle"]))
    story.append(
        Paragraph(
            f"The stored checkpoint slice is clearly late-stage training rather than an entire run from scratch. It begins at generation {rows[0].generation} and ends at generation {latest.generation}. Within this window, {len(finished_rows)} of {len(rows)} checkpoint evaluations finished the course. The best raw progress came from generation {best_progress.generation} at {best_progress.max_progress:.2f} m, but that rollout still terminated off-track. The strongest successful checkpoint by reward is generation {best_reward.generation}, which finished in {best_reward.finish_time:.2f} s with reward {best_reward.reward:.2f}.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 0.08 * inch))
    story.append(_image(checkpoint_figure, 7.0 * inch))
    story.append(Spacer(1, 0.08 * inch))
    story.append(_image(FIGURES_DIR / "training_monitor_checkpoint_dashboard.png", 7.0 * inch))
    story.append(Spacer(1, 0.08 * inch))
    two_img_table = Table(
        [[_image(FIGURES_DIR / "training_monitor_reward_curve.png", 3.35 * inch), _image(FIGURES_DIR / "training_monitor_max_progress_curve.png", 3.35 * inch)]],
        colWidths=[3.45 * inch, 3.45 * inch],
    )
    two_img_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(two_img_table)
    story.append(Spacer(1, 0.08 * inch))
    story.append(
        Paragraph(
            f"Across the recorded checkpoint evaluations in this artifact slice, the finish rate is {len(finished_rows) / len(rows) * 100:.0f}%, mean finish time among successful checkpoints is {mean_finish_time:.2f} s, mean progress is {mean_progress:.2f} m, and mean reward is {mean_reward:.2f}. Termination counts in the recorded slice are {checkpoint_termination_counts}. That is sufficient to show the learned controller repeatedly completes the full course, while still exhibiting off-track, stall, and timeout failure modes at other checkpoints.",
            styles["BodyText"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("Artifacts And Deployment State", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "The RL pipeline generated more than scalar metrics. Each checkpoint refresh writes a live dashboard, reward and progress curves, section-success and failure histograms, plus a 3D replay snapshot. These artifacts are useful because they make it possible to inspect where a policy improved, not only whether the scalar reward happened to rise on a given generation.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 0.08 * inch))
    replay_table = Table(
        [[_image(replay_frame, 3.4 * inch), _image(compat_figure, 3.4 * inch)]],
        colWidths=[3.5 * inch, 3.5 * inch],
    )
    replay_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(replay_table)
    story.append(Spacer(1, 0.1 * inch))
    story.append(
        Paragraph(
            "The replay frame on the left is taken from the live snapshot bundle of the latest checkpoint. The figure on the right captures an important implementation detail uncovered during this supplement build: the exported policy manifest describes a 2-action network, which matches the direct-control training environment, but the runtime residual bot wrapper expects 3 outputs. In the present code this means the exported policy is considered incompatible and the runtime mixed controller falls back to the heuristic bot.",
            styles["BodyText"],
        )
    )
    story.append(
        Paragraph(
            "That does not invalidate the training work itself. The training environment, checkpoints, and final saved PPO file all show a functioning learned controller. It does, however, mean that the current playable runtime path is not yet demonstrating the learned policy directly. Closing that gap would require either exporting a residual 3-output policy or adapting the runtime integration to the 2-output direct-control policy.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 0.12 * inch))
    findings = [
        "What was undertaken: a PPO-based direct-control bot for the marble racer, trained through a staged curriculum and checkpointed with live qualitative evidence.",
        f"How it was done: 23-D observation vector, compact 64-64 MLP policy, checkpoint evaluation every 100k steps, and a snapshot pipeline that wrote plots and replay media to the monitor page.",
        f"What the outcomes were: late-stage checkpoints reached {latest.timesteps:,} timesteps, successful finishes were common in the recorded slice, the fastest logged finish was {fastest_finish.finish_time:.2f} s, and the best successful checkpoint was generation {best_reward.generation} with reward {best_reward.reward:.2f}.",
        "What remains to finish the feature: align export format and runtime controller expectations so the learned policy can drive the in-game bot path without falling back to the heuristic controller.",
    ]
    findings_table = Table([[Paragraph(item, styles["SmallBody"])] for item in findings], colWidths=[7.0 * inch])
    findings_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7f2eb")),
                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#d3c4b3")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e3d8cb")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(findings_table)

    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=letter,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.5 * inch,
        title="RL Bot Supplementary Walkthrough",
        author="Codex",
    )
    doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
    print(f"Wrote {OUTPUT_PATH}")
    print(json.dumps(report_summary, indent=2))


if __name__ == "__main__":
    main()
