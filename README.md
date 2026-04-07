# POV Marbleracer

Panda3D + Bullet marble racing project with a playable downhill course, authored obstacles, time-trial ghosts, bot racing, training utilities, and report-generation scripts.

## Overview

This repository contains three main pieces:

- A playable marble racing game in `src/marbleracer` with `Time Trial` and `Bot Race` modes.
- Content and tooling for levels, figure generation, training runs, and evaluation in `levels/`, `scripts/`, `figures/`, and `artifacts/`.
- Project documentation and outputs, including the bundled report PDF.

Core gameplay features:

- banked downhill track physics using Panda3D + Bullet
- boost pads, rails, static and moving obstacles
- ghost replay for best time-trial runs
- bot marbles and RL/training support
- report and figure generation scripts for the project writeup

## Repo Layout

Important paths:

- `src/marbleracer/`: main application, physics, gameplay rules, builder, bot controller, and training environment
- `levels/`: authored level JSON files
- `scripts/`: figure generation, training, evaluation, monitoring, and report helper scripts
- `figures/`: generated figures and exported visual assets
- `artifacts/`: training logs, monitor outputs, and snapshots
- `tests/`: unit tests
- `Development_of_a_marble_racing_simulator.pdf`: project report

Useful entry points:

- `src/marbleracer/main.py`: game launcher
- `src/marbleracer/builder.py`: level/builder tooling
- `scripts/generate_report_figures.py`: regenerate the figure assets used in the report

## Getting Started

The project targets Python `3.11+`.

Create a virtual environment and install dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Run the game:

```bash
marbleracer
```

Or directly:

```bash
python -m marbleracer.main
```

List available levels:

```bash
marbleracer --list-levels
```

Start a specific mode:

```bash
marbleracer --level default --mode time_trial
marbleracer --level default --mode bot_race
```

Reset saved progress:

```bash
marbleracer --reset-progress
```

## Development

Run tests:

```bash
pytest
```

Regenerate report figures:

```bash
python scripts/generate_report_figures.py
```

Some scripts in `scripts/` are for RL training and analysis. Those use the packages already listed in `requirements.txt`, including PyTorch, Gymnasium, Stable-Baselines3, pandas, matplotlib, Pillow, and ReportLab.

## Controls

- `Up/Down`: change menu mode
- `Left/Right`: change selected track in menus
- `Enter` or `Space`: confirm/start
- `Left/Right`: steer while racing
- `Down`: brake
- `Space`: pause
- `R`: restart
- `M`: return to menu
- `Esc`: quit

## Report

Open the bundled report here:

- [Development_of_a_marble_racing_simulator.pdf](./Development_of_a_marble_racing_simulator.pdf)

Embedded preview:

<object data="./Development_of_a_marble_racing_simulator.pdf" type="application/pdf" width="100%" height="700">
  <p>PDF preview not available in this Markdown renderer. Open the report directly:
  <a href="./Development_of_a_marble_racing_simulator.pdf">Development_of_a_marble_racing_simulator.pdf</a></p>
</object>
