import json

from marbleracer.optimization import EpisodeResult, GenerationStats
from marbleracer.physics import SimulationConfig
from marbleracer.training_viz import (
    TrainingVizConfig,
    export_training_summary,
    generation_leader_index,
)


def test_generation_leader_prefers_completion_then_faster_finish() -> None:
    results = [
        EpisodeResult(8.0, False, None, 20.0, 0.62, 300, False, False, True, 0, 0.2),
        EpisodeResult(6.0, True, 7.4, 30.0, 1.0, 800, False, False, False, 0, 0.2),
        EpisodeResult(7.0, True, 6.8, 30.0, 1.0, 780, False, False, False, 0, 0.2),
    ]

    assert generation_leader_index(results) == 2


def test_export_training_summary_writes_json_and_csv(tmp_path) -> None:
    export_training_summary(
        output_dir=tmp_path,
        config=TrainingVizConfig(population_size=6, elite_count=2, generations=2),
        run_config=SimulationConfig(obstacle_count=0, auto_boost_pads=False),
        history=[
            GenerationStats(0, 1.2, 0.6, 0.25, None),
            GenerationStats(1, 2.1, 1.1, 0.50, 9.4),
        ],
        winner_history=[
            {"generation": 0, "winner_index": 1, "progress_ratio": 0.5, "completed": False, "finish_time": None, "reward": 1.2},
            {"generation": 1, "winner_index": 0, "progress_ratio": 1.0, "completed": True, "finish_time": 9.4, "reward": 2.1},
        ],
    )

    summary = json.loads((tmp_path / "training_summary.json").read_text())
    csv_text = (tmp_path / "generation_history.csv").read_text()

    assert summary["training"]["population_size"] == 6
    assert summary["history"][1]["best_finish_time"] == 9.4
    assert "winner_index" in csv_text
    assert "0.5" in csv_text
