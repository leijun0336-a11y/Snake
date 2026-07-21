import sys
from pathlib import Path

import pytest

from snake_ai.evaluate import (
    build_configs,
    find_latest_checkpoint,
    get_checkpoint_algorithm,
    open_eval_metrics_csv,
    parse_args,
)


def test_build_configs_resolves_evaluation_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["evaluate.py"])

    train_config, env_config = build_configs(parse_args())

    assert train_config.seed == 42
    assert env_config.width == 20
    assert env_config.height == 20


def test_build_configs_reads_board_size_from_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import torch

    checkpoint_path = tmp_path / "latest.pt"
    torch.save(
        {"run_config": {"environment": {"width": 10, "height": 10}}},
        checkpoint_path,
    )
    monkeypatch.setattr(sys, "argv", ["evaluate.py"])

    _, env_config = build_configs(parse_args(), checkpoint_path)

    assert (env_config.width, env_config.height) == (10, 10)


def test_build_configs_matches_environment_minimum_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--height", "4"])

    with pytest.raises(ValueError, match="at least 5"):
        build_configs(parse_args())


def test_find_latest_checkpoint_uses_latest_experiment_latest_pt(
    tmp_path: Path,
) -> None:
    older = tmp_path / "dqn_20260101_000000"
    newer = tmp_path / "dqn_20260102_000000"
    older.mkdir()
    newer.mkdir()
    (older / "latest.pt").touch()
    (newer / "latest.pt").touch()
    (newer / "best.pt").touch()

    assert find_latest_checkpoint(tmp_path) == newer / "latest.pt"


def test_find_latest_checkpoint_compares_dqn_and_ppo_timestamps(tmp_path: Path) -> None:
    dqn = tmp_path / "dqn_20260103_000000"
    ppo = tmp_path / "ppo_20260102_000000"
    dqn.mkdir()
    ppo.mkdir()
    (dqn / "latest.pt").touch()
    (ppo / "latest.pt").touch()

    assert find_latest_checkpoint(tmp_path) == dqn / "latest.pt"
    assert find_latest_checkpoint(tmp_path, algorithm="ppo") == ppo / "latest.pt"


def test_checkpoint_algorithm_defaults_legacy_checkpoint_to_dqn(tmp_path: Path) -> None:
    import torch

    legacy = tmp_path / "legacy.pt"
    ppo = tmp_path / "ppo.pt"
    torch.save({"policy_net": {}}, legacy)
    torch.save({"algorithm": "ppo", "policy_net": {}}, ppo)

    assert get_checkpoint_algorithm(legacy) == "dqn"
    assert get_checkpoint_algorithm(ppo) == "ppo"


def test_find_latest_checkpoint_does_not_fallback_to_best_pt(tmp_path: Path) -> None:
    latest_run = tmp_path / "dqn_20260102_000000"
    latest_run.mkdir()
    (latest_run / "best.pt").touch()

    with pytest.raises(FileNotFoundError, match=r"No latest\.pt found"):
        find_latest_checkpoint(tmp_path)


def test_open_eval_metrics_csv_overwrites_previous_evaluation(tmp_path: Path) -> None:
    csv_path = tmp_path / "eval_metrics.csv"
    csv_path.write_text(
        "episode,score,steps,score_per_step,max_snake_length\n"
        "1,99,999,0.099099,102\n",
        encoding="utf-8",
    )

    csv_file, metrics = open_eval_metrics_csv(csv_path)
    try:
        metrics.writerow([1, 7, 42, "0.166667", 10])
    finally:
        csv_file.close()

    assert csv_path.read_text(encoding="utf-8").splitlines() == [
        "episode,score,steps,score_per_step,max_snake_length",
        "1,7,42,0.166667,10",
    ]


def test_open_eval_metrics_csv_truncates_interrupted_run_immediately(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "eval_metrics.csv"
    csv_path.write_text("old interrupted evaluation\n", encoding="utf-8")

    csv_file, _ = open_eval_metrics_csv(csv_path)
    try:
        assert csv_path.read_text(encoding="utf-8").splitlines() == [
            "episode,score,steps,score_per_step,max_snake_length"
        ]
    finally:
        csv_file.close()
