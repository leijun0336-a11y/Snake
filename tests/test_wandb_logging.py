from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from snake_ai import wandb_logging


class FakeRun:
    id = "test-run-id"
    entity = "test-entity"
    project = "Snake"
    url = "https://wandb.invalid/run"

    def __init__(self) -> None:
        self.defined_metrics: list[tuple[str, dict[str, Any]]] = []
        self.finished_with: list[int | None] = []
        self.summary: dict[str, Any] = {}

    def define_metric(self, name: str, **kwargs: Any) -> None:
        self.defined_metrics.append((name, kwargs))

    def finish(self, exit_code: int | None = None) -> None:
        self.finished_with.append(exit_code)


def test_training_metrics_matches_every_reference_curve() -> None:
    metrics = wandb_logging.training_metrics(
        episode=3,
        scores=[1, 2, 6],
        mean_score_100=3.0,
        episode_reward=12.5,
        mean_reward_100=4.5,
        episode_steps=[10, 20, 30],
        loss=0.25,
        mean_loss_100=0.5,
        epsilon=0.75,
        replay_buffer_size=60,
    )

    assert metrics == {
        "episode": 3,
        "score": 6,
        "score_rolling50": 3.0,
        "mean_score_100": 3.0,
        "episode_reward": 12.5,
        "mean_reward_100": 4.5,
        "steps": 30,
        "steps_rolling50": 20.0,
        "loss": 0.25,
        "mean_loss_100": 0.5,
        "DQN-only/epsilon": 0.75,
        "DQN-only/replay_buffer_size": 60,
    }


def test_training_metrics_emits_ppo_diagnostics_only_after_update() -> None:
    common = dict(
        episode=3,
        scores=[1, 2, 6],
        mean_score_100=3.0,
        episode_reward=12.5,
        mean_reward_100=4.5,
        episode_steps=[10, 20, 30],
        loss=0.25,
        mean_loss_100=0.5,
        algorithm="ppo",
    )

    before_update = wandb_logging.training_metrics(**common)
    after_update = wandb_logging.training_metrics(
        **common,
        ppo_metrics={"policy_loss": -0.1, "approx_kl": 0.01},
    )

    assert "loss" not in before_update
    assert after_update["loss"] == pytest.approx(0.25)
    assert after_update["PPO-only/policy_loss"] == pytest.approx(-0.1)
    assert after_update["PPO-only/approx_kl"] == pytest.approx(0.01)


def test_start_wandb_uses_snake_project_and_episode_axis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run = FakeRun()
    init_calls: list[dict[str, Any]] = []
    fake_wandb = ModuleType("wandb")

    def fake_init(**kwargs: Any) -> FakeRun:
        init_calls.append(kwargs)
        return run

    fake_wandb.init = fake_init  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr(
        wandb_logging,
        "configure_workspace",
        lambda entity, project, run_id, run_name: (
            f"https://wandb.invalid/{entity}/{project}/{run_id}/{run_name}"
        ),
    )

    result = wandb_logging.start_wandb(
        run_name="dqn_test",
        run_dir=tmp_path,
        config={"seed": 42},
    )

    assert result is run
    assert init_calls == [
        {
            "project": "Snake",
            "name": "dqn_test",
            "config": {"seed": 42},
            "dir": str(tmp_path),
            "mode": "online",
        }
    ]
    assert run.defined_metrics == [
        ("episode", {"hidden": True}),
        ("*", {"step_metric": "episode"}),
    ]


def test_start_wandb_does_not_fallback_when_layout_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run = FakeRun()
    fake_wandb = ModuleType("wandb")
    fake_wandb.init = lambda **kwargs: run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    def fail_layout(entity: str, project: str, run_id: str, run_name: str) -> str:
        raise PermissionError("workspace denied")

    monkeypatch.setattr(wandb_logging, "configure_workspace", fail_layout)

    with pytest.raises(RuntimeError, match="2x3 workspace layout") as exc_info:
        wandb_logging.start_wandb(
            run_name="dqn_test",
            run_dir=tmp_path,
            config={},
        )

    assert isinstance(exc_info.value.__cause__, PermissionError)
    assert run.finished_with == [1]


def test_configure_workspace_keeps_reference_panel_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeWorkspace:
        url = "https://wandb.invalid/workspace"

        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def save(self) -> FakeWorkspace:
            return self

    fake_wr = ModuleType("wandb_workspaces.reports.v2")
    fake_wr.LinePlot = lambda **kwargs: SimpleNamespace(**kwargs)  # type: ignore[attr-defined]
    fake_ws = ModuleType("wandb_workspaces.workspaces")
    fake_ws.Workspace = FakeWorkspace  # type: ignore[attr-defined]
    fake_ws.Section = lambda **kwargs: SimpleNamespace(**kwargs)  # type: ignore[attr-defined]
    fake_ws.SectionLayoutSettings = lambda **kwargs: SimpleNamespace(  # type: ignore[attr-defined]
        **kwargs
    )
    fake_ws.WorkspaceSettings = lambda **kwargs: SimpleNamespace(  # type: ignore[attr-defined]
        **kwargs
    )
    fake_ws.RunsetSettings = lambda **kwargs: SimpleNamespace(  # type: ignore[attr-defined]
        **kwargs
    )
    package = ModuleType("wandb_workspaces")
    reports = ModuleType("wandb_workspaces.reports")
    reports.v2 = fake_wr  # type: ignore[attr-defined]
    package.reports = reports  # type: ignore[attr-defined]
    package.workspaces = fake_ws  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb_workspaces", package)
    monkeypatch.setitem(sys.modules, "wandb_workspaces.reports", reports)
    monkeypatch.setitem(sys.modules, "wandb_workspaces.reports.v2", fake_wr)
    monkeypatch.setitem(sys.modules, "wandb_workspaces.workspaces", fake_ws)

    url = wandb_logging.configure_workspace("entity", "Snake", "run-id", "dqn_test")

    sections = captured["sections"]
    assert url == "https://wandb.invalid/workspace"
    assert [section.name for section in sections] == ["Charts", "DQN-only"]
    assert (sections[0].layout_settings.columns, sections[0].layout_settings.rows) == (2, 2)
    assert [panel.title for panel in sections[0].panels] == [
        "Score",
        "Reward",
        "Episode Steps",
        "Loss",
    ]
    assert [panel.title for panel in sections[1].panels] == [
        "Epsilon",
        "Replay Buffer Size",
    ]
    assert sections[0].panels[0].y == ["score", "score_rolling50", "mean_score_100"]
    assert sections[0].panels[0].line_colors["run-id:score"] == "#9ecae1"
    assert captured["runset_settings"].filters == "Name = 'dqn_test'"

    wandb_logging.configure_workspace("entity", "Snake", "run-id", "ppo_test")
    sections = captured["sections"]
    assert [section.name for section in sections] == ["Charts", "PPO-only"]
    assert [panel.title for panel in sections[1].panels] == [
        "Policy and Value Losses",
        "Entropy",
        "PPO Diagnostics",
    ]
