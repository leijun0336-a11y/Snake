from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol


WANDB_PROJECT = "Snake"
WANDB_WORKSPACE = "Snake DQN Training Curves"


class WandbRun(Protocol):
    id: str
    entity: str
    project: str
    url: str
    summary: Any

    def define_metric(self, name: str, **kwargs: Any) -> Any: ...

    def log(self, data: Mapping[str, Any]) -> None: ...

    def finish(self, exit_code: int | None = None) -> None: ...


def _line_plot(
    wr: Any,
    *,
    run_id: str,
    title: str,
    metrics: list[str],
    y_axis: str,
    colors: list[str],
    widths: list[float],
) -> Any:
    return wr.LinePlot(
        title=title,
        x="episode",
        y=metrics,
        title_x="Episode",
        title_y=y_axis,
        smoothing_type="none",
        plot_type="line",
        point_visualization_method="bucketing-gorilla",
        line_colors={
            f"{run_id}:{metric}": color for metric, color in zip(metrics, colors, strict=True)
        },
        line_widths={
            f"{run_id}:{metric}": width for metric, width in zip(metrics, widths, strict=True)
        },
    )


def configure_workspace(entity: str, project: str, run_id: str, run_name: str) -> str:
    """Create the fixed 2x3 W&B view matching the local training-curves figure."""
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    panels = [
        _line_plot(
            wr,
            run_id=run_id,
            title="Score",
            metrics=["score", "score_rolling50", "mean_score_100"],
            y_axis="Score",
            colors=["#9ecae1", "#1685e5", "#e31a1c"],
            widths=[0.5, 2.0, 2.0],
        ),
        _line_plot(
            wr,
            run_id=run_id,
            title="Reward",
            metrics=["episode_reward", "mean_reward_100"],
            y_axis="Reward",
            colors=["#a8e6c8", "#159447"],
            widths=[0.5, 2.0],
        ),
        _line_plot(
            wr,
            run_id=run_id,
            title="Episode Steps",
            metrics=["steps", "steps_rolling50"],
            y_axis="Steps",
            colors=["#f8d49a", "#cc6e12"],
            widths=[0.5, 2.0],
        ),
        _line_plot(
            wr,
            run_id=run_id,
            title="Loss",
            metrics=["loss", "mean_loss_100"],
            y_axis="Loss",
            colors=["#d8c8fa", "#6f2cff"],
            widths=[0.5, 2.0],
        ),
        _line_plot(
            wr,
            run_id=run_id,
            title="Epsilon",
            metrics=["epsilon"],
            y_axis="Epsilon",
            colors=["#168c8c"],
            widths=[2.0],
        ),
        _line_plot(
            wr,
            run_id=run_id,
            title="Replay Buffer Size",
            metrics=["replay_buffer_size"],
            y_axis="Transitions",
            colors=["#596b7d"],
            widths=[2.0],
        ),
    ]
    workspace = ws.Workspace(
        name=f"{WANDB_WORKSPACE} - {run_name}",
        entity=entity,
        project=project,
        sections=[
            ws.Section(
                name="Training Curves",
                panels=panels,
                is_open=True,
                layout_settings=ws.SectionLayoutSettings(columns=2, rows=3),
            )
        ],
        settings=ws.WorkspaceSettings(
            x_axis="episode",
            smoothing_type="none",
            sort_panels_alphabetically=False,
            point_visualization_method="bucketing",
        ),
        runset_settings=ws.RunsetSettings(
            filters=f"Name = '{run_name}'",
            pinned_runs=[run_id],
        ),
        auto_generate_panels=False,
    ).save()
    return workspace.url


def start_wandb(
    *,
    run_name: str,
    run_dir: Path,
    config: Mapping[str, Any],
) -> WandbRun:
    """Start online W&B logging and install the required workspace layout."""
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - dependency is declared by the project
        raise RuntimeError(
            "--wandb requires the project's W&B dependencies; run `uv sync`"
        ) from exc

    run = wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        config=dict(config),
        dir=str(run_dir),
        mode="online",
    )
    if run is None:
        raise RuntimeError("wandb.init() did not return a run")

    run.define_metric("episode", hidden=True)
    run.define_metric("*", step_metric="episode")
    try:
        workspace_url = configure_workspace(run.entity, run.project, run.id, run_name)
    except Exception as exc:
        run.finish(exit_code=1)
        raise RuntimeError(
            "W&B run started, but the required 2x3 workspace layout could not be configured"
        ) from exc

    print(f"wandb_run={run.url}")
    print(f"wandb_workspace={workspace_url}")
    return run


def training_metrics(
    *,
    episode: int,
    scores: list[int],
    mean_score_100: float,
    episode_reward: float,
    mean_reward_100: float,
    episode_steps: list[int],
    loss: float,
    mean_loss_100: float,
    epsilon: float,
    replay_buffer_size: int,
) -> dict[str, float | int]:
    """Build the single per-episode payload used by every configured W&B panel."""
    return {
        "episode": episode,
        "score": scores[-1],
        "score_rolling50": sum(scores[-50:]) / min(len(scores), 50),
        "mean_score_100": mean_score_100,
        "episode_reward": episode_reward,
        "mean_reward_100": mean_reward_100,
        "steps": episode_steps[-1],
        "steps_rolling50": sum(episode_steps[-50:]) / min(len(episode_steps), 50),
        "loss": loss,
        "mean_loss_100": mean_loss_100,
        "epsilon": epsilon,
        "replay_buffer_size": replay_buffer_size,
    }
