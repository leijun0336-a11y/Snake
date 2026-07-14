import pytest
import torch

from snake_ai.utils import set_seed


def test_set_seed_toggles_strict_deterministic_algorithms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[bool, dict[str, object]]] = []

    def record_deterministic_call(mode: bool, **kwargs: object) -> None:
        calls.append((mode, kwargs))

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", record_deterministic_call)
    monkeypatch.setattr(
        torch.backends.cudnn,
        "deterministic",
        torch.backends.cudnn.deterministic,
    )
    monkeypatch.setattr(
        torch.backends.cudnn,
        "benchmark",
        torch.backends.cudnn.benchmark,
    )
    monkeypatch.setenv("PYTHONHASHSEED", "test-original")

    set_seed(7, deterministic=True)
    set_seed(7, deterministic=False)

    assert calls == [(True, {}), (False, {})]
