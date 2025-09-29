# tests/test_runners/test_base_runner_contract.py
import pytest

try:
    from core.runners.base import BaseRunner
    from core.types.runner_result import RunnerResult
    from core.runners.dummy_runner import DummyRunner
except ModuleNotFoundError:
    from core.runners.base import BaseRunner
    from core.types.runner_result import RunnerResult
    from core.runners.dummy_runner import DummyRunner


@pytest.fixture(params=[DummyRunner])  # later: add ClassifierRunner/DetectorRunner/SegmenterRunner
def runner_impl(request) -> BaseRunner:
    return request.param()


class _Cancel:
    def __init__(self, set_now: bool = False) -> None:
        self._c = set_now

    def set(self) -> None:
        self._c = True

    def is_set(self) -> bool:
        return self._c


def test_protocol_instance(runner_impl) -> None:
    # works thanks to @runtime_checkable on BaseRunner
    assert isinstance(runner_impl, BaseRunner)


def test_returns_shape_and_types(runner_impl) -> None:
    res: RunnerResult = runner_impl.run(dataset=[], config={}, cancel_token=_Cancel(False))
    assert isinstance(res, dict)
    assert "metrics" in res and "artifacts" in res
    assert isinstance(res["metrics"], dict)
    assert isinstance(res["artifacts"], dict)

    for k, v in res["metrics"].items():
        assert isinstance(k, str)
        assert isinstance(v, (float, int))

    for k, v in res["artifacts"].items():
        assert isinstance(k, str)
        assert isinstance(v, str)


def test_respects_cancel_token(runner_impl) -> None:
    res = runner_impl.run(dataset=[], config={}, cancel_token=_Cancel(True))
    # even on cancel — must return a valid structure (possibly empty)
    assert "metrics" in res and "artifacts" in res
    assert isinstance(res["metrics"], dict)
    assert isinstance(res["artifacts"], dict)
