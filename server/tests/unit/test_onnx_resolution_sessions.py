from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from insightface_server.inference.onnx_engine import OnnxInsightFaceEngine


class _RecordingSession:
    def __init__(self, name: str, providers: list[str]) -> None:
        self.name = name
        self._providers = providers
        self.runs: list[tuple[Any, dict[str, np.ndarray]]] = []

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=f"{self.name}_input")]

    def get_providers(self) -> list[str]:
        return list(self._providers)

    def run(self, outputs: Any, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.runs.append((outputs, inputs))
        return [np.ones((1,), dtype=np.float32)]


class _RoutingDetector:
    def __init__(self, sessions: dict[tuple[int, int], _RecordingSession]) -> None:
        self.resolution_sessions = sessions
        self.routes: list[tuple[int, int]] = []

    def _session_for_input_size(
        self, input_size: tuple[int, int]
    ) -> _RecordingSession:
        size = (int(input_size[0]), int(input_size[1]))
        self.routes.append(size)
        return self.resolution_sessions[size]


def _engine(
    detector: _RoutingDetector, primary: _RecordingSession
) -> OnnxInsightFaceEngine:
    engine = object.__new__(OnnxInsightFaceEngine)
    engine._detector = detector
    engine._detector_session = primary
    engine._detector_profile_audits = {}
    engine._detector_profile_failures = {}
    engine._runtime = {}
    return engine


def test_detector_warmup_routes_every_resolution_through_scrfd() -> None:
    small = _RecordingSession("small", ["CPUExecutionProvider"])
    large = _RecordingSession("large", ["CPUExecutionProvider"])
    detector = _RoutingDetector({(96, 128): small, (512, 640): large})
    engine = _engine(detector, small)

    engine._warm_up_detector_sizes(((96, 128), (512, 640)))

    assert detector.routes == [(96, 128), (512, 640)]
    assert list(small.runs[0][1]) == ["small_input"]
    assert small.runs[0][1]["small_input"].shape == (1, 3, 128, 96)
    assert list(large.runs[0][1]) == ["large_input"]
    assert large.runs[0][1]["large_input"].shape == (1, 3, 640, 512)


def test_detector_runtime_metadata_keeps_primary_key_and_lists_resolutions() -> None:
    shared = _RecordingSession(
        "shared", ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    large = _RecordingSession(
        "large", ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    detector = _RoutingDetector(
        {(96, 96): shared, (128, 128): shared, (512, 512): large}
    )
    engine = _engine(detector, shared)

    metadata = engine._detector_runtime_metadata()

    assert metadata["detector_session_providers"] == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert metadata["detector_resolution_session_count"] == 2
    assert metadata["detector_static_shape_sessions"] is True
    assert metadata["detector_resolution_session_providers"] == {
        "96x96": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "128x128": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "512x512": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    }


def test_unused_primary_session_is_not_counted_as_a_resolution_session() -> None:
    primary = _RecordingSession("primary", ["CPUExecutionProvider"])
    fixed = _RecordingSession("fixed", ["CPUExecutionProvider"])
    detector = _RoutingDetector({(512, 512): fixed})
    engine = _engine(detector, primary)

    records = engine._detector_sessions_with_sizes()

    assert records == [(fixed, ((512, 512),))]
    assert engine._detector_runtime_metadata()[
        "detector_resolution_session_count"
    ] == 1


def test_server_resolution_factory_passes_static_model_bytes_to_ort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from insightface.model_zoo import scrfd as scrfd_module

    reference = _RecordingSession("reference", ["CPUExecutionProvider"])
    engine = object.__new__(OnnxInsightFaceEngine)
    static_calls: list[tuple[str, tuple[int, int], str]] = []
    new_session_calls: list[tuple[object, dict[str, object]]] = []

    def static_model(
        model_file: str,
        input_size: tuple[int, int],
        input_name: str,
    ) -> bytes:
        static_calls.append((model_file, input_size, input_name))
        return b"fixed-scrfd"

    def new_session(model_source: object, **kwargs: object) -> str:
        new_session_calls.append((model_source, kwargs))
        return "fixed-session"

    monkeypatch.setattr(scrfd_module, "_static_scrfd_model", static_model)
    engine._new_session = new_session  # type: ignore[method-assign]

    session = engine._new_detector_resolution_session(
        "/models/det.onnx",
        (384, 256),
        reference,
    )

    assert session == "fixed-session"
    assert static_calls == [
        ("/models/det.onnx", (384, 256), "reference_input")
    ]
    assert new_session_calls == [
        (
            b"fixed-scrfd",
            {
                "profile_context": "detector-384x256",
                "profile_identity": "/models/det.onnx",
            },
        )
    ]


@pytest.mark.parametrize(
    ("provider", "input_shape", "expected_contexts"),
    [
        ("CPUExecutionProvider", [1, 3, 640, 640], [None]),
        ("CUDAExecutionProvider", [1, 3, "height", "width"], [None]),
        (
            "CUDAExecutionProvider",
            [1, 3, 640, 640],
            ["detector-640x640"],
        ),
    ],
)
def test_native_static_cuda_main_session_is_profiled_for_audit(
    provider: str,
    input_shape: list[object],
    expected_contexts: list[str | None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from insightface.model_zoo import scrfd as scrfd_module

    engine = object.__new__(OnnxInsightFaceEngine)
    engine._provider = provider
    model_path = Path("/models/det.onnx")
    engine.bundle = SimpleNamespace(detector=SimpleNamespace(path=model_path))
    calls: list[tuple[object, str | None, object]] = []
    native_size = (
        (int(input_shape[3]), int(input_shape[2]))
        if all(isinstance(value, int) for value in input_shape[2:4])
        else None
    )
    monkeypatch.setattr(
        scrfd_module,
        "_native_model_input_size",
        lambda _model_path: native_size,
    )

    class Session:
        def get_inputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="input", shape=input_shape)]

    def new_session(
        model_source: object,
        *,
        profile_context: str | None = "primary",
        profile_identity: object = None,
    ) -> Session:
        calls.append((model_source, profile_context, profile_identity))
        return Session()

    engine._new_session = new_session  # type: ignore[method-assign]

    selected = engine._new_detector_main_session()

    assert isinstance(selected, Session)
    assert [context for _source, context, _identity in calls] == expected_contexts
    if expected_contexts == ["detector-640x640"]:
        assert calls[0] == (
            model_path,
            "detector-640x640",
            model_path,
        )


def test_cuda_detector_audit_profiles_each_unique_session_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = _RecordingSession(
        "shared", ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    large = _RecordingSession(
        "large", ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    detector = _RoutingDetector(
        {(96, 96): shared, (128, 128): shared, (512, 512): large}
    )
    engine = _engine(detector, shared)
    finished: list[tuple[_RecordingSession, str]] = []

    def finish(
        session: _RecordingSession, *, model_name: str
    ) -> dict[str, object]:
        finished.append((session, model_name))
        return {
            "accepted": True,
            "cuda_kernel_count": 2,
            "cpu_shape_kernel_count": 1,
            "cpu_shape_operators": {"Gather": 1},
            "policy": (
                "CUDA compute required; CPU limited to small integer shape metadata"
            ),
        }

    monkeypatch.setattr(
        OnnxInsightFaceEngine,
        "_finish_cuda_profile",
        staticmethod(finish),
    )

    first = engine._audit_detector_sessions()
    second = engine._audit_detector_sessions()

    assert finished == [
        (shared, "detector[96x96,128x128]"),
        (large, "detector[512x512]"),
    ]
    assert first == second
    assert first["accepted"] is True
    assert first["session_count"] == 2
    assert first["cuda_kernel_count"] == 4
    assert first["cpu_shape_kernel_count"] == 2
    assert first["cpu_shape_operators"] == {"Gather": 2}
    session_audits = cast(list[dict[str, object]], first["sessions"])
    assert [item["input_sizes"] for item in session_audits] == [
        [[96, 96], [128, 128]],
        [[512, 512]],
    ]
