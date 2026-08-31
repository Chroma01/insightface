from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from insightface.app.privateframe import recognition, streaming


class _Recognizer:
    input_size = (112, 112)

    def __init__(self, *outputs: Any) -> None:
        self.outputs = list(outputs)
        self.calls = 0

    def get_feat(self, _aligned: np.ndarray) -> np.ndarray:
        output = self.outputs[self.calls]
        self.calls += 1
        if isinstance(output, BaseException):
            raise output
        return np.asarray(output)


def _candidate(frame_index: int = 0) -> recognition.RecognitionCandidate:
    return recognition.RecognitionCandidate(
        frame_index=frame_index,
        quality=1.0,
        aligned_face=np.zeros((112, 112, 3), dtype=np.uint8),
    )


def _sourced_candidate(
    frame_index: int,
    quality: float,
    landmark_source: str,
) -> recognition.RecognitionCandidate:
    return recognition.RecognitionCandidate(
        frame_index=frame_index,
        quality=quality,
        aligned_face=np.zeros((112, 112, 3), dtype=np.uint8),
        landmark_source=landmark_source,
    )


def test_recognition_candidate_defaults_to_local_scrfd_landmarks() -> None:
    assert _candidate().landmark_source == "local_scrfd"


def test_recognition_candidate_rejects_unknown_landmark_source() -> None:
    with pytest.raises(ValueError, match="landmark_source"):
        _sourced_candidate(0, 1.0, "optical_flow")


def test_temporal_selection_prefers_local_and_falls_back_to_global() -> None:
    candidates = [
        _sourced_candidate(0, 0.99, "global_scrfd"),
        _sourced_candidate(2, 0.40, "local_scrfd"),
        _sourced_candidate(8, 0.60, "global_scrfd"),
        _sourced_candidate(9, 0.90, "global_scrfd"),
    ]

    selected = recognition.select_temporally_distributed(candidates, 2)

    assert [value.frame_index for value in selected] == [2, 9]
    assert [value.landmark_source for value in selected] == [
        "local_scrfd",
        "global_scrfd",
    ]
    assert [
        (value.frame_index, value.landmark_source)
        for value in recognition.select_temporally_distributed(
            list(reversed(candidates)),
            2,
        )
    ] == [
        (2, "local_scrfd"),
        (9, "global_scrfd"),
    ]


def test_temporal_selection_uses_only_one_proposal_from_the_same_frame() -> None:
    candidates = [
        _sourced_candidate(5, 1.00, "global_scrfd"),
        _sourced_candidate(5, 0.25, "local_scrfd"),
        _sourced_candidate(12, 0.50, "global_scrfd"),
    ]

    selected = recognition.select_temporally_distributed(candidates, 3)

    assert [(value.frame_index, value.landmark_source) for value in selected] == [
        (5, "local_scrfd"),
        (12, "global_scrfd"),
    ]


def _recognition_engine(recognizer: Any) -> recognition.RecognitionEngine:
    gallery = SimpleNamespace(
        prototypes={
            "alice": np.asarray([0.6, 0.8], dtype=np.float32),
        },
        references=(),
        rejections=(),
        fingerprint="gallery-fingerprint",
    )
    return recognition.RecognitionEngine(
        enabled=True,
        mode="exempt",
        profile=recognition.RECOGNITION_PROFILES["fast"],
        target_persons=("alice",),
        similarity_threshold=0.5,
        recognizer=recognizer,
        gallery=gallery,
    )


@pytest.mark.parametrize(
    ("output", "error_name"),
    [
        (RuntimeError("onnx failure details"), "RuntimeError"),
        (np.asarray([[0.0, 0.0]], dtype=np.float32), "ValueError"),
        (np.asarray([3.0, 4.0], dtype=np.float32), "RuntimeError"),
    ],
)
def test_track_embedding_failure_returns_diagnostic_unknown_and_blurs(
    output: Any,
    error_name: str,
) -> None:
    engine = _recognition_engine(_Recognizer(output))

    decision = engine.identify_track(
        [_candidate()],
        frames_per_second=30.0,
    )

    assert decision.status is recognition.IdentityStatus.UNKNOWN
    assert decision.person_id is None
    assert decision.reason == f"track_recognition_error:{error_name}"
    assert decision.selected_frame_count == 1
    assert "onnx failure details" not in decision.reason
    for mode in ("blur_only", "exempt"):
        policy = recognition.apply_identity_policy(
            mode,
            decision,
            ("alice",),
        )
        assert policy.should_blur is True


def test_track_decision_failure_returns_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _recognition_engine(
        _Recognizer(np.asarray([[3.0, 4.0]], dtype=np.float32))
    )

    def fail_decision(*_args: Any, **_kwargs: Any) -> Any:
        raise LookupError("corrupt gallery prototype")

    monkeypatch.setattr(
        recognition,
        "decide_track_identity",
        fail_decision,
    )

    decision = engine.identify_track(
        [_candidate()],
        frames_per_second=30.0,
    )

    assert decision.status is recognition.IdentityStatus.UNKNOWN
    assert decision.reason == "track_recognition_error:LookupError"
    assert decision.selected_frame_count == 1


@pytest.mark.parametrize(
    "fatal_error",
    [KeyboardInterrupt(), SystemExit(2), MemoryError()],
    ids=["keyboard-interrupt", "system-exit", "memory-error"],
)
def test_track_does_not_swallow_fatal_errors(
    fatal_error: BaseException,
) -> None:
    engine = _recognition_engine(_Recognizer(fatal_error))

    with pytest.raises(type(fatal_error)):
        engine.identify_track(
            [_candidate()],
            frames_per_second=30.0,
        )


def _gallery_root(tmp_path: Path, names: tuple[str, ...]) -> Path:
    root = tmp_path / "gallery"
    person = root / "alice"
    person.mkdir(parents=True)
    for index, name in enumerate(names):
        (person / name).write_bytes(f"reference-{index}".encode())
    return root


def _valid_detection() -> dict[str, Any]:
    return {
        "box": [20.0, 20.0, 100.0, 100.0],
        "confidence": 0.99,
        "landmarks": recognition.ARC_FACE_TEMPLATE_112.copy(),
    }


def _patch_gallery_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        recognition,
        "arcface_align_112",
        lambda _image, _landmarks: np.zeros(
            (112, 112, 3),
            dtype=np.uint8,
        ),
    )
    monkeypatch.setattr(
        recognition,
        "recognition_candidate_quality",
        lambda *_args, **_kwargs: (1.0, True, {}),
    )


def test_gallery_rejects_one_recognizer_failure_and_keeps_valid_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_gallery_geometry(monkeypatch)
    root = _gallery_root(tmp_path, ("bad.jpg", "good.jpg"))
    recognizer = _Recognizer(
        RuntimeError("provider failure details"),
        np.asarray([[3.0, 4.0]], dtype=np.float32),
    )

    gallery = recognition.build_gallery(
        root,
        lambda _image: [_valid_detection()],
        recognizer,
        target_persons=("alice",),
        image_loader=lambda _path: np.zeros((128, 128, 3), dtype=np.uint8),
    )

    assert [value.file_name for value in gallery.references] == ["good.jpg"]
    assert [value.file_name for value in gallery.rejections] == ["bad.jpg"]
    assert gallery.rejections[0].reason == (
        "recognizer_inference_error:RuntimeError"
    )
    assert "provider failure details" not in gallery.rejections[0].reason
    assert set(gallery.prototypes) == {"alice"}
    np.testing.assert_allclose(
        gallery.references[0].embedding,
        np.asarray([0.6, 0.8], dtype=np.float32),
    )


def test_gallery_rejects_one_detector_failure_and_continues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_gallery_geometry(monkeypatch)
    root = _gallery_root(tmp_path, ("bad.jpg", "good.jpg"))
    detector_calls = 0

    def detector(_image: np.ndarray) -> list[dict[str, Any]]:
        nonlocal detector_calls
        detector_calls += 1
        if detector_calls == 1:
            raise RuntimeError("detector provider failure")
        return [_valid_detection()]

    gallery = recognition.build_gallery(
        root,
        detector,
        _Recognizer(np.asarray([[3.0, 4.0]], dtype=np.float32)),
        target_persons=("alice",),
        image_loader=lambda _path: np.zeros((128, 128, 3), dtype=np.uint8),
    )

    assert [value.file_name for value in gallery.references] == ["good.jpg"]
    assert gallery.rejections[0].reason == (
        "detector_inference_error:RuntimeError"
    )


def test_gallery_with_no_valid_target_reference_still_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_gallery_geometry(monkeypatch)
    root = _gallery_root(tmp_path, ("only.jpg",))

    with pytest.raises(
        ValueError,
        match="target persons have no usable reference images.*alice",
    ):
        recognition.build_gallery(
            root,
            lambda _image: [_valid_detection()],
            _Recognizer(RuntimeError("recognizer unavailable")),
            target_persons=("alice",),
            image_loader=lambda _path: np.zeros(
                (128, 128, 3),
                dtype=np.uint8,
            ),
        )


@pytest.mark.parametrize(
    "fatal_type",
    [KeyboardInterrupt, SystemExit, MemoryError],
)
def test_gallery_does_not_swallow_fatal_recognizer_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fatal_type: type[BaseException],
) -> None:
    _patch_gallery_geometry(monkeypatch)
    root = _gallery_root(tmp_path, ("only.jpg",))

    with pytest.raises(fatal_type):
        recognition.build_gallery(
            root,
            lambda _image: [_valid_detection()],
            _Recognizer(fatal_type()),
            target_persons=("alice",),
            image_loader=lambda _path: np.zeros(
                (128, 128, 3),
                dtype=np.uint8,
            ),
        )


class _StreamingRecognitionEngine:
    enabled = True
    mode = "exempt"
    profile = recognition.RECOGNITION_PROFILES["fast"]
    target_persons = ("alice",)
    similarity_threshold = 0.5
    gallery = SimpleNamespace(
        prototypes={"alice": np.asarray([0.6, 0.8], dtype=np.float32)},
        references=(),
        rejections=(),
        fingerprint="gallery-fingerprint",
    )

    def __init__(self) -> None:
        self.calls = 0

    @property
    def max_frames_per_track(self) -> int:
        return self.profile.max_frames_per_track

    def identify_track(
        self,
        _candidates: Any,
        *,
        frames_per_second: float,
    ) -> recognition.IdentityDecision:
        assert frames_per_second == 30.0
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("unexpected custom engine failure")
        return recognition.IdentityDecision(
            recognition.IdentityStatus.CONFIRMED,
            person_id="alice",
            reason="confirmed",
        )

    @staticmethod
    def unknown_decision(reason: str) -> recognition.IdentityDecision:
        return recognition.IdentityDecision(
            recognition.IdentityStatus.UNKNOWN,
            reason=reason,
        )


def _bare_streaming_engine(
    recognition_engine: Any,
) -> streaming.StreamingEngine:
    engine = object.__new__(streaming.StreamingEngine)
    engine.recognition_engine = recognition_engine
    engine.fps = 30.0
    engine.tracks = [
        {
            "accepted": True,
            "track_id": "track-1",
            "detections": [],
            "accepted_intervals": [],
        },
        {
            "accepted": True,
            "track_id": "track-2",
            "detections": [],
            "accepted_intervals": [],
        },
    ]
    engine.recognition_candidates = {}
    engine.recognition_candidate_overflow_tracks = set()
    engine.recognition_candidates_prepared = 0
    engine.recognition_candidate_errors = 0
    engine.recognition_candidate_rejections = 0
    engine.recognition_setup_seconds = 0.0
    engine.recognition_candidate_prepare_seconds = 0.0
    return engine


def test_streaming_isolates_unexpected_failure_to_one_track() -> None:
    recognition_engine = _StreamingRecognitionEngine()
    engine = _bare_streaming_engine(recognition_engine)

    artifact = engine._finalize_recognition_impl({})

    assert recognition_engine.calls == 2
    first = artifact["tracks"]["track-1"]
    second = artifact["tracks"]["track-2"]
    assert first["status"] == "UNKNOWN"
    assert first["reason"] == "track_recognition_error:RuntimeError"
    assert second["status"] == "CONFIRMED"
    assert second["person_id"] == "alice"
    assert artifact["statistics"]["status_counts"] == {
        "CONFIRMED": 1,
        "UNKNOWN": 1,
        "CONFLICT": 0,
    }
