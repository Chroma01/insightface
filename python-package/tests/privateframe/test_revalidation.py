from types import SimpleNamespace

import pytest
from insightface.app.privateframe import revalidation
from insightface.app.privateframe.revalidation import LocalReviewer, _finalize


def test_local_reviewer_accepts_detection_only_host_and_explicit_verifier() -> None:
    detector = object()
    verifier = object()
    detection_host = SimpleNamespace(models={"detection": detector})
    config = {
        "models": {"detection": {"max_detections": 300}},
        "revalidation": {},
    }

    reviewer = LocalReviewer(
        config,
        detector=detector,
        verifier=verifier,
        face_analysis=detection_host,
    )

    assert reviewer.face_analysis is detection_host
    assert reviewer.detector is detector
    assert reviewer.verifier is verifier


def test_finalize_keeps_one_best_geometry_per_accepted_track_frame() -> None:
    track = {
        "track_id": "face",
        "accepted": True,
        "accepted_intervals": [[5, 5]],
    }
    motion_box = [10.0, 10.0, 30.0, 30.0]
    detector_box = [11.0, 11.0, 31.0, 31.0]

    output = _finalize(
        [track],
        {
            "detections": [
                {
                    "track_id": "face",
                    "frame_idx": 5,
                    "source": "detector",
                    "box": detector_box,
                    "confidence": 0.9,
                }
            ]
        },
        {
            "observations": [
                {
                    "track_id": "face",
                    "frame_idx": 5,
                    "source": "kalman_optical_flow",
                    "box": motion_box,
                    "motion_box": motion_box,
                }
            ]
        },
        [],
    )

    assert len(output) == 1
    assert output[0]["track_id"] == "face"
    assert output[0]["source"] == "detector"
    assert output[0]["box"] == detector_box


def test_finalize_keeps_distinct_tracks_until_output_boxes_are_stabilized() -> None:
    boxes = {
        "reference": [0.0, 0.0, 100.0, 100.0],
        "overlapping": [15.0, 0.0, 115.0, 100.0],
    }
    tracks = [
        {
            "track_id": track_id,
            "accepted": True,
            "accepted_intervals": [[5, 5]],
        }
        for track_id in boxes
    ]
    detections = [
        {
            "track_id": track_id,
            "frame_idx": 5,
            "source": "detector",
            "box": box,
            "confidence": 0.9,
        }
        for track_id, box in boxes.items()
    ]

    output = _finalize(
        tracks,
        {"detections": detections},
        {"observations": []},
        [],
    )

    assert {item["track_id"] for item in output} == set(boxes)


def test_stitched_accepted_intervals_do_not_bridge_frames_without_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fragment identity does not manufacture boxes in an unobserved gap."""

    summary = {
        "detector_source_frames": 4,
        "local_match_fraction": 1.0,
        "confidence_p50": 0.9,
        "verifier_p50": 0.9,
    }
    monkeypatch.setattr(
        revalidation,
        "_summary",
        lambda *_args, **_kwargs: dict(summary),
    )
    monkeypatch.setattr(
        revalidation,
        "_admission_decision",
        lambda *_args, **_kwargs: {
            "accepted": True,
            "admission_path": "standard",
        },
    )
    track = {
        "track_id": "stitched",
        "stitched_track_ids": ["left-fragment", "right-fragment"],
    }
    evidence = [
        {
            "track_id": "stitched",
            "frame_idx": frame_idx,
            "source": "tracking",
            "box": [10.0, 10.0, 30.0, 30.0],
        }
        for frame_idx in (10, 11, 14, 15)
    ]
    config = {
        "revalidation": {
            "policy": {
                "continuity": {
                    "segment_max_center_jump": 1.0,
                    "segment_max_area_ratio": 2.0,
                }
            }
        },
        "tracking": {
            "fragment_stitching": {
                # The old interval merger bridged this two-frame hole merely
                # because it was below this threshold.
                "max_interval_gap_frames": 8,
            }
        },
        "recognition": {"mode": "all"},
        "scan": {
            "global_nms_iou": 0.45,
            "containment_threshold": 0.80,
        },
    }

    revalidation.finalize_precomputed(
        {"frames": [], "detections": []},
        [track],
        {"observations": [], "shadows": []},
        evidence,
        config,
        detector_frame_stride=1,
    )

    assert track["accepted"] is True
    assert track["accepted_intervals"] == [[10, 11], [14, 15]]


def test_overall_admission_falls_back_to_each_actual_continuity_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whole-track admission must not invent geometry or publish no interval."""

    def summary(
        values: list[dict[str, object]], *_args: object, **_kwargs: object
    ) -> dict[str, object]:
        return {
            "scope": "whole" if len(values) == 4 else "segment",
            "detector_source_frames": len(values),
            "local_match_fraction": 1.0,
            "confidence_p50": 0.9,
            "verifier_p50": 0.9,
        }

    monkeypatch.setattr(revalidation, "_summary", summary)
    monkeypatch.setattr(
        revalidation,
        "_admission_decision",
        lambda value, *_args, **_kwargs: {
            "accepted": value["scope"] == "whole",
            "admission_path": "standard",
        },
    )
    track = {"track_id": "accepted-as-a-whole"}
    evidence = [
        {
            "track_id": track["track_id"],
            "frame_idx": frame_idx,
            "source": "tracking",
            "box": [10.0, 10.0, 30.0, 30.0],
        }
        for frame_idx in (10, 11, 14, 15)
    ]
    config = {
        "revalidation": {
            "policy": {
                "continuity": {
                    "segment_max_center_jump": 1.0,
                    "segment_max_area_ratio": 2.0,
                }
            }
        },
        "tracking": {"fragment_stitching": {}},
    }

    revalidation.finalize_precomputed(
        {"frames": [], "detections": []},
        [track],
        {"observations": []},
        evidence,
        config,
        detector_frame_stride=1,
    )

    assert track["accepted"] is True
    assert track["accepted_intervals"] == [[10, 11], [14, 15]]
