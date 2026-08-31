from types import SimpleNamespace

import pytest
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


@pytest.mark.parametrize(
    ("suppressor_box", "expected_tracks"),
    [
        ([25.0, 0.0, 125.0, 100.0], {"reference", "suppressor"}),
        ([15.0, 0.0, 115.0, 100.0], {"suppressor"}),
    ],
)
def test_finalize_restores_shadow_unless_suppressor_covers_privacy_box(
    suppressor_box: list[float],
    expected_tracks: set[str],
) -> None:
    reference_box = [0.0, 0.0, 100.0, 100.0]
    tracks = [
        {
            "track_id": track_id,
            "accepted": True,
            "accepted_intervals": [[5, 5]],
        }
        for track_id in ("reference", "suppressor")
    ]
    tracking = {
        "observations": [
            {
                "track_id": "suppressor",
                "frame_idx": 5,
                "source": "kalman_optical_flow",
                "box": suppressor_box,
                "motion_box": suppressor_box,
            }
        ],
        "shadows": [
            {
                "track_id": "reference",
                "frame_idx": 5,
                "source": "kalman_optical_flow",
                "box": reference_box,
                "motion_box": reference_box,
                "suppressor_tracks": ["suppressor"],
            }
        ],
    }
    config = {
        "recognition": {"mode": "all"},
        "scan": {
            "global_nms_iou": 0.45,
            "containment_threshold": 0.80,
        },
    }

    output = _finalize(
        tracks,
        {"detections": []},
        tracking,
        [],
        config,
    )

    assert {item["track_id"] for item in output} == expected_tracks


@pytest.mark.parametrize("mode", ["exempt", "blur_only"])
def test_finalize_preserves_each_accepted_track_geometry_in_selective_recognition(
    mode: str,
) -> None:
    reference_box = [0.0, 0.0, 100.0, 100.0]
    suppressor_box = [15.0, 0.0, 115.0, 100.0]
    tracks = [
        {
            "track_id": track_id,
            "accepted": True,
            "accepted_intervals": [[5, 5]],
        }
        for track_id in ("reference", "suppressor")
    ]
    tracking = {
        "observations": [
            {
                "track_id": "suppressor",
                "frame_idx": 5,
                "source": "kalman_optical_flow",
                "box": suppressor_box,
                "motion_box": suppressor_box,
            }
        ],
        "shadows": [
            {
                "track_id": "reference",
                "frame_idx": 5,
                "source": "kalman_optical_flow",
                "box": reference_box,
                "motion_box": reference_box,
                "suppressor_tracks": ["suppressor"],
            }
        ],
    }
    config = {
        "recognition": {"mode": mode},
        "scan": {
            "global_nms_iou": 0.45,
            "containment_threshold": 0.80,
        },
    }

    output = _finalize(
        tracks,
        {"detections": []},
        tracking,
        [],
        config,
    )

    assert {item["track_id"] for item in output} == {"reference", "suppressor"}
