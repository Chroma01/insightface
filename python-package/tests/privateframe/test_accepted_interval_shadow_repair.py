from __future__ import annotations

import pytest

from insightface.app.privateframe.streaming import (
    _accepted_interval_coverage,
    _restore_uncovered_accepted_shadows,
)


REFERENCE_BOX = [0.0, 0.0, 100.0, 100.0]


def _track(
    *,
    track_id: str = "accepted",
    accepted: bool = True,
    intervals: list[list[int]] | None = None,
) -> dict[str, object]:
    return {
        "track_id": track_id,
        "accepted": accepted,
        "accepted_intervals": [[5, 5]] if intervals is None else intervals,
    }


def _evidence(*, track_id: str = "accepted") -> dict[str, object]:
    return {
        "track_id": track_id,
        "frame_idx": 5,
        "box": list(REFERENCE_BOX),
        "local_match_count": 1,
        "local_confidence": 0.91,
        "local_review_reason": "scheduled",
        "verifier_face_probability": 0.88,
    }


def _shadow(
    *,
    track_id: str = "accepted",
    frame_idx: int = 5,
    box: list[float] | None = None,
) -> dict[str, object]:
    value = list(REFERENCE_BOX if box is None else box)
    return {
        "track_id": track_id,
        "frame_idx": frame_idx,
        "source": "kalman_optical_flow",
        "box": value,
        "motion_box": value,
        "shadow": True,
        "shadow_reason": 1,
        "suppressor_tracks": ["suppressor"],
    }


def test_restores_shadow_when_final_suppressor_covers_only_79_percent() -> None:
    tracks = [_track()]
    observations = [
        {
            "track_id": "suppressor",
            "frame_idx": 5,
            "box": [0.0, 0.0, 79.0, 100.0],
        }
    ]
    evidence = [_evidence()]

    repaired, repair_count = _restore_uncovered_accepted_shadows(
        tracks,
        observations,
        [_shadow()],
        evidence,
        allow_cross_track_coverage=True,
    )

    assert repair_count == 1
    repair = next(item for item in repaired if item["track_id"] == "accepted")
    assert repair["box"] == REFERENCE_BOX
    assert repair["motion_box"] == REFERENCE_BOX
    assert repair["shadow"] is False
    assert repair["shadow_reason"] == 0
    assert repair["suppressor_tracks"] == []
    assert repair["pre_stabilization_suppressor_tracks"] == ["suppressor"]
    assert repair["accepted_interval_shadow_repair"] is True
    assert repair["local_match_count"] == 1
    assert repair["local_confidence"] == pytest.approx(0.91)
    assert repair["verifier_face_probability"] == pytest.approx(0.88)
    assert _accepted_interval_coverage(
        tracks,
        repaired,
        evidence,
        allow_cross_track_coverage=True,
    ) == (1, 0, 0)


def test_does_not_restore_when_final_suppressor_reaches_80_percent() -> None:
    tracks = [_track()]
    observations = [
        {
            "track_id": "suppressor",
            "frame_idx": 5,
            "box": [0.0, 0.0, 80.0, 100.0],
        }
    ]
    evidence = [_evidence()]

    repaired, repair_count = _restore_uncovered_accepted_shadows(
        tracks,
        observations,
        [_shadow()],
        evidence,
        allow_cross_track_coverage=True,
    )

    assert repair_count == 0
    assert repaired == observations
    assert _accepted_interval_coverage(
        tracks,
        repaired,
        evidence,
        allow_cross_track_coverage=True,
    ) == (1, 0, 1)


def test_does_not_restore_shadow_that_cannot_cover_its_own_evidence() -> None:
    tracks = [_track()]
    evidence = [_evidence()]

    repaired, repair_count = _restore_uncovered_accepted_shadows(
        tracks,
        [],
        [_shadow(box=[0.0, 0.0, 79.0, 100.0])],
        evidence,
        allow_cross_track_coverage=True,
    )

    assert repair_count == 0
    assert repaired == []
    assert _accepted_interval_coverage(
        tracks,
        repaired,
        evidence,
        allow_cross_track_coverage=True,
    ) == (1, 1, 0)


@pytest.mark.parametrize(
    ("tracks", "shadows"),
    [
        ([_track(accepted=False)], [_shadow()]),
        ([_track(intervals=[[6, 6]])], [_shadow()]),
        ([_track()], [_shadow(track_id="unrelated")]),
    ],
    ids=["rejected-track", "outside-interval", "no-matching-shadow"],
)
def test_does_not_restore_rejected_outside_or_unmatched_shadow(
    tracks: list[dict[str, object]],
    shadows: list[dict[str, object]],
) -> None:
    observations: list[dict[str, object]] = []

    repaired, repair_count = _restore_uncovered_accepted_shadows(
        tracks,
        observations,
        shadows,
        [_evidence()],
        allow_cross_track_coverage=True,
    )

    assert repair_count == 0
    assert repaired == observations


def test_cross_track_observation_cannot_replace_shadow_when_disabled() -> None:
    tracks = [_track()]
    observations = [
        {
            "track_id": "suppressor",
            "frame_idx": 5,
            "box": list(REFERENCE_BOX),
        }
    ]
    evidence = [_evidence()]

    repaired, repair_count = _restore_uncovered_accepted_shadows(
        tracks,
        observations,
        [_shadow()],
        evidence,
        allow_cross_track_coverage=False,
    )

    assert repair_count == 1
    assert any(
        item["track_id"] == "accepted"
        and item.get("accepted_interval_shadow_repair") is True
        for item in repaired
    )
    assert _accepted_interval_coverage(
        tracks,
        repaired,
        evidence,
        allow_cross_track_coverage=False,
    ) == (1, 0, 0)
