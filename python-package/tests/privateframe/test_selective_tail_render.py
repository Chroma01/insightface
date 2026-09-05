from __future__ import annotations

import pytest

from insightface.app.privateframe.artifact_render import _identity_should_blur


@pytest.mark.parametrize(
    ("mode", "person_id", "target_person", "direction"),
    [
        ("exempt", "alice", "alice", -1),
        ("exempt", "alice", "alice", 1),
        ("blur_only", "bob", "alice", -1),
        ("blur_only", "bob", "alice", 1),
    ],
)
def test_reduced_assurance_interpolate_endpoints_are_blurred_in_selective_modes(
    mode: str,
    person_id: str,
    target_person: str,
    direction: int,
) -> None:
    item = {
        "track_id": "t00001",
        "frame_idx": 100,
        "direction": direction,
        "force_blur": True,
        "reduced_assurance": True,
    }
    policy = {"mode": mode, "target_persons": [target_person]}
    recognition = {
        "enabled": True,
        "gallery_persons": ["alice", "bob"],
        "tracks": {
            "t00001": {
                "status": "CONFIRMED",
                "person_id": person_id,
            }
        },
    }

    should_blur, reason = _identity_should_blur(item, policy, recognition)

    assert should_blur is True
    assert reason == "fail_safe_reduced_assurance_interpolate_endpoint"


def test_legacy_endpoint_reason_remains_a_force_blur_signal() -> None:
    item = {
        "track_id": "t00001",
        "frame_idx": 100,
        "endpoint_repair_reason": "interpolate_unanchored_endpoint",
    }
    policy = {"mode": "exempt", "target_persons": ["alice"]}
    recognition = {
        "enabled": True,
        "gallery_persons": ["alice"],
        "tracks": {
            "t00001": {"status": "CONFIRMED", "person_id": "alice"}
        },
    }

    should_blur, reason = _identity_should_blur(item, policy, recognition)

    assert should_blur is True
    assert reason == "fail_safe_reduced_assurance_interpolate_endpoint"
