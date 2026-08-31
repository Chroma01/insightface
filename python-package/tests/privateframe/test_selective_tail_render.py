from __future__ import annotations

import pytest

from insightface.app.privateframe.artifact_render import _identity_should_blur


@pytest.mark.parametrize(
    ("mode", "person_id", "target_person"),
    [
        ("exempt", "alice", "alice"),
        ("blur_only", "bob", "alice"),
    ],
)
def test_unreviewed_interpolate_endpoint_is_blurred_in_selective_modes(
    mode: str,
    person_id: str,
    target_person: str,
) -> None:
    item = {
        "track_id": "t00001",
        "frame_idx": 100,
        "endpoint_repair_reason": "interpolate_unanchored_endpoint",
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
    assert reason == "fail_safe_unreviewed_interpolate_endpoint"
