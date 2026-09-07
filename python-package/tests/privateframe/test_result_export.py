from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from insightface.app.privateframe import artifact_render, pipeline
from insightface.app.privateframe.artifact_render import RenderTarget
from insightface.app.privateframe.artifacts import write_json


def test_user_observations_keep_only_render_and_editor_fields() -> None:
    values = [
        {
            "frame_idx": 7,
            "track_id": "t00003",
            "box": [10.0, 20.0, 30.0, 40.0],
            "source": "kalman_optical_flow",
            "reduced_assurance": True,
            "endpoint_repair_reason": "interpolate_unanchored_endpoint",
            "motion_box": [10.0, 20.0, 30.0, 40.0],
            "source_aabb": [10.0, 20.0, 30.0, 40.0],
            "raw_output_box": [9.0, 19.0, 29.0, 39.0],
            "detector_landmarks": [[1.0, 2.0]] * 5,
            "verifier_face_probability": 0.99,
        }
    ]

    assert pipeline._export_observations(values) == [
        {
            "frame_idx": 7,
            "track_id": "t00003",
            "box": [10.0, 20.0, 30.0, 40.0],
            "source": "repaired",
            "reduced_assurance": True,
            "identity_unconfirmed": True,
        }
    ]


def test_internal_tracking_sources_are_normalized_for_users() -> None:
    assert pipeline._public_observation_source({"source": "detector"}) == "detector"
    assert (
        pipeline._public_observation_source(
            {
                "source": "kalman_optical_flow",
                "geometry_source": "detector_anchor_interpolation",
            }
        )
        == "interpolated"
    )
    assert (
        pipeline._public_observation_source({"source": "kalman_optical_flow"})
        == "tracked"
    )


def test_user_recognition_keeps_reference_audit_and_render_decisions() -> None:
    references = {
        "accepted_images": 1, "skipped_images": 1,
        "files": [{"file": "photo.jpg", "detected_face_count": 2,
                   "selected_box": [1, 2, 31, 42]}],
        "skipped": [{"file": "empty.png", "reason": "no_face"}],
        "fingerprint": "abc",
    }
    record = {"status": "CONFIRMED", "matched_reference_files": ["photo.jpg"],
              "similarity": 0.83, "reason": "confirmed_reference_set"}
    value = {
        "enabled": True, "references": references,
        "statistics": {"recognizer_calls": 8},
        "tracks": {"t00001": {**record, "frame_indices": [1, 20, 40]}},
    }
    exported = pipeline._export_recognition(value)
    assert exported == {"enabled": True, "references": references,
                        "tracks": {"t00001": record}}
    assert exported["references"] is not references
    assert pipeline._export_recognition(
        {"enabled": False, "reason": "policy_all"}
    ) == {"enabled": False}


def test_development_report_cleanup_refuses_an_unowned_file(tmp_path) -> None:
    result = tmp_path / "clip_privateframe.json"
    report = tmp_path / "clip_privateframe.dev.json"
    report.write_text('{"owned_by":"user"}', encoding="utf-8")

    with pytest.raises(FileExistsError, match="unrecognized development report"):
        pipeline._remove_owned_development_report(report, result)

    assert report.read_text(encoding="utf-8") == '{"owned_by":"user"}'


def test_development_report_cleanup_removes_only_the_paired_generated_file(
    tmp_path,
) -> None:
    result = tmp_path / "clip_privateframe.json"
    report = tmp_path / "clip_privateframe.dev.json"
    report.write_text(
        json.dumps(
            {
                "format": "privateframe-development-report",
                "schema_version": 1,
                "result_file": result.name,
            }
        ),
        encoding="utf-8",
    )

    pipeline._remove_owned_development_report(report, result)

    assert not report.exists()


def test_compact_json_writer_uses_no_formatting_whitespace(tmp_path) -> None:
    path = tmp_path / "result.json"

    write_json(path, {"b": [1, 2], "a": {"value": True}}, indent=None)

    assert path.read_text(encoding="utf-8") == (
        '{"a":{"value":true},"b":[1,2]}\n'
    )


def test_renderer_uses_lightweight_source_geometry_checks(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        artifact_render,
        "probe_video",
        lambda _path: SimpleNamespace(width=320, height=240, fps=30.0),
    )

    with pytest.raises(ValueError, match="source video dimensions do not match"):
        artifact_render.render_artifacts(
            source=tmp_path / "input.mp4",
            targets=[RenderTarget("redacted", tmp_path / "output.mp4")],
            settings={},
            analysis_result={
                "source_video": {
                    "metadata": {
                        "width": 640,
                        "height": 360,
                        "fps": 30.0,
                        "frame_count": 1,
                    }
                },
                "observations": [],
            },
        )
