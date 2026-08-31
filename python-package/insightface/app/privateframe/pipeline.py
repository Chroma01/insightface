"""Artifact-first application entry points for analysis and video rendering."""

from __future__ import annotations

import gc
import json
import time
import unicodedata
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .artifact_render import RenderTarget, render_artifacts
from .artifacts import (
    git_version,
    sha256_file,
    sha256_json,
    write_json,
    write_jsonl,
)
from .base_config import apply_config_overrides, validate_config_keys
from .config import load_config, validate_redaction, validate_video_output
from .model_catalog import (
    RECOGNITION_TASK,
    VERIFICATION_TASK,
    verify_model_file,
)
from .models import (
    active_face_detector,
    active_face_verifier,
    packaged_face_recognizer,
)
from .streaming import run_stream
from .video import paths_are_distinct

RESULT_FILENAME = "result.privateframe.json"
ARTIFACT_LEVELS = {"final", "audit", "debug"}
_GENERATED_FILENAMES = {
    RESULT_FILENAME,
    "detections.streaming-onnx.jsonl",
    "tracking.streaming-onnx.jsonl",
    "bidirectional-fusion.streaming-onnx.jsonl",
    "revalidation.streaming-onnx.jsonl",
    "observations.streaming-onnx.jsonl",
    "tracks.streaming-onnx.json",
    "effective-config.streaming-onnx.json",
    "manifest.streaming-onnx.json",
    "summary.streaming-onnx.json",
    "render-summary.streaming-onnx.json",
}
_RUNTIME_CACHE_FILENAMES = {
    "encoded-packets.sqlite",
    "encoded-packets.sqlite-wal",
    "encoded-packets.sqlite-shm",
}
_RESERVED_WORK_FILENAMES = _GENERATED_FILENAMES | _RUNTIME_CACHE_FILENAMES


def _raise_if_cancelled(is_cancelled: Callable[[], bool] | None) -> None:
    if is_cancelled is not None and is_cancelled():
        raise InterruptedError("PrivateFrame operation was cancelled")


def _merge(
    target: dict[str, Any],
    update: dict[str, Any],
    path: tuple[str, ...] = (),
) -> None:
    for key, value in update.items():
        child_path = (*path, key)
        if child_path[-1] == "rate_control":
            target[key] = value
        elif isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value, child_path)
        else:
            target[key] = value


def _reject_reserved_work_path(
    path: Path,
    workdir: Path,
    *,
    label: str,
    allowed: set[Path] | None = None,
) -> None:
    if path.parent != workdir or path.name not in _RESERVED_WORK_FILENAMES:
        return
    if allowed is not None and path in allowed:
        return
    raise ValueError(f"{label} path conflicts with a reserved work artifact: {path}")


def _reject_reserved_result_name(path: Path) -> None:
    if path.name in _RESERVED_WORK_FILENAMES and path.name != RESULT_FILENAME:
        raise ValueError(f"result JSON uses a reserved artifact name: {path.name}")


def _model_fingerprints(config: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    detector_id, detector = active_face_detector(config)
    models = {
        detector_id: detector,
        VERIFICATION_TASK: active_face_verifier(config),
    }
    if str(config.get("recognition", {}).get("mode", "all")) != "all":
        recognizer = packaged_face_recognizer(config)
        models[RECOGNITION_TASK] = recognizer
    for model_id, model in sorted(models.items()):
        if "path" not in model:
            raise TypeError(f"active model {model_id} has no path mapping")
        path = verify_model_file(model)
        values[str(model_id)] = {
            "sha256": str(model["sha256"]),
            "bytes": path.stat().st_size,
            "file": str(model["file"]),
            "preprocessing": (
                dict(model["preprocessing"])
                if isinstance(model["preprocessing"], Mapping)
                else str(model["preprocessing"])
            ),
        }
    return values


def _model_package_fingerprint(
    config: dict[str, Any],
) -> dict[str, Any]:
    models = config["models"]
    manifest_path = Path(str(models["manifest_path"]))
    expected_sha256 = str(models["manifest_sha256"])
    actual_sha256 = sha256_file(manifest_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Model package manifest SHA256 mismatch for {manifest_path}: "
            f"{actual_sha256} != {expected_sha256}"
        )
    return {
        "name": str(models["name"]),
        "manifest": {
            "file": manifest_path.name,
            "sha256": actual_sha256,
            "bytes": manifest_path.stat().st_size,
        },
    }


def _result_path(workdir: Path | None, result_path: str | Path | None) -> Path:
    if result_path is not None:
        return Path(result_path).expanduser().resolve()
    if workdir is None:
        raise ValueError("--result or --workdir is required")
    return (workdir / RESULT_FILENAME).resolve()


def _export_observations(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in values:
        exported = dict(item)
        if "box" in exported and "source_aabb" not in exported:
            exported["source_aabb"] = exported["box"]
        output.append(exported)
    return output


def _load_result(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("format") != "privateframe-result":
        raise ValueError(f"unsupported PrivateFrame result format: {path}")
    if int(value.get("schema_version", 0)) != 1:
        raise ValueError(f"unsupported PrivateFrame result schema: {path}")
    if not isinstance(value.get("observations"), list):
        raise TypeError("PrivateFrame result observations must be an array")
    return value


def _render_settings(
    result: dict[str, Any],
    render_config: str | Path | None,
    config_overrides: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    settings = deepcopy(result["render_defaults"])
    if render_config is not None:
        source = Path(render_config).expanduser().resolve()
        override = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(override, dict):
            raise TypeError("render config must be a mapping")
        if "render" in override:
            if set(override) != {"render"} or not isinstance(override["render"], dict):
                raise ValueError("a nested render config must contain only the render mapping")
            override = override["render"]
        unknown = set(override) - set(settings)
        if unknown:
            raise ValueError("unknown render settings: " + ", ".join(sorted(unknown)))
        validate_config_keys(
            {
                "render": {
                    key: value
                    for key, value in override.items()
                    if key != "recognition_policy"
                }
            }
        )
        _merge(settings, override)
    if config_overrides is not None and not isinstance(config_overrides, Mapping):
        raise TypeError("config_overrides must be a dotted-path mapping")
    if config_overrides:
        if any(not isinstance(path, str) for path in config_overrides):
            raise TypeError("configuration override paths must be strings")
        invalid = sorted(
            path
            for path in config_overrides
            if not path.startswith("render.")
        )
        if invalid:
            raise ValueError(
                "render only accepts render.* configuration overrides: "
                + invalid[0]
            )
        # Command-line/Python dotted overrides are the final configuration
        # layer, above both the analyzed defaults and --render-config YAML.
        wrapped = {"render": settings}
        apply_config_overrides(wrapped, config_overrides)
        settings = wrapped["render"]
        validate_config_keys(
            {
                "render": {
                    key: value
                    for key, value in settings.items()
                    if key != "recognition_policy"
                }
            }
        )
    validate_redaction(settings["redaction"])
    validate_video_output(settings["video_output"])
    _validate_recognition_render_policy(settings, result.get("recognition"))
    return settings, sha256_json(settings)


def _validate_recognition_render_policy(settings: dict[str, Any], recognition: Any) -> None:
    policy = settings.get("recognition_policy")
    if not isinstance(policy, dict):
        raise TypeError("render.recognition_policy must be a mapping in the analysis result")
    unknown = set(policy) - {"mode", "target_persons"}
    if unknown:
        raise ValueError("unknown render recognition policy settings: " + ", ".join(sorted(unknown)))
    mode = str(policy.get("mode", "all"))
    if mode not in {"all", "blur_only", "exempt"}:
        raise ValueError("render.recognition_policy.mode must be all, blur_only, or exempt")
    policy["mode"] = mode
    if mode == "all":
        policy["target_persons"] = []
        return
    if not isinstance(recognition, dict) or recognition.get("enabled") is not True:
        raise ValueError("selective rendering requires a result produced with recognition enabled")
    raw_targets = policy.get("target_persons")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("render.recognition_policy.target_persons must be a non-empty list")
    targets: list[str] = []
    for value in raw_targets:
        if not isinstance(value, str) or not value.strip():
            raise TypeError("render recognition target names must be non-empty strings")
        targets.append(unicodedata.normalize("NFC", value.strip()))
    if len(set(targets)) != len(targets):
        raise ValueError("render recognition target names must be unique")
    gallery_people = recognition.get("gallery_persons")
    if gallery_people is None and isinstance(recognition.get("gallery"), dict):
        gallery_people = recognition["gallery"].get("persons")
    if not isinstance(gallery_people, list):
        raise TypeError("recognition result does not declare gallery persons")
    known = {str(value) for value in gallery_people}
    missing = sorted(set(targets) - known)
    if missing:
        raise ValueError(
            "render recognition targets are absent from the analyzed gallery: " + ", ".join(missing)
        )
    policy["target_persons"] = targets


def _remove_runtime_cache(workdir: Path) -> None:
    for filename in _RUNTIME_CACHE_FILENAMES:
        path = workdir / filename
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _remove_unrequested_artifacts(workdir: Path, retained: set[str]) -> None:
    for filename in _GENERATED_FILENAMES - retained:
        path = workdir / filename
        if path.exists():
            path.unlink()


def _analyze_streaming_pipeline_impl(
    *,
    config_path: str | Path,
    input_path: str | Path,
    workdir: str | Path,
    result_path: str | Path | None = None,
    config_overrides: Mapping[str, Any] | None = None,
    config_override_root: str | Path | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    _raise_if_cancelled(is_cancelled)
    source = Path(input_path).expanduser().resolve()
    work = Path(workdir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    destination = _result_path(work, result_path)
    if not paths_are_distinct([source, destination]):
        raise ValueError("input video and result JSON paths must be distinct")
    _reject_reserved_result_name(destination)
    _reject_reserved_work_path(source, work, label="input video")
    _reject_reserved_work_path(
        destination,
        work,
        label="result JSON",
        allowed={work / RESULT_FILENAME},
    )
    config = load_config(
        config_path,
        config_overrides=config_overrides,
        config_override_root=config_override_root,
    )
    _raise_if_cancelled(is_cancelled)
    artifacts_level = str(config.get("output", {}).get("artifacts_level", "final"))
    if artifacts_level not in ARTIFACT_LEVELS:
        raise ValueError("artifacts_level must be final, audit, or debug")
    # Capture the exact files immediately before Session construction. The
    # package manifest and result artifact both pin the bytes about to load.
    model_fingerprints = _model_fingerprints(config)
    model_package = _model_package_fingerprint(config)
    _raise_if_cancelled(is_cancelled)
    result = run_stream(
        source,
        work,
        config,
        progress=progress,
        is_cancelled=is_cancelled,
    )
    _raise_if_cancelled(is_cancelled)
    # ``run_stream`` constructs the detector, recognizer and gallery before
    # entering StreamingEngine.run(). Measure the whole analysis phase here so
    # selective setup cannot be mislabeled as artifact-writing time.
    analysis_seconds = time.perf_counter() - started
    artifact_started = time.perf_counter()
    timestamps = {int(item["frame_idx"]): float(item["time_seconds"]) for item in result["scan"]["frames"]}
    for values in (
        result["scan"]["detections"],
        result["tracking"]["observations"],
        result["review"]["evidence"],
        result["review"]["observations"],
    ):
        for item in values:
            item.setdefault("time_seconds", timestamps[int(item["frame_idx"])])
    observations = _export_observations(result["review"]["observations"])
    scene_cut_candidates = sum(
        bool(item.get("scene_cut_candidate", False)) for item in result["scan"]["frames"]
    )
    scene_cut_flow_confirmed = sum(
        bool(item.get("scene_cut_flow_confirmed", False)) for item in result["scan"]["frames"]
    )
    scene_cut_appearance_confirmed = sum(
        bool(item.get("scene_cut_appearance_confirmed", False)) for item in result["scan"]["frames"]
    )
    scene_cut_confirmed = sum(
        bool(item.get("scene_cut_confirmed", False)) for item in result["scan"]["frames"]
    )
    scene_cut_flash_suppressed = sum(
        bool(item.get("scene_cut_flash_suppressed", False)) for item in result["scan"]["frames"]
    )
    scene_cuts = sum(bool(item.get("scene_cut_from_previous", False)) for item in result["scan"]["frames"])
    repository = Path(__file__).resolve().parents[2]
    source_document = {
        "path": str(source),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
        "metadata": result["scan"]["metadata"],
        "timing_contract": "cfr_frame_index",
        "coordinate_system": "pixel_xyxy",
        "frame_index_origin": 0,
    }
    analysis_statistics = {
        "frame_count": int(result["scan"]["frame_count"]),
        "detections": len(result["scan"]["detections"]),
        "tracks": len(result["tracks"]),
        "accepted_tracks": int(result["review"]["accepted_tracks"]),
        "observations": len(observations),
        "reverse_jobs": int(result["reverse_jobs"]),
        "reverse_frames": int(result["reverse_frames"]),
        "long_gap_reanchors": int(result["long_gap_reanchors"]),
        "discarded_unanchored_tail_frames": int(result["discarded_unanchored_tail_frames"]),
        "endpoint_affine_jobs": int(result["endpoint_affine_jobs"]),
        "endpoint_affine_frames": int(result["endpoint_affine_frames"]),
        "endpoint_affine_published_frames": int(result["endpoint_affine_published_frames"]),
        "interpolate_endpoint_jobs": int(result["interpolate_endpoint_jobs"]),
        "interpolate_endpoint_frames": int(result["interpolate_endpoint_frames"]),
        "interpolate_endpoint_published_frames": int(
            result["interpolate_endpoint_published_frames"]
        ),
        "interpolate_endpoint_seconds": float(
            result["interpolate_endpoint_seconds"]
        ),
        "interpolate_endpoint_reason_counts": dict(
            result["interpolate_endpoint_reason_counts"]
        ),
        "fragment_stitches": int(result["fragment_stitches"]),
        "scene_cut_candidates": scene_cut_candidates,
        "scene_cut_flow_confirmed": scene_cut_flow_confirmed,
        "scene_cut_appearance_confirmed": scene_cut_appearance_confirmed,
        "scene_cut_confirmed": scene_cut_confirmed,
        "scene_cut_flash_suppressed": scene_cut_flash_suppressed,
        "scene_cuts": scene_cuts,
        "detector_analyzed_frames": int(result["detector_sampling"]["analyzed_frames"]),
        "detector_skipped_scan_frames": int(result["detector_sampling"]["skipped_scan_frames"]),
        "local_review_attempts": int(result["local_review_sampling"]["attempts"]),
        "local_review_sampled_out": int(result["local_review_sampling"]["sampled_out"]),
        "local_review_forced_attempts": int(
            result["local_review_sampling"]["forced_attempts"]
        ),
        "verifier_review_calls": int(result["local_review_sampling"]["verifier_calls"]),
        "verifier_review_cache_hits": int(
            result["local_review_sampling"]["verifier_cache_hits"]
        ),
    }
    analysis_statistics.update(
        {
            "bidirectional_gap_jobs": int(result["bidirectional_gap_jobs"]),
            "bidirectional_gap_frames": int(result["bidirectional_gap_frames"]),
            "bidirectional_accepted_frames": int(result["bidirectional_accepted_frames"]),
            "bidirectional_rejected_frames": int(result["bidirectional_rejected_frames"]),
            "bidirectional_review_resolutions": int(result["bidirectional_review_resolutions"]),
            "bidirectional_skipped_jobs": int(result["bidirectional_skipped_jobs"]),
            "bidirectional_association_attempts": int(result["bidirectional_association_attempts"]),
            "bidirectional_association_rescues": int(result["bidirectional_association_rescues"]),
        }
    )
    detector_id, _detector = active_face_detector(config)
    analysis_document = {
        "backend": "onnxruntime-streaming-gop-roi",
        "provider": config["runtime"]["resolved_provider"],
        "active_face_detector": detector_id,
        "effective_config_sha256": sha256_json(config),
        "models": model_fingerprints,
        "git": git_version(repository),
        "artifacts_level": artifacts_level,
        "statistics": analysis_statistics,
        "analysis_seconds": analysis_seconds,
        "detector_sampling": result["detector_sampling"],
        "local_review_sampling": result["local_review_sampling"],
    }
    analysis_document["model_package"] = model_package
    recognition = result.get("recognition", {"enabled": False, "reason": "policy_all"})
    render_defaults = deepcopy(config["render"])
    recognition_settings = config.get("recognition", {"mode": "all", "target_persons": []})
    recognition_mode = str(recognition_settings.get("mode", "all"))
    render_defaults["recognition_policy"] = {
        "mode": recognition_mode,
        # Selective-only settings are deliberately not inspected in all mode.
        # This preserves the zero-recognition contract even for stale/null
        # values left in a user override.
        "target_persons": ([] if recognition_mode == "all" else list(recognition_settings["target_persons"])),
    }
    final_result = {
        "format": "privateframe-result",
        "schema_version": 1,
        "source_video": source_document,
        "analysis": analysis_document,
        "render_defaults": render_defaults,
        "recognition": recognition,
        "observations": observations,
    }
    summary = {
        "backend": "onnxruntime-streaming-gop-roi",
        "phase": "analysis",
        "provider": config["runtime"]["resolved_provider"],
        "input": str(source),
        "workdir": str(work),
        "result": str(destination),
        "artifacts_level": artifacts_level,
        "frame_count": int(result["scan"]["frame_count"]),
        "detections": len(result["scan"]["detections"]),
        "tracks": len(result["tracks"]),
        "accepted_tracks": int(result["review"]["accepted_tracks"]),
        "observations": len(result["review"]["observations"]),
        "recognition": recognition,
        "reverse_jobs": int(result["reverse_jobs"]),
        "reverse_frames": int(result["reverse_frames"]),
        "long_gap_reanchors": int(result["long_gap_reanchors"]),
        "discarded_unanchored_tail_frames": int(result["discarded_unanchored_tail_frames"]),
        "endpoint_affine_jobs": int(result["endpoint_affine_jobs"]),
        "endpoint_affine_frames": int(result["endpoint_affine_frames"]),
        "endpoint_affine_published_frames": int(result["endpoint_affine_published_frames"]),
        "interpolate_endpoint_jobs": int(result["interpolate_endpoint_jobs"]),
        "interpolate_endpoint_frames": int(result["interpolate_endpoint_frames"]),
        "interpolate_endpoint_published_frames": int(
            result["interpolate_endpoint_published_frames"]
        ),
        "interpolate_endpoint_seconds": float(
            result["interpolate_endpoint_seconds"]
        ),
        "interpolate_endpoint_reason_counts": dict(
            result["interpolate_endpoint_reason_counts"]
        ),
        "fragment_stitches": int(result["fragment_stitches"]),
        "scene_cut_candidates": scene_cut_candidates,
        "scene_cut_flow_confirmed": scene_cut_flow_confirmed,
        "scene_cut_appearance_confirmed": scene_cut_appearance_confirmed,
        "scene_cut_confirmed": scene_cut_confirmed,
        "scene_cut_flash_suppressed": scene_cut_flash_suppressed,
        "scene_cuts": scene_cuts,
        "detector_sampling": result["detector_sampling"],
        "local_review_sampling": result["local_review_sampling"],
        "cache": result["cache"],
        "timings": {
            "analysis_seconds": analysis_seconds,
            "artifact_seconds": 0.0,
            "total_seconds": analysis_seconds,
        },
        "profile": {
            "scene_cut_detector": "adaptive",
            "detector_frame_stride": int(config["scan"].get("frame_stride", 1)),
            "max_missed_seconds": float(config["streaming"]["max_missed_seconds"]),
            "max_retroactive_seconds": float(config["streaming"]["max_retroactive_seconds"]),
            "pre_roll_decode_chunk_frames": int(config["streaming"]["pre_roll_decode_chunk_frames"]),
            "recent_frame_cache_frames": int(config["streaming"]["recent_frame_cache_frames"]),
            "roi_size": int(config["tracking"]["kalman_optical_flow"]["roi_size"]),
            "roi_expansion": float(config["tracking"]["kalman_optical_flow"]["roi_expansion"]),
            "bidirectional_fusion_mode": "symmetric_local_soft",
            "measurement_filter": bool(
                config["revalidation"]["geometry_refinement"]["measurement_filter"]["enabled"]
            ),
            "box_stabilization": bool(config["render"]["box_stabilization"]["enabled"]),
        },
    }
    summary.update(
        {
            "bidirectional_gap_jobs": int(result["bidirectional_gap_jobs"]),
            "bidirectional_gap_frames": int(result["bidirectional_gap_frames"]),
            "bidirectional_accepted_frames": int(result["bidirectional_accepted_frames"]),
            "bidirectional_rejected_frames": int(result["bidirectional_rejected_frames"]),
            "bidirectional_review_resolutions": int(result["bidirectional_review_resolutions"]),
            "bidirectional_skipped_jobs": int(result["bidirectional_skipped_jobs"]),
            "bidirectional_association_attempts": int(result["bidirectional_association_attempts"]),
            "bidirectional_association_rescues": int(result["bidirectional_association_rescues"]),
        }
    )
    retained = {destination.name} if destination.parent == work else set()
    if artifacts_level in {"audit", "debug"}:
        write_json(work / "tracks.streaming-onnx.json", result["tracks"])
        write_json(work / "effective-config.streaming-onnx.json", config)
        retained.update(
            {
                "tracks.streaming-onnx.json",
                "effective-config.streaming-onnx.json",
                "summary.streaming-onnx.json",
                "manifest.streaming-onnx.json",
            }
        )
        write_jsonl(
            work / "bidirectional-fusion.streaming-onnx.jsonl",
            result["bidirectional_audits"],
        )
        retained.add("bidirectional-fusion.streaming-onnx.jsonl")
    if artifacts_level == "debug":
        write_jsonl(
            work / "detections.streaming-onnx.jsonl",
            result["scan"]["detections"],
        )
        write_jsonl(
            work / "tracking.streaming-onnx.jsonl",
            result["tracking"]["observations"],
        )
        write_jsonl(
            work / "revalidation.streaming-onnx.jsonl",
            result["review"]["evidence"],
        )
        write_jsonl(work / "observations.streaming-onnx.jsonl", observations)
        retained.update(
            {
                "detections.streaming-onnx.jsonl",
                "tracking.streaming-onnx.jsonl",
                "revalidation.streaming-onnx.jsonl",
                "observations.streaming-onnx.jsonl",
            }
        )

    # This is the only durable production artifact. It contains everything
    # needed to render debug or privacy output again without rerunning models.
    write_json(destination, final_result)
    summary["timings"]["artifact_seconds"] = time.perf_counter() - artifact_started
    summary["timings"]["total_seconds"] = time.perf_counter() - started
    if artifacts_level in {"audit", "debug"}:
        write_json(work / "summary.streaming-onnx.json", summary)
        named_artifacts = {
            filename: work / filename
            for filename in retained
            if filename != "manifest.streaming-onnx.json" and (work / filename).exists()
        }
        if destination.parent != work:
            named_artifacts[destination.name] = destination
        manifest = {
            "schema_version": 1,
            "artifacts_level": artifacts_level,
            "source_video_sha256": source_document["sha256"],
            "artifacts": {
                name: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for name, path in sorted(named_artifacts.items())
            },
        }
        write_json(work / "manifest.streaming-onnx.json", manifest)
    _remove_unrequested_artifacts(work, retained)
    return summary


def analyze_streaming_pipeline(
    *,
    config_path: str | Path,
    input_path: str | Path,
    workdir: str | Path,
    result_path: str | Path | None = None,
    config_overrides: Mapping[str, Any] | None = None,
    config_override_root: str | Path | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Analyze one video while treating the encoded packet DB as runtime-only."""

    work = Path(workdir).expanduser().resolve()
    # A prior process may have terminated before it could run its finalizer.
    # The cache is never a resumable artifact, so every analysis starts fresh.
    _remove_runtime_cache(work)
    try:
        return _analyze_streaming_pipeline_impl(
            config_path=config_path,
            input_path=input_path,
            workdir=work,
            result_path=result_path,
            config_overrides=config_overrides,
            config_override_root=config_override_root,
            progress=progress,
            is_cancelled=is_cancelled,
        )
    finally:
        # ``run_stream`` closes the SQLite connection before returning or
        # raising. Remove only its three fixed runtime files; result, audit,
        # and user-selected output files remain untouched.
        _remove_runtime_cache(work)


def render_streaming_artifacts(
    *,
    input_path: str | Path,
    workdir: str | Path | None = None,
    result_path: str | Path | None = None,
    debug_path: str | Path | None = None,
    redacted_path: str | Path | None = None,
    render_config: str | Path | None = None,
    config_overrides: Mapping[str, Any] | None = None,
    verify_source: bool = True,
    progress: Callable[[int, int, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    _raise_if_cancelled(is_cancelled)
    work = Path(workdir).expanduser().resolve() if workdir is not None else None
    destination = _result_path(work, result_path)
    analysis_result = _load_result(destination)
    render_settings, render_settings_sha256 = _render_settings(
        analysis_result,
        render_config,
        config_overrides,
    )
    targets: list[RenderTarget] = []
    if debug_path is not None:
        targets.append(RenderTarget("debug", Path(debug_path).expanduser().resolve()))
    if redacted_path is not None:
        targets.append(RenderTarget("redacted", Path(redacted_path).expanduser().resolve()))
    render_paths = [Path(input_path).expanduser().resolve(), destination, *(target.path for target in targets)]
    if not paths_are_distinct(render_paths):
        raise ValueError("input, result JSON, and render output paths must be distinct")
    if work is not None:
        _reject_reserved_work_path(render_paths[0], work, label="input video")
        for target in targets:
            _reject_reserved_work_path(target.path, work, label=f"{target.mode} video")
    effective_render_settings = {
        **render_settings,
        **render_settings["video_output"],
    }
    result = render_artifacts(
        source=input_path,
        targets=targets,
        settings=effective_render_settings,
        analysis_result=analysis_result,
        verify_source=verify_source,
        progress=progress,
        is_cancelled=is_cancelled,
    )
    if str(effective_render_settings.get("backend", "pyav")) == "pyav":
        # ``render_artifacts`` has now returned and released its Python frame.
        # Collect once more at the public API boundary so any PyAV cycles that
        # depended on frame locals cannot leak into interpreter shutdown.
        gc.collect()
    result.update(
        {
            "phase": "render",
            "result": str(destination),
            "render_settings_sha256": render_settings_sha256,
            "recognition_policy": deepcopy(render_settings["recognition_policy"]),
        }
    )
    artifacts_level = str(analysis_result["analysis"]["artifacts_level"])
    if artifacts_level in {"audit", "debug"} and work is not None:
        write_json(work / "render-summary.streaming-onnx.json", result)
    return result


def run_streaming_pipeline(
    *,
    config_path: str | Path,
    input_path: str | Path,
    debug_path: str | Path | None,
    workdir: str | Path,
    redacted_path: str | Path | None = None,
    result_path: str | Path | None = None,
    render_config: str | Path | None = None,
    config_overrides: Mapping[str, Any] | None = None,
    config_override_root: str | Path | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    _raise_if_cancelled(is_cancelled)
    source = Path(input_path).expanduser().resolve()
    if debug_path is None and redacted_path is None:
        raise ValueError("process requires a debug and/or redacted video output")
    outputs = [source, _result_path(Path(workdir).expanduser().resolve(), result_path)]
    if debug_path is not None:
        outputs.append(Path(debug_path).expanduser().resolve())
    if redacted_path is not None:
        outputs.append(Path(redacted_path).expanduser().resolve())
    if not paths_are_distinct(outputs):
        raise ValueError("input and output paths must be distinct")
    work = Path(workdir).expanduser().resolve()
    _reject_reserved_result_name(outputs[1])
    _reject_reserved_work_path(source, work, label="input video")
    _reject_reserved_work_path(
        outputs[1],
        work,
        label="result JSON",
        allowed={work / RESULT_FILENAME},
    )
    for output in outputs[2:]:
        _reject_reserved_work_path(output, work, label="render video")

    analysis_total = 0

    def report_analysis(current: int, total: int, _message: str) -> None:
        nonlocal analysis_total
        analysis_total = total
        if progress is not None:
            progress(current, total * 2, "analysis")

    def report_render(current: int, total: int, _message: str) -> None:
        if progress is not None:
            progress(analysis_total + current, analysis_total + total, "render")

    analysis = analyze_streaming_pipeline(
        config_path=config_path,
        input_path=source,
        workdir=workdir,
        result_path=result_path,
        config_overrides=config_overrides,
        config_override_root=config_override_root,
        progress=report_analysis if progress is not None else None,
        is_cancelled=is_cancelled,
    )
    _raise_if_cancelled(is_cancelled)
    rendered = render_streaming_artifacts(
        input_path=source,
        workdir=workdir,
        result_path=result_path,
        debug_path=debug_path,
        redacted_path=redacted_path,
        render_config=render_config,
        config_overrides=(
            {
                path: value
                for path, value in config_overrides.items()
                if path.startswith("render.")
            }
            if config_overrides
            else None
        ),
        progress=report_render if progress is not None else None,
        is_cancelled=is_cancelled,
    )
    return {"analysis": analysis, "render": rendered}


__all__ = [
    "analyze_streaming_pipeline",
    "render_streaming_artifacts",
    "run_streaming_pipeline",
]
