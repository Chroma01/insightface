"""Minimal manifest-driven model-pack discovery."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from ...model_zoo.package_manifest import (
    DETECTION_TASK,
    EMBEDDED_PREPROCESSING,
    MODEL_PACKAGE_MANIFEST,
    MODEL_PACKAGE_TASKS,
    RECOGNITION_TASK,
    SUPPORTED_MANIFEST_PACKAGES,
    VERIFICATION_TASK,
    load_model_package,
    normalize_preprocessing,
)
from ...utils import ensure_available

SUPPORTED_MODEL_PACKAGES = SUPPORTED_MANIFEST_PACKAGES

DEFAULT_INSIGHTFACE_ROOT = "~/.insightface"

_MODEL_CONFIG_KEYS = {"name", DETECTION_TASK}
_DETECTION_TUNABLE_KEYS = {
    "nms_iou_threshold",
    "max_detections",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    result = value.strip().lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{field} must be a 64-character SHA256")
    return result


def _materialize_package_models(
    config: dict[str, Any],
) -> None:
    models = config.get("models")
    if not isinstance(models, dict):
        raise TypeError("models configuration must be a mapping")
    if "name" not in models:
        raise ValueError("models.name is required")
    extra = set(models) - _MODEL_CONFIG_KEYS
    if extra:
        raise ValueError(f"models contains unsupported keys: {sorted(extra)}")
    package_name = models["name"]
    if not isinstance(package_name, str):
        raise TypeError("models.name must be a string")
    package_name = package_name.strip()
    if package_name not in SUPPORTED_MODEL_PACKAGES:
        raise ValueError("models.name must be raccoon_s or raccoon_l")
    package_path = ensure_available(
        "models",
        package_name,
        root=DEFAULT_INSIGHTFACE_ROOT,
    )
    package = load_model_package(package_path)
    if package.name != package_name:
        raise RuntimeError(
            f"resolved model package name {package.name} does not match {package_name}"
        )
    declared_tasks = {
        task: package.task(task).as_config()
        for task in MODEL_PACKAGE_TASKS
    }

    detection_settings = models.get(DETECTION_TASK, {})
    if not isinstance(detection_settings, Mapping):
        raise TypeError("models.detection must be a mapping")
    detection_extra = set(detection_settings) - _DETECTION_TUNABLE_KEYS
    if detection_extra:
        raise ValueError(
            f"models.detection contains unsupported keys: {sorted(detection_extra)}"
        )
    tunable_detection_settings = deepcopy(dict(detection_settings))
    declared_tasks[DETECTION_TASK].update(tunable_detection_settings)

    models.clear()
    models.update(
        {
            "name": package.name,
            "manifest_path": str(package.manifest_path),
            "manifest_sha256": package.manifest_sha256,
            **declared_tasks,
        }
    )


def materialize_model_package(config: dict[str, Any]) -> None:
    """Resolve one named raccoon package from InsightFace's default model root."""

    _materialize_package_models(config)


def verify_model_file(model: Mapping[str, Any]) -> Path:
    """Require the package model to exist and match its declared SHA256."""

    path = Path(str(model["path"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {path}")
    expected_sha256 = _expected_sha256(model.get("sha256"), field="model.sha256")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"Model SHA256 mismatch for {path}: {actual_sha256} != {expected_sha256}")
    return path


__all__ = [
    "DEFAULT_INSIGHTFACE_ROOT",
    "DETECTION_TASK",
    "EMBEDDED_PREPROCESSING",
    "MODEL_PACKAGE_MANIFEST",
    "MODEL_PACKAGE_TASKS",
    "RECOGNITION_TASK",
    "SUPPORTED_MODEL_PACKAGES",
    "VERIFICATION_TASK",
    "materialize_model_package",
    "normalize_preprocessing",
    "verify_model_file",
]
