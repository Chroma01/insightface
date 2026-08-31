"""Strict, task-aware descriptors for manifest-backed model packages.

Parsing a manifest validates its schema and path boundaries without reading
every model binary. Each selected task is verified immediately before its
inference session is built.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


MODEL_PACKAGE_MANIFEST = "manifest.json"
EMBEDDED_PREPROCESSING = "embedded"
DETECTION_TASK = "detection"
VERIFICATION_TASK = "verification"
RECOGNITION_TASK = "recognition"
MODEL_PACKAGE_TASKS = (
    DETECTION_TASK,
    VERIFICATION_TASK,
    RECOGNITION_TASK,
)
SUPPORTED_MANIFEST_PACKAGES = ("raccoon_s", "raccoon_l")

_TASK_FIELDS = {
    DETECTION_TASK: {"file", "sha256", "preprocessing"},
    VERIFICATION_TASK: {
        "file",
        "sha256",
        "expansion",
        "preprocessing",
    },
    RECOGNITION_TASK: {"file", "sha256", "preprocessing"},
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_mapping(value: Any, expected: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    keys = set(value)
    if keys != expected:
        raise ValueError(
            f"{field} keys must be exactly {sorted(expected)}; "
            f"received {sorted(keys)}"
        )
    return dict(value)


def _expected_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    result = value.strip().lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{field} must be a 64-character SHA256")
    return result


def _finite_number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        requirement = "finite and positive" if positive else "finite"
        raise ValueError(f"{field} must be {requirement}")
    return result


def normalize_preprocessing(value: Any, field: str = "preprocessing") -> Any:
    """Validate and freeze an embedded or scalar mean/std specification."""

    if isinstance(value, str):
        if value != EMBEDDED_PREPROCESSING:
            raise ValueError(
                f'{field} must be "embedded" or a mean/std object'
            )
        return EMBEDDED_PREPROCESSING
    if not isinstance(value, Mapping):
        raise TypeError(
            f'{field} must be "embedded" or a mean/std object'
        )
    values = _require_exact_mapping(value, {"mean", "std"}, field)
    return MappingProxyType(
        {
            "mean": _finite_number(values["mean"], f"{field}.mean"),
            "std": _finite_number(
                values["std"],
                f"{field}.std",
                positive=True,
            ),
        }
    )


def _model_path(package_path: Path, task: str, value: Any) -> tuple[str, Path]:
    if not isinstance(value, str):
        raise TypeError(f"{task}.file must be a string")
    filename = value.strip()
    if not filename:
        raise ValueError(f"{task}.file must not be empty")
    relative = Path(filename)
    if relative.is_absolute():
        raise ValueError(f"{task}.file must be relative")
    if relative.suffix.lower() != ".onnx":
        raise ValueError(f"{task}.file must end in .onnx")
    resolved = (package_path / relative).resolve()
    try:
        resolved.relative_to(package_path)
    except ValueError as error:
        raise ValueError(f"{task}.file escapes the package directory") from error
    return filename, resolved


@dataclass(frozen=True)
class ModelTaskDescriptor:
    """One unverified model task declared by a package manifest."""

    task: str
    file: str
    path: Path
    package_path: Path
    sha256: str
    metadata: Mapping[str, Any]

    def as_config(self) -> dict[str, Any]:
        metadata = dict(self.metadata)
        preprocessing = metadata.get("preprocessing")
        if isinstance(preprocessing, Mapping):
            metadata["preprocessing"] = dict(preprocessing)
        return {
            "file": self.file,
            "path": str(self.path),
            "sha256": self.sha256,
            **metadata,
        }


@dataclass(frozen=True)
class ModelPackageDescriptor:
    """A parsed package whose model binaries have not yet been verified."""

    name: str
    path: Path
    manifest_path: Path
    manifest_sha256: str
    tasks: Mapping[str, ModelTaskDescriptor]

    def task(self, name: str) -> ModelTaskDescriptor:
        try:
            return self.tasks[str(name)]
        except KeyError as error:
            raise ValueError(
                f"model_task must be one of {list(MODEL_PACKAGE_TASKS)}"
            ) from error

    def verify_task(self, name: str) -> ModelTaskDescriptor:
        descriptor = self.task(name)
        verify_model_artifact(descriptor)
        return descriptor


def has_model_package_manifest(path: str | Path) -> bool:
    return (Path(path).expanduser() / MODEL_PACKAGE_MANIFEST).is_file()


def load_model_package(path: str | Path) -> ModelPackageDescriptor:
    """Parse one raccoon package without hashing its three model files."""

    package_path = Path(path).expanduser().resolve()
    if package_path.name not in SUPPORTED_MANIFEST_PACKAGES:
        raise ValueError(
            "manifest model package directory must be raccoon_s or raccoon_l"
        )
    if not package_path.is_dir():
        raise FileNotFoundError(f"model package directory does not exist: {package_path}")

    unresolved_manifest = package_path / MODEL_PACKAGE_MANIFEST
    if not unresolved_manifest.is_file():
        raise FileNotFoundError(
            f"model package manifest does not exist: {unresolved_manifest}"
        )
    manifest_path = unresolved_manifest.resolve()
    try:
        manifest_path.relative_to(package_path)
    except ValueError as error:
        raise ValueError("model package manifest escapes the package directory") from error

    manifest_bytes = manifest_path.read_bytes()
    try:
        raw = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid model package manifest: {manifest_path}") from error
    manifest = _require_exact_mapping(
        raw,
        set(MODEL_PACKAGE_TASKS),
        "model package manifest",
    )

    tasks: dict[str, ModelTaskDescriptor] = {}
    for task in MODEL_PACKAGE_TASKS:
        model = _require_exact_mapping(manifest[task], _TASK_FIELDS[task], task)
        filename, model_path = _model_path(package_path, task, model.pop("file"))
        checksum = _expected_sha256(model.pop("sha256"), f"{task}.sha256")
        model["preprocessing"] = normalize_preprocessing(
            model["preprocessing"],
            f"{task}.preprocessing",
        )
        if task == VERIFICATION_TASK:
            model["expansion"] = _finite_number(
                model["expansion"],
                "verification.expansion",
                positive=True,
            )
        tasks[task] = ModelTaskDescriptor(
            task=task,
            file=filename,
            path=model_path,
            package_path=package_path,
            sha256=checksum,
            metadata=MappingProxyType(model),
        )

    paths = [descriptor.path for descriptor in tasks.values()]
    if len(set(paths)) != len(paths):
        raise ValueError("model package tasks must reference distinct files")

    return ModelPackageDescriptor(
        name=package_path.name,
        path=package_path,
        manifest_path=manifest_path,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        tasks=MappingProxyType(tasks),
    )


def verify_model_artifact(descriptor: ModelTaskDescriptor) -> Path:
    """Verify one selected task immediately before Session construction."""

    path = descriptor.path.resolve()
    try:
        # ``package_path`` is the root frozen at parse time. Re-resolving it
        # here would trust a package directory that an attacker replaced with
        # an external symlink between parsing and Session construction.
        path.relative_to(descriptor.package_path)
    except ValueError as error:
        raise ValueError(f"{descriptor.task}.file escapes its package directory") from error
    if not path.is_file():
        raise FileNotFoundError(f"model file does not exist: {path}")
    actual = _sha256_file(path)
    if actual != descriptor.sha256:
        raise RuntimeError(
            f"model SHA256 mismatch for {path}: {actual} != {descriptor.sha256}"
        )
    return path


# Descriptive aliases keep callers readable without committing the public API
# to one particular noun for the package parser.
load_package_manifest = load_model_package
ModelPackage = ModelPackageDescriptor
ModelArtifact = ModelTaskDescriptor


__all__ = [
    "DETECTION_TASK",
    "EMBEDDED_PREPROCESSING",
    "MODEL_PACKAGE_MANIFEST",
    "MODEL_PACKAGE_TASKS",
    "RECOGNITION_TASK",
    "SUPPORTED_MANIFEST_PACKAGES",
    "VERIFICATION_TASK",
    "ModelArtifact",
    "ModelPackage",
    "ModelPackageDescriptor",
    "ModelTaskDescriptor",
    "has_model_package_manifest",
    "load_model_package",
    "load_package_manifest",
    "normalize_preprocessing",
    "verify_model_artifact",
]
