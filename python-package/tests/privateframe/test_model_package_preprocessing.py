from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from insightface.app.privateframe import model_catalog, pipeline
from insightface.app.privateframe.base_config import validate_model_package_contracts
from insightface.app.privateframe.model_catalog import materialize_model_package


def _write_package(tmp_path: Path, name: str = "raccoon_s") -> Path:
    package = tmp_path / name
    package.mkdir()
    manifest = {
        "detection": {
            "file": "detector.onnx",
            "sha256": "1" * 64,
            "preprocessing": {"mean": 127.5, "std": 128.0},
        },
        "verification": {
            "file": "verifier.onnx",
            "sha256": "2" * 64,
            "expansion": 1.3,
            "preprocessing": "embedded",
        },
        "recognition": {
            "file": "recognizer.onnx",
            "sha256": "3" * 64,
            "preprocessing": {"mean": 127.5, "std": 127.5},
        },
    }
    (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return package


def _use_package(
    monkeypatch: pytest.MonkeyPatch,
    package: Path,
) -> list[tuple[str, str, str]]:
    calls: list[tuple[str, str, str]] = []

    def ensure_available(sub_dir: str, name: str, *, root: str) -> str:
        calls.append((sub_dir, name, root))
        return str(package)

    monkeypatch.setattr(model_catalog, "ensure_available", ensure_available)
    return calls


@pytest.mark.parametrize("name", ["raccoon_s", "raccoon_l"])
def test_named_package_uses_the_default_insightface_model_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    name: str,
) -> None:
    package = _write_package(tmp_path, name)
    calls = _use_package(monkeypatch, package)
    config: dict[str, Any] = {
        "models": {
            "name": name,
            "detection": {
                "nms_iou_threshold": 0.4,
                "max_detections": 300,
            },
        }
    }

    materialize_model_package(config)

    assert calls == [("models", name, "~/.insightface")]
    assert config["models"]["name"] == name
    assert config["models"]["manifest_path"] == str(package / "manifest.json")


@pytest.mark.parametrize("name", ["", "buffalo_l", "../raccoon_s", 123])
def test_model_package_name_is_strict(
    monkeypatch: pytest.MonkeyPatch,
    name: Any,
) -> None:
    monkeypatch.setattr(
        model_catalog,
        "ensure_available",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid names must fail before model resolution")
        ),
    )
    config = {"models": {"name": name, "detection": {}}}

    with pytest.raises((TypeError, ValueError), match="models.name"):
        materialize_model_package(config)


def test_source_config_rejects_model_package_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_catalog,
        "ensure_available",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid config must fail before model resolution")
        ),
    )
    config = {
        "models": {
            "name": "raccoon_s",
            "package_path": "/models/raccoon_s",
            "detection": {},
        }
    }

    with pytest.raises(ValueError, match="unsupported keys.*package_path"):
        materialize_model_package(config)


def test_effective_config_uses_one_preprocessing_key_for_every_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    package = _write_package(tmp_path)
    calls = _use_package(monkeypatch, package)
    config: dict[str, Any] = {
        "models": {
            "name": "raccoon_s",
            "detection": {
                "nms_iou_threshold": 0.4,
                "max_detections": 300,
            },
        },
        "revalidation": {},
    }

    materialize_model_package(config)
    validate_model_package_contracts(config)

    models = config["models"]
    assert calls == [("models", "raccoon_s", "~/.insightface")]
    assert models["name"] == "raccoon_s"
    assert models["detection"]["preprocessing"] == {
        "mean": 127.5,
        "std": 128.0,
    }
    assert models["verification"]["preprocessing"] == "embedded"
    assert models["recognition"]["preprocessing"] == {
        "mean": 127.5,
        "std": 127.5,
    }
    for task in ("detection", "verification", "recognition"):
        assert "mean" not in models[task]
        assert "std" not in models[task]


def test_effective_contract_does_not_force_verification_to_embedded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    package = _write_package(tmp_path)
    _use_package(monkeypatch, package)
    config: dict[str, Any] = {
        "models": {
            "name": "raccoon_s",
            "detection": {
                "nms_iou_threshold": 0.4,
                "max_detections": 300,
            },
        },
        "revalidation": {},
    }
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["verification"]["preprocessing"] = {"mean": 0.5, "std": 0.25}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    materialize_model_package(config)
    validate_model_package_contracts(config)

    assert config["models"]["verification"]["preprocessing"] == {
        "mean": 0.5,
        "std": 0.25,
    }


def test_model_fingerprints_include_each_selected_preprocessing_contract(
    tmp_path: Path,
) -> None:
    models: dict[str, Any] = {}
    specifications: dict[str, tuple[bytes, Any]] = {
        "detection": (b"detector", {"mean": 127.5, "std": 128.0}),
        "verification": (b"verifier", "embedded"),
        "recognition": (b"recognizer", {"mean": 127.5, "std": 127.5}),
    }
    for task, (content, preprocessing) in specifications.items():
        path = tmp_path / f"{task}.onnx"
        path.write_bytes(content)
        models[task] = {
            "file": path.name,
            "path": str(path),
            "sha256": hashlib.sha256(content).hexdigest(),
            "preprocessing": preprocessing,
        }
    config = {
        "models": models,
        "recognition": {"mode": "blur_only"},
    }

    fingerprints = pipeline._model_fingerprints(config)

    assert {
        task: value["preprocessing"]
        for task, value in fingerprints.items()
    } == {
        task: preprocessing
        for task, (_content, preprocessing) in specifications.items()
    }


def test_effective_contract_rejects_legacy_top_level_mean_std() -> None:
    config = {
        "models": {
            "name": "raccoon_s",
            "manifest_path": "/models/raccoon_s/manifest.json",
            "manifest_sha256": "0" * 64,
            "detection": {
                "file": "detector.onnx",
                "path": "/models/raccoon_s/detector.onnx",
                "sha256": "1" * 64,
                "mean": 127.5,
                "std": 128.0,
                "preprocessing": {"mean": 127.5, "std": 128.0},
                "nms_iou_threshold": 0.4,
                "max_detections": 300,
            },
            "verification": {
                "file": "verifier.onnx",
                "path": "/models/raccoon_s/verifier.onnx",
                "sha256": "2" * 64,
                "expansion": 1.3,
                "preprocessing": "embedded",
            },
            "recognition": {
                "file": "recognizer.onnx",
                "path": "/models/raccoon_s/recognizer.onnx",
                "sha256": "3" * 64,
                "preprocessing": {"mean": 127.5, "std": 127.5},
            },
        },
        "revalidation": {},
    }

    with pytest.raises(ValueError, match="models.detection keys"):
        validate_model_package_contracts(config)
