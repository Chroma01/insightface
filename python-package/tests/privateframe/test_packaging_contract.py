from __future__ import annotations

import builtins
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import setuptools
import pytest


def test_desktop_bundle_preserves_privateframe_config_paths(
    monkeypatch,
) -> None:
    package_root = Path(__file__).resolve().parents[2]
    captured: dict[str, Any] = {}

    def analysis(*args, **kwargs):
        captured["analysis_args"] = args
        captured["analysis_kwargs"] = kwargs
        return SimpleNamespace(
            pure=[],
            zipped_data=[],
            scripts=[],
            binaries=[],
            zipfiles=[],
            datas=[],
        )

    monkeypatch.chdir(package_root)
    runpy.run_path(
        str(package_root / "packaging" / "desktop" / "pyinstaller.spec"),
        init_globals={
            "Analysis": analysis,
            "PYZ": lambda *args, **kwargs: object(),
            "EXE": lambda *args, **kwargs: object(),
            "COLLECT": lambda *args, **kwargs: object(),
            "BUNDLE": lambda *args, **kwargs: object(),
        },
        run_name="desktop_packaging_test",
    )

    config_dir = package_root / "insightface" / "app" / "privateframe" / "configs"
    expected_sources = {
        str(config_path) for config_path in config_dir.glob("*.yaml")
    }
    bundled_configs = {
        source
        for source, destination in captured["analysis_kwargs"]["datas"]
        if destination == "insightface/app/privateframe/configs"
    }
    trusted_key_dir = (
        package_root / "insightface" / "model_zoo" / "trusted_keys"
    )
    expected_keys = {str(path) for path in trusted_key_dir.glob("*.pem")}
    bundled_keys = {
        source
        for source, destination in captured["analysis_kwargs"]["datas"]
        if destination == "insightface/model_zoo/trusted_keys"
    }

    assert expected_sources
    assert bundled_configs == expected_sources
    assert expected_keys
    assert bundled_keys == expected_keys


def test_privateframe_extra_and_yaml_are_packaged(
    monkeypatch,
) -> None:
    package_root = Path(__file__).resolve().parents[2]
    captured: dict[str, Any] = {}

    monkeypatch.chdir(package_root)
    monkeypatch.delenv("INSIGHTFACE_WITH_FACE3D", raising=False)
    monkeypatch.setitem(sys.modules, "pypandoc", None)
    monkeypatch.setattr(
        setuptools,
        "setup",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr("platform.system", lambda: "Linux")

    runpy.run_path(str(package_root / "setup.py"), run_name="packaging_test")

    assert captured["package_data"]["insightface.app.privateframe"] == [
        "configs/*.yaml"
    ]
    assert captured["extras_require"]["privateframe"] == [
        "av>=12",
        "PyYAML>=6.0",
    ]
    assert captured["extras_require"]["gui"] == [
        "PySide6-Essentials>=6.5",
        "Pillow",
        "reportlab",
        "scikit-learn",
        "cryptography>=42.0.0",
        "rfc8785>=0.1.4",
        "av>=12",
        "PyYAML>=6.0",
    ]
    assert captured["package_data"]["insightface.model_zoo"] == [
        "trusted_keys/*.pem"
    ]
    assert "PySide6-Essentials>=6.5" not in captured["extras_require"][
        "privateframe"
    ]
    assert captured["install_requires"].count("onnxruntime") == 1
    assert "onnxruntime-gpu" not in captured["install_requires"]
    assert "runtime" not in captured["extras_require"]
    assert "runtime-gpu" not in captured["extras_require"]
    assert "gpu" not in captured["extras_require"]


def test_missing_onnxruntime_error_explains_runtime_choices(
    monkeypatch,
) -> None:
    package_root = Path(__file__).resolve().parents[2]
    original_import = builtins.__import__

    def import_without_onnxruntime(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ModuleNotFoundError(
                "No module named 'onnxruntime'",
                name="onnxruntime",
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_onnxruntime)

    with pytest.raises(ImportError) as exc_info:
        runpy.run_path(
            str(package_root / "insightface" / "__init__.py"),
            run_name="insightface_missing_runtime_test",
        )

    message = str(exc_info.value)
    assert "default InsightFace installation includes `onnxruntime`" in message
    assert "python -m pip install onnxruntime" in message
    assert "install `onnxruntime-gpu` only after uninstalling" in message
    assert "Do not keep both runtime distributions" in message
