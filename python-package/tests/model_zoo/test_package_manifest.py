import json

import pytest

from insightface.model_zoo.package_manifest import (
    MODEL_PACKAGE_TASKS,
    load_model_package,
    verify_model_artifact,
)


@pytest.mark.parametrize("name", ["raccoon_s", "raccoon_l"])
def test_supported_packages_expose_exact_tasks(manifest_package_factory, name):
    package, _manifest = manifest_package_factory(name)

    descriptor = load_model_package(package)

    assert descriptor.name == name
    assert tuple(descriptor.tasks) == MODEL_PACKAGE_TASKS


def test_descriptor_parsing_defers_model_binary_verification(
    manifest_package_factory,
):
    package, _manifest = manifest_package_factory()
    descriptor = load_model_package(package)

    # Parsing metadata does not eagerly hash every potentially large ONNX file.
    (package / "verifier.onnx").write_bytes(b"corrupt")

    assert verify_model_artifact(descriptor.task("detection")).name == "detector.onnx"
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        verify_model_artifact(descriptor.task("verification"))


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_manifest_top_level_tasks_are_exact(manifest_package_factory, mutation):
    package, manifest = manifest_package_factory()
    if mutation == "missing":
        manifest.pop("verification")
    else:
        manifest["attribute"] = {}
    (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="keys must be exactly"):
        load_model_package(package)


@pytest.mark.parametrize("task", MODEL_PACKAGE_TASKS)
def test_manifest_task_fields_are_exact(manifest_package_factory, task):
    package, manifest = manifest_package_factory()
    manifest[task]["unexpected"] = True
    (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"{task} keys must be exactly"):
        load_model_package(package)


@pytest.mark.parametrize(
    "task",
    MODEL_PACKAGE_TASKS,
)
@pytest.mark.parametrize(
    "preprocessing",
    [
        "embedded",
        {"mean": 12.5, "std": 64.0},
    ],
)
def test_every_task_accepts_both_preprocessing_modes(
    manifest_package_factory,
    task,
    preprocessing,
):
    package, manifest = manifest_package_factory()
    manifest[task]["preprocessing"] = preprocessing
    (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    descriptor = load_model_package(package).task(task)

    assert descriptor.metadata["preprocessing"] == preprocessing


@pytest.mark.parametrize("task", MODEL_PACKAGE_TASKS)
def test_preprocessing_is_required_for_every_task(
    manifest_package_factory,
    task,
):
    package, manifest = manifest_package_factory()
    manifest[task].pop("preprocessing")
    (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"{task} keys must be exactly"):
        load_model_package(package)


@pytest.mark.parametrize("task", MODEL_PACKAGE_TASKS)
def test_legacy_top_level_mean_and_std_are_rejected(
    manifest_package_factory,
    task,
):
    package, manifest = manifest_package_factory()
    manifest[task].pop("preprocessing")
    manifest[task].update({"mean": 0.0, "std": 1.0})
    (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"{task} keys must be exactly"):
        load_model_package(package)


@pytest.mark.parametrize("task", MODEL_PACKAGE_TASKS)
@pytest.mark.parametrize(
    ("preprocessing", "error_type"),
    [
        (None, TypeError),
        (False, TypeError),
        (1, TypeError),
        ([], TypeError),
        ("external", ValueError),
    ],
)
def test_preprocessing_rejects_wrong_union_types_and_unknown_strings(
    manifest_package_factory,
    task,
    preprocessing,
    error_type,
):
    package, manifest = manifest_package_factory()
    manifest[task]["preprocessing"] = preprocessing
    (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(error_type, match=rf"{task}\.preprocessing"):
        load_model_package(package)


@pytest.mark.parametrize("task", MODEL_PACKAGE_TASKS)
@pytest.mark.parametrize(
    "preprocessing",
    [
        {},
        {"mean": 0.0},
        {"std": 1.0},
        {"mean": 0.0, "std": 1.0, "mode": "external"},
    ],
)
def test_preprocessing_mean_std_mapping_has_exact_fields(
    manifest_package_factory,
    task,
    preprocessing,
):
    package, manifest = manifest_package_factory()
    manifest[task]["preprocessing"] = preprocessing
    (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=rf"{task}\.preprocessing keys must be exactly",
    ):
        load_model_package(package)


@pytest.mark.parametrize("task", MODEL_PACKAGE_TASKS)
@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("mean", True, TypeError),
        ("mean", "0", TypeError),
        ("mean", None, TypeError),
        ("mean", float("nan"), ValueError),
        ("mean", float("inf"), ValueError),
        ("std", False, TypeError),
        ("std", "1", TypeError),
        ("std", None, TypeError),
        ("std", 0.0, ValueError),
        ("std", -1.0, ValueError),
        ("std", float("nan"), ValueError),
        ("std", float("inf"), ValueError),
    ],
)
def test_preprocessing_mean_std_values_are_strictly_validated(
    manifest_package_factory,
    task,
    field,
    value,
    error_type,
):
    package, manifest = manifest_package_factory()
    preprocessing = {"mean": 0.0, "std": 1.0}
    preprocessing[field] = value
    manifest[task]["preprocessing"] = preprocessing
    (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        error_type,
        match=rf"{task}\.preprocessing\.{field}",
    ):
        load_model_package(package)


def test_legacy_model_name_keys_are_strictly_rejected(
    manifest_package_factory,
):
    package, manifest = manifest_package_factory()
    legacy_manifest = {
        "detector": manifest["detection"],
        "verifier": manifest["verification"],
        "recognizer": manifest["recognition"],
    }
    (package / "manifest.json").write_text(
        json.dumps(legacy_manifest),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="keys must be exactly"):
        load_model_package(package)


def test_manifest_model_path_cannot_escape_package(manifest_package_factory, tmp_path):
    package, manifest = manifest_package_factory()
    outside = tmp_path / "outside.onnx"
    outside.write_bytes(b"outside")
    manifest["detection"]["file"] = "../outside.onnx"
    (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="escapes"):
        load_model_package(package)


def test_manifest_tasks_must_reference_distinct_files(manifest_package_factory):
    package, manifest = manifest_package_factory()
    manifest["verification"]["file"] = manifest["detection"]["file"]
    (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="distinct files"):
        load_model_package(package)


def test_verification_rejects_package_root_replaced_by_symlink(
    manifest_package_factory,
    tmp_path,
):
    package, _manifest = manifest_package_factory()
    descriptor = load_model_package(package)
    original = tmp_path / "original-package"
    outside = tmp_path / "outside-package"
    package.rename(original)
    outside.mkdir()
    (outside / "detector.onnx").write_bytes(b"detector-bytes")
    try:
        package.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")

    with pytest.raises(ValueError, match="escapes"):
        verify_model_artifact(descriptor.task("detection"))
