import hashlib
import json

import pytest


@pytest.fixture
def manifest_package_factory(tmp_path):
    def create(name="raccoon_s"):
        package = tmp_path / name
        package.mkdir()
        files = {
            "detection": ("detector.onnx", b"detector-bytes"),
            "verification": ("verifier.onnx", b"verifier-bytes"),
            "recognition": ("recognizer.onnx", b"recognizer-bytes"),
        }
        manifest = {}
        for task, (filename, content) in files.items():
            (package / filename).write_bytes(content)
            entry = {
                "file": filename,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            if task == "verification":
                entry.update({
                    "expansion": 1.3,
                    "preprocessing": "embedded",
                })
            elif task == "detection":
                entry["preprocessing"] = {"mean": 11.0, "std": 22.0}
            else:
                entry["preprocessing"] = {"mean": 33.0, "std": 44.0}
            manifest[task] = entry
        (package / "manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        return package, manifest

    return create
