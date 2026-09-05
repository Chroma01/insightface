import zipfile
from pathlib import Path

from insightface.utils import storage


def test_ensure_available_downloads_from_model_zoo_release(tmp_path, monkeypatch):
    downloads = []

    def download_file(url, path, overwrite):
        downloads.append((url, overwrite))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("detector.onnx", b"model")

    monkeypatch.setattr(storage, "download_file", download_file)

    package = storage.ensure_available(
        "models",
        "raccoon_s",
        root=str(tmp_path),
    )

    assert package == str(tmp_path / "models" / "raccoon_s")
    assert downloads == [
        (
            f"{storage.MODEL_ZOO_RELEASE_DOWNLOAD_URL}raccoon_s.zip",
            True,
        )
    ]
    assert (tmp_path / "models" / "raccoon_s" / "detector.onnx").is_file()


def test_download_onnx_uses_model_zoo_release(tmp_path, monkeypatch):
    downloads = []

    def download_file(url, path, overwrite):
        downloads.append((url, path, overwrite))

    monkeypatch.setattr(storage, "download_file", download_file)

    result = storage.download_onnx(
        "models",
        "inswapper_128.onnx",
        root=str(tmp_path),
    )

    assert result is None
    assert downloads == [
        (
            f"{storage.MODEL_ZOO_RELEASE_DOWNLOAD_URL}inswapper_128.onnx",
            str(tmp_path / "models" / "inswapper_128.onnx"),
            True,
        )
    ]
