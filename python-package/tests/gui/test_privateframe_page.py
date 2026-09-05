from __future__ import annotations

import os
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _grid_position_for_widget(grid, widget):
    """Return the row/column occupied by a widget or its field wrapper."""

    for index in range(grid.count()):
        item = grid.itemAt(index)
        candidate = item.widget()
        if candidate is widget or (
            candidate is not None and candidate.isAncestorOf(widget)
        ):
            row, column, _row_span, _column_span = grid.getItemPosition(index)
            return row, column
    raise AssertionError(f"{widget.objectName() or type(widget).__name__} is absent")


@pytest.mark.parametrize(
    "movies_location",
    [
        "/Users/alice/Movies",
        r"C:\Users\alice\Videos",
        "/home/alice/Videos",
    ],
)
def test_privateframe_default_output_uses_platform_movies_location(
    monkeypatch,
    movies_location,
):
    pytest.importorskip("PySide6")
    from insightface.gui.pages import privateframe_page

    monkeypatch.setattr(
        privateframe_page.QStandardPaths,
        "writableLocation",
        staticmethod(lambda _location: movies_location),
    )

    assert str(privateframe_page._default_privateframe_output_directory()) == (
        movies_location
    )


@pytest.mark.parametrize(
    ("platform_name", "folder_name"),
    [("darwin", "Movies"), ("win32", "Videos"), ("linux", "Videos")],
)
def test_privateframe_default_output_has_visible_platform_fallback(
    monkeypatch,
    platform_name,
    folder_name,
):
    pytest.importorskip("PySide6")
    from insightface.gui.pages import privateframe_page

    monkeypatch.setattr(
        privateframe_page.QStandardPaths,
        "writableLocation",
        staticmethod(lambda _location: ""),
    )
    monkeypatch.setattr(privateframe_page.sys, "platform", platform_name)

    assert privateframe_page._default_privateframe_output_directory() == (
        privateframe_page.Path.home() / folder_name
    )


def test_privateframe_page_does_not_default_to_hidden_workspace_exports(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.pages import privateframe_page

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    videos = tmp_path / "Videos"
    monkeypatch.setattr(
        privateframe_page,
        "_default_privateframe_output_directory",
        lambda: videos,
    )
    config = AppConfig(workspace_path=str(tmp_path / ".insightface"))

    class Context:
        pass

    context = Context()
    context.config = config
    page = privateframe_page.PrivateFramePage(context)

    assert page.output_dir.text() == str(videos)
    assert page.output_dir.text() != config.export_dir
    assert not videos.exists()
    page.close()


@pytest.mark.parametrize("frame_shape", [(40, 80, 3), (80, 40, 3)])
def test_privateframe_video_preview_uses_first_frame_metadata_and_black_letterbox(
    tmp_path,
    monkeypatch,
    frame_shape,
):
    pytest.importorskip("PySide6")
    from insightface.app.privateframe.video import VideoMetadata
    from insightface.gui.pages import privateframe_page

    source = tmp_path / "holiday.mp4"
    source.write_bytes(b"x" * 1536)
    frame = np.full(frame_shape, (17, 83, 149), dtype=np.uint8)
    calls = {"thumbnail": [], "metadata": []}

    def fake_thumbnail(path):
        calls["thumbnail"].append(Path(path))
        return frame.copy()

    def fake_probe(path):
        calls["metadata"].append(Path(path))
        return VideoMetadata(
            path=str(source),
            width=1920,
            height=1080,
            fps=29.97,
            frame_count=1800,
            duration=65.2,
        )

    video_stream = SimpleNamespace(codec_context=SimpleNamespace(name="h264"))
    audio_stream = SimpleNamespace(codec_context=SimpleNamespace(name="aac"))

    class FakeContainer:
        def __init__(self):
            self.streams = SimpleNamespace(
                video=[video_stream],
                audio=[audio_stream],
            )
            self.closed = False

        def close(self):
            self.closed = True

    container = FakeContainer()
    monkeypatch.setattr(privateframe_page, "read_video_thumbnail", fake_thumbnail)
    monkeypatch.setattr(privateframe_page, "probe_video", fake_probe)
    monkeypatch.setattr(privateframe_page.av, "open", lambda _path: container)

    preview = privateframe_page._read_video_preview(source)

    resolved_source = source.resolve()
    preview_size = privateframe_page._VIDEO_PREVIEW_SIZE
    midpoint = preview_size // 2
    assert calls == {
        "thumbnail": [resolved_source],
        "metadata": [resolved_source],
    }
    assert preview.path == resolved_source
    assert preview.image.shape == (preview_size, preview_size, 3)
    assert np.array_equal(preview.image[midpoint, midpoint], frame[0, 0])
    if frame_shape[1] > frame_shape[0]:
        assert np.all(preview.image[0] == 0)
        assert np.all(preview.image[-1] == 0)
        assert np.array_equal(preview.image[midpoint, 0], frame[0, 0])
        assert np.array_equal(preview.image[midpoint, -1], frame[0, 0])
    else:
        assert np.all(preview.image[:, 0] == 0)
        assert np.all(preview.image[:, -1] == 0)
        assert np.array_equal(preview.image[0, midpoint], frame[0, 0])
        assert np.array_equal(preview.image[-1, midpoint], frame[0, 0])
    assert (preview.width, preview.height) == (1920, 1080)
    assert preview.fps == pytest.approx(29.97)
    assert preview.frame_count == 1800
    assert preview.duration == pytest.approx(65.2)
    assert preview.file_size == 1536
    assert preview.video_codec == "h264"
    assert preview.audio_codec == "aac"
    assert preview.has_audio is True
    assert container.closed


def test_privateframe_page_displays_video_preview_metadata_and_clears_it(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.pages import privateframe_page

    configure_qt_plugin_paths()
    _app = QApplication.instance() or QApplication([])
    config = AppConfig(workspace_path=str(tmp_path), auto_load_model=False)

    class Context:
        pass

    context = Context()
    context.config = config
    source = tmp_path / "holiday.mp4"
    source.write_bytes(b"x" * 1536)
    preview_size = privateframe_page._VIDEO_PREVIEW_SIZE
    square_frame = np.full((preview_size, preview_size, 3), 127, dtype=np.uint8)
    loaded_paths = []

    def fake_preview(path):
        loaded_paths.append(Path(path).resolve())
        return privateframe_page.VideoPreviewData(
            path=source.resolve(),
            image=square_frame.copy(),
            width=1920,
            height=1080,
            fps=29.97,
            frame_count=1800,
            duration=65.2,
            file_size=1536,
            video_codec="h264",
            audio_codec="aac",
            has_audio=True,
        )

    monkeypatch.setattr(privateframe_page, "_read_video_preview", fake_preview)
    page = privateframe_page.PrivateFramePage(context)

    page.video_input.set_path(str(source))

    assert loaded_paths == [source.resolve()]
    assert page.video_input.width() == page.video_input.height() == preview_size
    assert page.video_input.file_label.isHidden()
    assert page.video_input.viewer.image is not None
    assert np.array_equal(page.video_input.viewer.image, square_frame)
    values = {key: label.text() for key, label in page.video_metadata_values.items()}
    assert values["file"] == "holiday.mp4"
    assert "1920" in values["resolution"] and "1080" in values["resolution"]
    assert "01:05" in values["duration"]
    assert "29.97" in values["fps"]
    assert values["frames"].replace(",", "") == "1800"
    assert "h264" in values["video_codec"].lower()
    assert "aac" in values["audio"].lower()
    assert "1.5" in values["size"] and "kib" in values["size"].lower()

    input_layout = page.video_input.parentWidget().layout()
    video_row = input_layout.itemAt(0).layout()
    assert video_row.itemAt(0).widget() is page.video_input
    assert video_row.itemAt(1).widget() is page.video_metadata

    page.video_input.clear()

    assert page.video_input.path() == ""
    assert page.video_input.viewer.image is None
    assert {label.text() for label in page.video_metadata_values.values()} == {"—"}
    page.close()


def test_privateframe_video_preview_ignores_stale_and_cleared_results(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QWidget

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.pages import privateframe_page

    configure_qt_plugin_paths()
    _app = QApplication.instance() or QApplication([])
    config = AppConfig(workspace_path=str(tmp_path), auto_load_model=False)

    class Context:
        pass

    class FakeWorker:
        def __init__(self):
            self.cancelled = False

        def cancel(self):
            self.cancelled = True

    class TaskHost(QWidget):
        def __init__(self):
            super().__init__()
            self.requests = []

        def run_task(
            self,
            title,
            fn,
            on_result,
            *,
            show_dialog=True,
            on_progress=None,
            on_error=None,
            on_finished=None,
        ):
            worker = FakeWorker()
            self.requests.append(
                {
                    "title": title,
                    "fn": fn,
                    "on_result": on_result,
                    "on_error": on_error,
                    "on_finished": on_finished,
                    "worker": worker,
                }
            )
            return worker

    def preview(path, value):
        size = privateframe_page._VIDEO_PREVIEW_SIZE
        return privateframe_page.VideoPreviewData(
            path=path.resolve(),
            image=np.full((size, size, 3), value, dtype=np.uint8),
            width=640,
            height=360,
            fps=25.0,
            frame_count=250,
            duration=10.0,
            file_size=1024,
            video_codec="h264",
            audio_codec=None,
            has_audio=False,
        )

    context = Context()
    context.config = config
    host = TaskHost()
    page = privateframe_page.PrivateFramePage(context, parent=host)
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    third = tmp_path / "third.mp4"
    for source in (first, second, third):
        source.write_bytes(b"video")

    page.video_input.set_path(str(first))
    page.video_input.set_path(str(second))

    assert len(host.requests) == 2
    assert host.requests[0]["worker"].cancelled
    host.requests[1]["on_result"](preview(second, 22))
    host.requests[1]["on_finished"]()
    assert page.video_metadata_values["file"].text() == "second.mp4"
    assert np.all(page.video_input.viewer.image == 22)

    host.requests[0]["on_result"](preview(first, 11))
    host.requests[0]["on_finished"]()
    assert page.video_input.path() == str(second.resolve())
    assert page.video_metadata_values["file"].text() == "second.mp4"
    assert np.all(page.video_input.viewer.image == 22)

    page.video_input.set_path(str(third))
    pending = host.requests[2]
    page.video_input.clear()

    assert pending["worker"].cancelled
    pending["on_result"](preview(third, 33))
    pending["on_finished"]()
    assert page.video_input.path() == ""
    assert page.video_input.viewer.image is None
    assert {label.text() for label in page.video_metadata_values.values()} == {"—"}
    page.close()
    host.close()


def test_privateframe_primary_options_reflow_without_losing_values(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QResizeEvent
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.pages.privateframe_page import PrivateFramePage

    configure_qt_plugin_paths()
    _app = QApplication.instance() or QApplication([])
    config = AppConfig(workspace_path=str(tmp_path), auto_load_model=False)

    class Context:
        pass

    context = Context()
    context.config = config
    page = PrivateFramePage(context)
    page.analysis_mode.setCurrentIndex(page.analysis_mode.findData(15))
    page.redaction_method.setCurrentIndex(page.redaction_method.findData("mosaic"))
    fields = [
        page.model_summary,
        page.analysis_mode,
        page.recognition_policy,
        page.redaction_method,
        page.output_mode,
        page.more_options_button,
    ]

    page._layout_primary_options(two_columns=True)
    page.options_grid.activate()
    wide_positions = [
        _grid_position_for_widget(page.options_grid, field) for field in fields
    ]
    wide_height = page.options_grid.sizeHint().height()

    assert [row for row, _column in wide_positions] == [0, 0, 1, 1, 2, 2]
    assert [column for _row, column in wide_positions] == [1, 3, 1, 3, 1, 3]
    assert page.options_grid.count() == 12

    page._layout_primary_options(two_columns=False)
    page.options_grid.activate()
    narrow_positions = [
        _grid_position_for_widget(page.options_grid, field) for field in fields
    ]
    narrow_height = page.options_grid.sizeHint().height()

    assert [row for row, _column in narrow_positions] == list(range(6))
    assert {column for _row, column in narrow_positions} == {1}
    assert page.options_grid.count() == 12
    assert narrow_height > wide_height
    assert page.analysis_mode.currentData() == 15
    assert page.redaction_method.currentData() == "mosaic"

    page.resizeEvent(QResizeEvent(QSize(1100, 900), QSize(700, 900)))
    assert page._primary_options_two_columns is True
    assert [
        _grid_position_for_widget(page.options_grid, field)[0] for field in fields
    ] == [0, 0, 1, 1, 2, 2]

    page.resizeEvent(QResizeEvent(QSize(700, 900), QSize(1100, 900)))
    assert page._primary_options_two_columns is False
    assert [
        _grid_position_for_widget(page.options_grid, field)[0] for field in fields
    ] == list(range(6))
    assert page.analysis_mode.currentData() == 15
    assert page.redaction_method.currentData() == "mosaic"
    page.close()


def test_build_privateframe_job_maps_gui_options(tmp_path):
    pytest.importorskip("PySide6")
    from insightface.gui.pages.privateframe_page import build_privateframe_job

    source = tmp_path / "input.mov"
    source.write_bytes(b"video")
    output = tmp_path / "output"

    job = build_privateframe_job(
        input_path=source,
        output_dir=output,
        model_package="raccoon_l",
        max_analysis_fps=15,
        redaction_method="mosaic",
        runtime_provider="CUDA",
        model_root=tmp_path / "model-root",
        preserve_aac_audio=False,
    )

    assert job.config_path.name == "base.yaml"
    assert job.result_path == output / "input_privateframe.json"
    assert job.redacted_path == output / "input_privateframe.mp4"
    assert job.workdir == output / ".input_privateframe_work"
    assert job.output_mode == "json_and_video"
    assert job.config_overrides["models.name"] == "raccoon_l"
    assert job.model_name == "raccoon_l"
    assert job.model_root == (tmp_path / "model-root").resolve()
    assert job.provider_choice == "CUDA"
    assert job.config_overrides["models.root"] == str(
        (tmp_path / "model-root").resolve()
    )
    assert job.config_overrides["scan.max_analysis_fps"] == 15
    assert job.config_overrides["runtime.provider"] == "CUDAExecutionProvider"
    assert job.config_overrides["render.redaction.method"] == "mosaic"
    assert job.config_overrides["render.redaction.box_scale"] == 1.0
    assert job.config_overrides["render.video_output.preset"] == "medium"
    assert job.config_overrides["render.video_output.rate_control.mode"] == "crf"
    assert job.config_overrides["render.video_output.rate_control.quality"] == 18
    assert job.config_overrides["render.video_output.audio.redacted"] == "none"
    assert job.config_overrides["recognition.mode"] == "all"
    assert "tracking.between_scan_frames" not in job.config_overrides
    assert "recognition.gallery_dir" not in job.config_overrides


def test_build_privateframe_job_maps_advanced_and_selective_options(tmp_path):
    pytest.importorskip("PySide6")
    from insightface.gui.pages.privateframe_page import build_privateframe_job

    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    gallery = tmp_path / "gallery"
    (gallery / "Alice").mkdir(parents=True)
    (gallery / "Bob").mkdir()
    (gallery / ".hidden-person").mkdir()

    job = build_privateframe_job(
        input_path=source,
        output_dir=tmp_path / "output",
        model_package="raccoon_s",
        max_analysis_fps=15,
        redaction_method="gaussian",
        runtime_provider="CPU",
        between_scan_frames="visual",
        box_scale=1.30,
        video_preset="slow",
        video_crf=28,
        recognition_mode="exempt",
        recognition_gallery_dir=gallery,
        recognition_target_persons=["Bob", "Alice"],
        recognition_profile="accurate",
    )

    assert job.config_overrides["scan.max_analysis_fps"] == 15
    assert job.config_overrides["tracking.between_scan_frames"] == "visual"
    assert job.config_overrides["render.redaction.box_scale"] == 1.30
    assert job.config_overrides["render.video_output.preset"] == "slow"
    assert job.config_overrides["render.video_output.rate_control.mode"] == "crf"
    assert job.config_overrides["render.video_output.rate_control.quality"] == 28
    assert job.config_overrides["recognition.mode"] == "exempt"
    assert job.config_overrides["recognition.gallery_dir"] == str(gallery.resolve())
    assert job.config_overrides["recognition.target_persons"] == ["Bob", "Alice"]
    assert job.config_overrides["recognition.profile"] == "accurate"


def test_advanced_gui_job_overrides_load_as_valid_config(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from insightface.app.privateframe import base_config
    from insightface.app.privateframe.config import load_config
    from insightface.gui.pages.privateframe_page import build_privateframe_job

    monkeypatch.setattr(base_config, "materialize_model_package", lambda _config: None)
    monkeypatch.setattr(
        base_config,
        "validate_model_package_contracts",
        lambda _config: None,
    )
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    gallery = tmp_path / "gallery"
    (gallery / "Alice").mkdir(parents=True)
    job = build_privateframe_job(
        input_path=source,
        output_dir=tmp_path / "output",
        model_package="raccoon_l",
        max_analysis_fps=15,
        redaction_method="mosaic",
        runtime_provider="CPU",
        between_scan_frames="visual",
        box_scale=1.15,
        video_preset="veryfast",
        video_crf=23,
        recognition_mode="exempt",
        recognition_gallery_dir=gallery,
        recognition_target_persons=["Alice"],
        recognition_profile="fast",
    )

    config = load_config(
        job.config_path,
        config_overrides=job.config_overrides,
        config_override_root=job.config_path.parent,
    )

    assert config["scan"]["max_analysis_fps"] == 15.0
    assert config["models"]["root"] == str(Path("~/.insightface").expanduser())
    assert config["tracking"]["between_scan_frames"] == "visual"
    assert config["render"]["redaction"]["box_scale"] == 1.15
    assert config["render"]["video_output"]["preset"] == "veryfast"
    assert config["render"]["video_output"]["rate_control"] == {
        "mode": "crf",
        "quality": 23,
    }
    assert config["recognition"]["target_persons"] == ["Alice"]


def test_all_recognition_policy_never_reads_gallery_arguments(tmp_path):
    pytest.importorskip("PySide6")
    from insightface.gui.pages.privateframe_page import build_privateframe_job

    class UnreadableGallery:
        def __str__(self):
            raise AssertionError("all policy must not inspect gallery")

        def __iter__(self):
            raise AssertionError("all policy must not inspect targets")

    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    poison = UnreadableGallery()

    job = build_privateframe_job(
        input_path=source,
        output_dir=tmp_path / "output",
        model_package="raccoon_s",
        max_analysis_fps=30,
        redaction_method="gaussian",
        runtime_provider="auto",
        recognition_mode="all",
        recognition_gallery_dir=poison,
        recognition_target_persons=poison,
    )

    assert job.config_overrides["recognition.mode"] == "all"
    assert "recognition.gallery_dir" not in job.config_overrides
    assert "recognition.target_persons" not in job.config_overrides


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("max_analysis_fps", 0),
        ("max_analysis_fps", -1),
        ("max_analysis_fps", 29.97),
        ("max_analysis_fps", "15"),
        ("max_analysis_fps", True),
        ("max_analysis_fps", float("nan")),
        ("max_analysis_fps", float("inf")),
        ("between_scan_frames", "hybrid"),
        ("box_scale", 0.9),
        ("video_preset", "fast"),
        ("video_crf", 19),
        ("recognition_mode", "unknown"),
        ("recognition_profile", "maximum"),
    ],
)
def test_build_privateframe_job_rejects_unsupported_advanced_values(
    tmp_path,
    option,
    value,
):
    pytest.importorskip("PySide6")
    from insightface.gui.pages.privateframe_page import build_privateframe_job

    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    options = {
        "input_path": source,
        "output_dir": tmp_path / "output",
        "model_package": "raccoon_s",
        "max_analysis_fps": 30,
        "redaction_method": "gaussian",
        "runtime_provider": "auto",
    }
    options[option] = value

    with pytest.raises(ValueError):
        build_privateframe_job(**options)


def test_run_privateframe_job_calls_python_pipeline_directly(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from insightface.gui.pages import privateframe_page

    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    job = privateframe_page.build_privateframe_job(
        input_path=source,
        output_dir=tmp_path / "output",
        model_package="raccoon_s",
        max_analysis_fps=15,
        redaction_method="gaussian",
        runtime_provider="Auto",
        preserve_aac_audio=True,
    )
    captured = {}

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return {"analysis": {}, "render": {}}

    def progress(current, total, message):
        return None

    def is_cancelled():
        return False

    monkeypatch.setattr(privateframe_page, "run_streaming_pipeline", fake_pipeline)
    monkeypatch.setattr(privateframe_page, "_source_audio_codec", lambda _path: "aac")

    result = privateframe_page.run_privateframe_job(
        job,
        progress=progress,
        is_cancelled=is_cancelled,
    )

    assert result == {
        "analysis": {},
        "render": {},
        "gui": {
            "output_mode": "json_and_video",
            "source_audio_codec": "aac",
            "audio_output_mode": "aac",
        },
    }
    assert captured["input_path"] == job.input_path
    assert captured["redacted_path"] == job.redacted_path
    assert captured["result_path"] == job.result_path
    assert captured["debug_path"] is None
    assert captured["progress"] is progress
    assert captured["is_cancelled"] is is_cancelled
    assert captured["config_overrides"]["scan.max_analysis_fps"] == 15
    assert captured["config_overrides"]["runtime.provider"] == "auto"


@pytest.mark.parametrize(
    ("render_video", "expected"),
    [
        (True, 14.75),
        (False, 10.25),
    ],
)
def test_privateframe_pipeline_total_includes_artifact_and_optional_render(
    render_video,
    expected,
):
    pytest.importorskip("PySide6")
    from insightface.gui.pages.privateframe_page import _pipeline_total_seconds

    analysis = {
        "timings": {
            "analysis_seconds": 10.0,
            "artifact_seconds": 0.25,
            "total_seconds": 10.25,
        }
    }
    render = {"seconds": 4.5}

    assert (
        _pipeline_total_seconds(
            analysis,
            render,
            render_video=render_video,
        )
        == expected
    )


def test_non_aac_audio_is_omitted_before_pipeline(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from insightface.gui.pages import privateframe_page

    source = tmp_path / "input.webm"
    source.write_bytes(b"video")
    job = privateframe_page.build_privateframe_job(
        input_path=source,
        output_dir=tmp_path / "output",
        model_package="raccoon_s",
        max_analysis_fps=30,
        redaction_method="gaussian",
        runtime_provider="CPU",
        preserve_aac_audio=True,
    )
    captured = {}
    monkeypatch.setattr(privateframe_page, "_source_audio_codec", lambda _path: "opus")
    monkeypatch.setattr(
        privateframe_page,
        "run_streaming_pipeline",
        lambda **kwargs: captured.update(kwargs) or {"analysis": {}, "render": {}},
    )

    result = privateframe_page.run_privateframe_job(job)

    assert captured["config_overrides"]["render.video_output.audio.redacted"] == "none"
    assert captured["config_overrides"]["runtime.provider"] == "CPUExecutionProvider"
    assert result["gui"] == {
        "output_mode": "json_and_video",
        "source_audio_codec": "opus",
        "audio_output_mode": "none",
    }


def test_json_only_job_analyzes_without_probing_audio_or_rendering(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("PySide6")
    from insightface.gui.pages import privateframe_page

    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    job = privateframe_page.build_privateframe_job(
        input_path=source,
        output_dir=tmp_path / "results",
        model_package="raccoon_s",
        max_analysis_fps=15,
        redaction_method="mosaic",
        runtime_provider="CoreML",
        preserve_aac_audio=True,
        output_mode="json_only",
    )
    captured = {}
    expected_analysis = {"frame_count": 12, "accepted_tracks": 2}

    monkeypatch.setattr(
        privateframe_page,
        "_source_audio_codec",
        lambda _path: pytest.fail("JSON-only mode must not inspect audio"),
    )
    monkeypatch.setattr(
        privateframe_page,
        "run_streaming_pipeline",
        lambda **_kwargs: pytest.fail("JSON-only mode must not render video"),
    )
    monkeypatch.setattr(
        privateframe_page,
        "analyze_streaming_pipeline",
        lambda **kwargs: captured.update(kwargs) or expected_analysis,
    )

    result = privateframe_page.run_privateframe_job(job)

    assert job.result_path == tmp_path / "results" / "clip_privateframe.json"
    assert job.redacted_path is None
    assert job.config_overrides["render.redaction.method"] == "mosaic"
    assert job.config_overrides["render.redaction.box_scale"] == 1.0
    assert job.config_overrides["render.video_output.preset"] == "medium"
    assert job.config_overrides["render.video_output.rate_control.mode"] == "crf"
    assert job.config_overrides["render.video_output.rate_control.quality"] == 18
    assert job.config_overrides["render.video_output.audio.redacted"] == "none"
    assert captured["result_path"] == job.result_path
    assert captured["workdir"] == job.workdir
    assert result == {
        "analysis": expected_analysis,
        "render": {},
        "gui": {
            "output_mode": "json_only",
            "source_audio_codec": None,
            "audio_output_mode": "none",
        },
    }


def test_privateframe_page_has_non_blocking_controls(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.pages.privateframe_page import PrivateFramePage

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    config = AppConfig(workspace_path=str(tmp_path), auto_load_model=False)

    class Context:
        pass

    context = Context()
    context.config = config
    page = PrivateFramePage(context)

    assert page.model_label.text() == "raccoon_s"
    assert page.model_root_label.text() == str(
        Path(config.model_root).expanduser().resolve()
    )
    assert not hasattr(page, "model_package")
    assert page.analysis_mode.count() == 2
    assert page.analysis_mode.currentData() == 30
    assert page.analysis_fps_label.text() == "Target analysis FPS"
    assert page.analysis_mode.itemText(0) == "Normal (target 30 analysis FPS)"
    assert page.analysis_mode.itemText(1) == "Fast (target 15 analysis FPS)"
    assert page.analysis_mode.itemData(1) == 15
    assert "does not change the output video's frame rate" in (
        page.analysis_mode.toolTip()
    )
    assert page.redaction_method.count() == 2
    assert page.output_mode.count() == 2
    assert page.output_mode.currentData() == "json_and_video"
    assert page.recognition_policy.count() == 3
    assert page.recognition_policy.currentData() == "all"
    assert page.more_options_button.objectName() == "privateFrameMoreOptionsButton"
    assert not page.more_options_dialog.isModal()
    assert page.between_scan_frames.currentData() == "auto"
    assert page.box_scale.currentData() == 1.0
    assert page.video_preset.currentData() == "medium"
    assert page.video_crf.currentData() == 18
    assert page.recognition_profile.currentData() == "balanced"
    assert not page.gallery_dir.isEnabled()
    assert not page.target_persons.isEnabled()
    assert page.gallery_row.isHidden()
    assert page.selective_privacy_note.isHidden()
    assert page.progress_bar.value() == 0
    assert page.start_button.isEnabled()
    assert not page.cancel_button.isEnabled()
    assert page.summary.isReadOnly()
    page._set_summary_text("\n".join(f"line {index}" for index in range(100)))
    assert page.summary.textCursor().atEnd()
    assert (
        page.summary.verticalScrollBar().value()
        == page.summary.verticalScrollBar().maximum()
    )
    page.more_options_button.click()
    assert page.more_options_dialog.isVisible()
    page.video_preset.setCurrentIndex(page.video_preset.findData("slow"))
    page.more_options_dialog.close()
    page.more_options_button.click()
    assert page.video_preset.currentData() == "slow"
    page.more_options_dialog.close()
    page.close()


def test_privateframe_non_raccoon_blocks_workspace_with_top_message(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.pages import privateframe_page

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    config = AppConfig(
        workspace_path=str(tmp_path / "workspace"),
        model_name="buffalo_l",
        model_root=str(tmp_path / "model-root"),
        auto_load_model=False,
    )

    class Context:
        pass

    context = Context()
    context.config = config
    page = privateframe_page.PrivateFramePage(context)

    assert page.content.itemAt(0).widget() is page.model_requirement_banner
    assert page.content.itemAt(1).widget() is page.operation_panel
    assert page.model_requirement_banner.isHidden() is False
    assert page.model_requirement_banner.parentWidget() is page
    assert "raccoon_s or raccoon_l" in page.model_requirement_message.text()
    assert page.model_requirement_current_model_value.text() == "buffalo_l"
    assert page.model_requirement_open_models_button.isEnabled()
    assert page.operation_panel.isEnabled() is False
    assert page.operation_opacity.opacity() == pytest.approx(
        privateframe_page._UNAVAILABLE_CONTENT_OPACITY
    )
    for widget in (
        page.video_input,
        page.output_dir,
        page.analysis_mode,
        page.recognition_policy,
        page.redaction_method,
        page.output_mode,
        page.more_options_button,
        page.start_button,
    ):
        assert widget.isEnabled() is False
    assert page.more_options_dialog.isEnabled() is False

    config.model_name = "raccoon_s"
    page.refresh()

    assert page.model_requirement_banner.isHidden()
    assert page.operation_panel.isEnabled()
    assert page.operation_opacity.opacity() == pytest.approx(1.0)
    assert page.video_input.isEnabled()
    assert page.output_dir.isEnabled()
    assert page.analysis_mode.isEnabled()
    assert page.start_button.isEnabled()
    assert page.more_options_dialog.isEnabled()

    page.more_options_dialog.show()
    assert page.more_options_dialog.isVisible()
    config.model_name = "buffalo_m"
    page.refresh()

    assert page.more_options_dialog.isHidden()
    assert page.operation_panel.isEnabled() is False
    page.close()


def test_privateframe_invalid_raccoon_keeps_workspace_visible(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.pages.privateframe_page import PrivateFramePage

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    model_root = tmp_path / "model-root"
    (model_root / "models" / "raccoon_s").mkdir(parents=True)
    config = AppConfig(
        workspace_path=str(tmp_path / "workspace"),
        model_name="raccoon_s",
        model_root=str(model_root),
        auto_load_model=False,
    )

    class Context:
        pass

    context = Context()
    context.config = config
    page = PrivateFramePage(context)

    assert page.model_requirement_banner.isHidden()
    assert page.operation_panel.isEnabled()
    assert page.start_button.isEnabled() is False
    assert "manifest is missing" in page.model_status_label.text()
    page.close()


def test_privateframe_model_requirement_message_survives_language_switch(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.core.i18n import apply_translations
    from insightface.gui.pages.privateframe_page import PrivateFramePage

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    config = AppConfig(
        workspace_path=str(tmp_path / "workspace"),
        model_name="buffalo_l",
        model_root=str(tmp_path / "model-root"),
        auto_load_model=False,
        ui_language="zh",
    )

    class Context:
        pass

    context = Context()
    context.config = config
    page = PrivateFramePage(context)
    apply_translations(page, "zh")

    assert page.model_requirement_title.text() == "PrivateFrame 需要 Raccoon 模型。"
    assert "仅支持 raccoon_s 或 raccoon_l" in (
        page.model_requirement_message.text()
    )
    assert page.model_requirement_current_model_caption.text() == "当前全局模型"

    config.ui_language = "en"
    apply_translations(page, "en")

    assert page.model_requirement_title.text() == (
        "PrivateFrame requires a Raccoon model."
    )
    assert page.model_requirement_message.text().startswith(
        "PrivateFrame supports only raccoon_s or raccoon_l."
    )
    assert page.model_requirement_current_model_caption.text() == (
        "Current global model"
    )
    page.close()


@pytest.mark.parametrize("language", ["zh", "ja", "ko", "es", "fr", "de", "pt", "ru"])
def test_privateframe_core_controls_are_localized_in_every_gui_language(
    tmp_path,
    language,
):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.core.i18n import apply_translations, tr
    from insightface.gui.pages.privateframe_page import PrivateFramePage

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    config = AppConfig(
        workspace_path=str(tmp_path / language),
        model_root=str(tmp_path / language / "model-root"),
        auto_load_model=False,
        ui_language=language,
    )
    context = SimpleNamespace(config=config)
    page = PrivateFramePage(context)
    apply_translations(page, language)

    assert page.analysis_fps_label.text() == tr("Target analysis FPS", language)
    assert page.analysis_mode.itemText(0) == tr(
        "Normal (target 30 analysis FPS)", language
    )
    assert page.analysis_mode.itemText(1) == tr(
        "Fast (target 15 analysis FPS)", language
    )
    assert page.analysis_mode.currentData() == 30
    assert page.analysis_mode.itemData(1) == 15
    assert page.analysis_mode.toolTip() == tr(
        "Sets the target sampling rate for face analysis. Extra safety scans may occur. It does not change the output video's frame rate.",
        language,
    )
    assert page.video_input.dialog_filter == (
        f"{tr('Videos', language)} (*.mp4 *.mov *.m4v *.mkv *.avi *.webm);;"
        f"{tr('All Files', language)} (*)"
    )
    assert page.more_options_dialog.windowTitle() == tr(
        "PrivateFrame More Options", language
    )
    assert page.redaction_method.itemText(0) == tr("Gaussian blur", language)
    assert page.recognition_policy.itemText(0) == tr(
        "Blur every face (no identity recognition)", language
    )
    assert tr("Target analysis FPS", language) != "Target analysis FPS"
    page.close()


def test_privateframe_dynamic_content_retranslates_without_losing_values(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.core.i18n import apply_translations
    from insightface.gui.pages.privateframe_page import PrivateFramePage

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    config = AppConfig(
        workspace_path=str(tmp_path / "workspace"),
        model_root=str(tmp_path / "model-root"),
        auto_load_model=False,
        ui_language="en",
    )
    page = PrivateFramePage(SimpleNamespace(config=config))
    source = tmp_path / "holiday.mp4"
    source.write_bytes(b"video")
    output_dir = tmp_path / "output"
    gallery = tmp_path / "gallery"
    (gallery / "Alice").mkdir(parents=True)
    page.video_input._path = str(source)
    page._video_preview_state = "failed"
    page._video_preview_path = source
    page.output_dir.setText(str(output_dir))
    page.recognition_policy.setCurrentIndex(
        page.recognition_policy.findData("exempt")
    )
    page.gallery_dir.setText(str(gallery))
    page._refresh_gallery_people()
    page._last_job = SimpleNamespace(
        output_mode="json_and_video",
        result_path=output_dir / "holiday_privateframe.json",
        redacted_path=output_dir / "holiday_privateframe.mp4",
    )
    result = {
        "analysis": {
            "frame_count": 10,
            "accepted_tracks": 2,
            "timings": {"analysis_seconds": 1.25, "total_seconds": 1.5},
        },
        "render": {"frame_count": 10, "seconds": 0.5},
        "gui": {"source_audio_codec": "aac", "audio_output_mode": "aac"},
    }
    page._summary_state = "completed"
    page._summary_result = result
    page._set_stage(
        "Analyzing video frames: {current}/{total}",
        current=3,
        total=10,
    )
    page._set_progress_format("Completed")

    apply_translations(page, "zh")

    assert page.stage_label.text() == "正在分析视频帧：3/10"
    assert page.progress_bar.format() == "已完成"
    assert "分析结果 JSON" in page.output_preview.text()
    assert "无法生成预览" in page.video_input.placeholder.text()
    assert "找到 1 位人员" in page.gallery_status.text()
    assert "已接受轨迹：2" in page.summary.toPlainText()
    assert "总耗时：2.00 秒" in page.summary.toPlainText()

    apply_translations(page, "en")

    assert page.stage_label.text() == "Analyzing video frames: 3/10"
    assert page.progress_bar.format() == "Completed"
    assert "Analysis JSON" in page.output_preview.text()
    assert "Preview unavailable" in page.video_input.placeholder.text()
    assert "Found 1 people" in page.gallery_status.text()
    assert "Accepted tracks: 2" in page.summary.toPlainText()
    assert "Total seconds: 2.00" in page.summary.toPlainText()
    page.close()


def test_privateframe_dynamic_error_and_provider_tooltip_retranslate(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.core.i18n import apply_translations, tr
    from insightface.gui.pages.privateframe_page import PrivateFramePage

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    config = AppConfig(
        workspace_path=str(tmp_path / "workspace"),
        auto_load_model=False,
        provider="CPU",
        ui_language="en",
    )
    page = PrivateFramePage(SimpleNamespace(config=config))
    page.show_error = lambda _message: None
    page._processing_error("The output directory does not exist.")
    assert page.summary.toPlainText() == "The output directory does not exist."

    config.ui_language = "zh"
    apply_translations(page, "zh")

    assert page.summary.toPlainText() == tr(
        "The output directory does not exist.",
        "zh",
    )
    assert "Configured selection:" not in page.provider_label.toolTip()
    assert "CPU" in page.provider_label.toolTip()

    config.ui_language = "en"
    apply_translations(page, "en")
    assert page.summary.toPlainText() == "The output directory does not exist."
    assert "Configured selection: CPU" in page.provider_label.toolTip()
    page.close()


def test_privateframe_invalid_model_reason_is_localized(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.core.i18n import apply_translations
    from insightface.gui.pages.privateframe_page import PrivateFramePage

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    model_root = tmp_path / "model-root"
    model_root.write_text("not a directory", encoding="utf-8")
    config = AppConfig(
        workspace_path=str(tmp_path / "workspace"),
        model_root=str(model_root),
        model_name="raccoon_s",
        auto_load_model=False,
        ui_language="zh",
    )
    page = PrivateFramePage(SimpleNamespace(config=config))
    apply_translations(page, "zh")

    assert "模型根目录不是目录" in page.model_status_label.text()
    assert str(model_root) in page.model_status_label.text()
    page.close()


def test_privateframe_json_only_controls_preview_and_progress(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.pages.privateframe_page import PrivateFramePage

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    config = AppConfig(workspace_path=str(tmp_path), auto_load_model=False)

    class Context:
        pass

    context = Context()
    context.config = config
    page = PrivateFramePage(context)
    source = tmp_path / "holiday.mov"
    source.write_bytes(b"video")
    page.video_input.set_path(str(source))
    page.output_dir.setText(str(tmp_path / "output"))

    assert "holiday_privateframe.json" in page.output_preview.text()
    assert "holiday_privateframe.mp4" in page.output_preview.text()
    page.output_mode.setCurrentIndex(page.output_mode.findData("json_only"))

    assert not page.preserve_audio.isEnabled()
    assert page.video_preset.isEnabled()
    assert page.video_crf.isEnabled()
    assert page.box_scale.isEnabled()
    assert "holiday_privateframe.json" in page.output_preview.text()
    assert "holiday_privateframe.mp4" not in page.output_preview.text()
    job = page._selected_job()
    page._last_job = job
    page._processing_progress(3, 10, "analysis")
    assert page.progress_bar.maximum() == 10
    assert page.progress_bar.value() == 3
    assert page.stage_label.text() == "Analyzing video frames: 3/10"

    page.output_mode.setCurrentIndex(page.output_mode.findData("json_and_video"))
    assert page.preserve_audio.isEnabled()

    page.recognition_policy.setCurrentIndex(page.recognition_policy.findData("exempt"))
    assert page.gallery_dir.isEnabled()
    assert page.target_persons.isEnabled()
    assert page.recognition_profile.isEnabled()
    assert not page.gallery_row.isHidden()
    assert not page.selective_privacy_note.isHidden()

    page._set_running(True)
    for widget in (
        page.analysis_mode,
        page.recognition_policy,
        page.more_options_button,
        page.between_scan_frames,
        page.box_scale,
        page.video_preset,
        page.video_crf,
        page.preserve_audio,
        page.gallery_dir,
        page.browse_gallery_button,
        page.target_persons,
        page.recognition_profile,
    ):
        assert not widget.isEnabled()
    page._set_running(False)
    assert page.more_options_button.isEnabled()
    assert page.gallery_dir.isEnabled()
    page.close()


def test_privateframe_gallery_scan_multiselect_and_job_validation(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.pages.privateframe_page import PrivateFramePage

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    config = AppConfig(workspace_path=str(tmp_path), auto_load_model=False)

    class Context:
        pass

    context = Context()
    context.config = config
    page = PrivateFramePage(context)
    source = tmp_path / "holiday.mp4"
    source.write_bytes(b"video")
    output = tmp_path / "output"
    gallery = tmp_path / "gallery"
    (gallery / "Alice").mkdir(parents=True)
    (gallery / "Bob").mkdir()
    (gallery / "not-a-person.txt").write_text("ignored", encoding="utf-8")
    (gallery / ".hidden-person").mkdir()
    try:
        (gallery / "Linked").symlink_to(gallery / "Alice", target_is_directory=True)
    except OSError:
        pass

    page.video_input.set_path(str(source))
    page.output_dir.setText(str(output))
    page.recognition_policy.setCurrentIndex(page.recognition_policy.findData("exempt"))
    page.gallery_dir.setText(str(gallery))
    page._refresh_gallery_people()

    people = [
        page.target_persons.item(index).text()
        for index in range(page.target_persons.count())
    ]
    assert people == ["Alice", "Bob"]
    with pytest.raises(ValueError, match="Select at least one target person"):
        page._selected_job()

    page.target_persons.item(0).setSelected(True)
    page.target_persons.item(1).setSelected(True)
    page.analysis_mode.setCurrentIndex(page.analysis_mode.findData(15))
    page.between_scan_frames.setCurrentIndex(
        page.between_scan_frames.findData("visual")
    )
    page.box_scale.setCurrentIndex(page.box_scale.findData(1.15))
    page.video_preset.setCurrentIndex(page.video_preset.findData("veryfast"))
    page.video_crf.setCurrentIndex(page.video_crf.findData(23))
    page.recognition_profile.setCurrentIndex(page.recognition_profile.findData("fast"))

    job = page._selected_job()

    assert job.config_overrides["scan.max_analysis_fps"] == 15
    assert job.config_overrides["tracking.between_scan_frames"] == "visual"
    assert job.config_overrides["render.redaction.box_scale"] == 1.15
    assert job.config_overrides["render.video_output.preset"] == "veryfast"
    assert job.config_overrides["render.video_output.rate_control.quality"] == 23
    assert job.config_overrides["recognition.gallery_dir"] == str(gallery.resolve())
    assert job.config_overrides["recognition.target_persons"] == ["Alice", "Bob"]
    assert job.config_overrides["recognition.profile"] == "fast"
    page.close()


def test_existing_json_is_included_in_replace_confirmation(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QMessageBox

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.pages import privateframe_page

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    config = AppConfig(workspace_path=str(tmp_path), auto_load_model=False)

    class Context:
        pass

    context = Context()
    context.config = config
    page = privateframe_page.PrivateFramePage(context)
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    output = tmp_path / "output"
    output.mkdir()
    (output / "input_privateframe.json").write_text("{}", encoding="utf-8")
    page.video_input.set_path(str(source))
    page.output_dir.setText(str(output))
    page.output_mode.setCurrentIndex(page.output_mode.findData("json_only"))
    confirmations = []

    def reject_replace(*args):
        confirmations.append(args)
        return QMessageBox.No

    monkeypatch.setattr(QMessageBox, "question", reject_replace)
    monkeypatch.setattr(
        page,
        "run_task",
        lambda *_args, **_kwargs: pytest.fail("declined replacement must not run"),
    )

    page.start_processing()

    assert len(confirmations) == 1
    assert page._running is False
    assert page._last_job is None
    page.close()


def test_privateframe_refresh_recomputes_resolved_provider(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core import face_engine
    from insightface.gui.core.config import AppConfig
    from insightface.gui.pages.privateframe_page import PrivateFramePage

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    available = ["CPUExecutionProvider"]
    monkeypatch.setattr(
        face_engine,
        "available_execution_providers",
        lambda: list(available),
    )
    config = AppConfig(
        workspace_path=str(tmp_path),
        auto_load_model=False,
        provider="Auto",
    )

    class Context:
        pass

    context = Context()
    context.config = config
    page = PrivateFramePage(context)
    assert page.provider_label.text() == "CPUExecutionProvider"

    available[:] = [
        "CoreMLExecutionProvider",
        "AzureExecutionProvider",
        "CPUExecutionProvider",
    ]
    page.refresh()

    assert page.provider_label.text() == "CoreMLExecutionProvider"
    assert "CPUExecutionProvider (fallback)" in page.provider_label.toolTip()
    assert "AzureExecutionProvider" not in page.provider_label.toolTip()
    page.close()


def test_privateframe_page_runs_python_api_on_background_worker(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import StudioContext, configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.core.face_engine import FaceEngine
    from insightface.gui.core.storage import Storage
    from insightface.gui.main_window import MainWindow
    from insightface.gui.pages import privateframe_page

    configure_qt_plugin_paths()
    app = QApplication.instance() or QApplication([])
    config = AppConfig(
        workspace_path=str(tmp_path / "workspace"),
        model_root=str(tmp_path / "models"),
        auto_load_model=False,
        safe_mode=True,
        ui_language="en",
    )
    window = MainWindow(
        StudioContext(
            config,
            True,
            Storage(config.database_path),
            FaceEngine(model_name=config.model_name, root=config.model_root),
            str(tmp_path / "app.log"),
        )
    )
    page = window.page_registry.get("private_frame")
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    page.video_input.set_path(str(source))
    page.output_dir.setText(str(tmp_path / "output"))
    page.output_mode.setCurrentIndex(page.output_mode.findData("json_only"))
    main_thread = threading.get_ident()
    worker_threads = []
    release = threading.Event()

    def fake_run(job, *, progress=None, is_cancelled=None):
        worker_threads.append(threading.get_ident())
        assert job.output_mode == "json_only"
        assert job.result_path.name == "input_privateframe.json"
        assert job.redacted_path is None
        release.wait(timeout=1.0)
        assert is_cancelled is not None and not is_cancelled()
        assert progress is not None
        progress(1, 1, "analysis")
        return {
            "analysis": {
                "frame_count": 1,
                "accepted_tracks": 0,
                "timings": {"analysis_seconds": 0.01},
            },
            "render": {"frame_count": 1, "seconds": 0.01},
        }

    monkeypatch.setattr(privateframe_page, "run_privateframe_job", fake_run)

    page.start_processing()
    assert page._running is True
    assert page.start_button.isEnabled() is False
    assert page.open_models_button.isEnabled() is False
    assert window.context.privateframe_jobs_in_progress == 1
    release.set()

    loop = QEventLoop()
    poll = QTimer()
    poll.timeout.connect(lambda: loop.quit() if not page._running else None)
    poll.start(5)
    QTimer.singleShot(1000, loop.quit)
    loop.exec()
    poll.stop()

    assert page._running is False
    assert window.context.privateframe_jobs_in_progress == 0
    assert worker_threads and worker_threads[0] != main_thread
    assert page.progress_bar.value() == 100
    assert page.start_button.isEnabled()
    assert "Analysis JSON:" in page.summary.toPlainText()
    assert "Video rendering: skipped (JSON only)" in page.summary.toPlainText()
    assert "Total seconds: 0.01" in page.summary.toPlainText()
    window.close()
    app.processEvents()
