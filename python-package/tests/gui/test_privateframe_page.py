from __future__ import annotations

import os
import threading

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


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
        performance_mode="ultra_fast",
        redaction_method="mosaic",
        runtime_provider="CUDA",
        preserve_aac_audio=False,
    )

    assert job.config_path.name == "base.yaml"
    assert job.result_path == output / "input_privateframe.json"
    assert job.redacted_path == output / "input_privateframe.mp4"
    assert job.workdir == output / ".input_privateframe_work"
    assert job.output_mode == "json_and_video"
    assert job.config_overrides["models.name"] == "raccoon_l"
    assert job.config_overrides["scan.performance_mode"] == "ultra_fast"
    assert job.config_overrides["runtime.provider"] == "CUDAExecutionProvider"
    assert job.config_overrides["render.redaction.method"] == "mosaic"
    assert job.config_overrides["render.redaction.box_scale"] == 1.0
    assert job.config_overrides["render.video_output.preset"] == "medium"
    assert job.config_overrides["render.video_output.rate_control.mode"] == "crf"
    assert job.config_overrides["render.video_output.rate_control.quality"] == 18
    assert job.config_overrides["render.video_output.audio.redacted"] == "none"
    assert job.config_overrides["recognition.mode"] == "all"
    assert "scan.frame_stride" not in job.config_overrides
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
        performance_mode="fast",
        redaction_method="gaussian",
        runtime_provider="CPU",
        custom_frame_stride=3,
        between_scan_frames="visual",
        box_scale=1.30,
        video_preset="slow",
        video_crf=28,
        recognition_mode="exempt",
        recognition_gallery_dir=gallery,
        recognition_target_persons=["Bob", "Alice"],
        recognition_profile="accurate",
    )

    assert job.config_overrides["scan.frame_stride"] == 3
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
        performance_mode="fast",
        redaction_method="mosaic",
        runtime_provider="CPU",
        custom_frame_stride=3,
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

    assert config["scan"]["frame_stride"] == 3
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
        performance_mode="normal",
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
        ("custom_frame_stride", 1.5),
        ("custom_frame_stride", 5),
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
        "performance_mode": "normal",
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
        performance_mode="fast",
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
    assert captured["config_overrides"]["scan.performance_mode"] == "fast"
    assert captured["config_overrides"]["runtime.provider"] == "auto"


def test_non_aac_audio_is_omitted_before_pipeline(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from insightface.gui.pages import privateframe_page

    source = tmp_path / "input.webm"
    source.write_bytes(b"video")
    job = privateframe_page.build_privateframe_job(
        input_path=source,
        output_dir=tmp_path / "output",
        model_package="raccoon_s",
        performance_mode="normal",
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
        performance_mode="fast",
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

    assert page.model_package.count() == 2
    assert page.performance_mode.count() == 3
    assert page.redaction_method.count() == 2
    assert page.output_mode.count() == 2
    assert page.output_mode.currentData() == "json_and_video"
    assert page.recognition_policy.count() == 3
    assert page.recognition_policy.currentData() == "all"
    assert page.more_options_button.objectName() == "privateFrameMoreOptionsButton"
    assert not page.more_options_dialog.isModal()
    assert page.custom_frame_stride.currentData() is None
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
    page.more_options_button.click()
    assert page.more_options_dialog.isVisible()
    page.video_preset.setCurrentIndex(page.video_preset.findData("slow"))
    page.more_options_dialog.close()
    page.more_options_button.click()
    assert page.video_preset.currentData() == "slow"
    page.more_options_dialog.close()
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

    page.recognition_policy.setCurrentIndex(
        page.recognition_policy.findData("exempt")
    )
    assert page.gallery_dir.isEnabled()
    assert page.target_persons.isEnabled()
    assert page.recognition_profile.isEnabled()
    assert not page.gallery_row.isHidden()
    assert not page.selective_privacy_note.isHidden()

    page._set_running(True)
    for widget in (
        page.recognition_policy,
        page.more_options_button,
        page.custom_frame_stride,
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
    page.recognition_policy.setCurrentIndex(
        page.recognition_policy.findData("exempt")
    )
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
    page.custom_frame_stride.setCurrentIndex(
        page.custom_frame_stride.findData(4)
    )
    page.between_scan_frames.setCurrentIndex(
        page.between_scan_frames.findData("visual")
    )
    page.box_scale.setCurrentIndex(page.box_scale.findData(1.15))
    page.video_preset.setCurrentIndex(page.video_preset.findData("veryfast"))
    page.video_crf.setCurrentIndex(page.video_crf.findData(23))
    page.recognition_profile.setCurrentIndex(
        page.recognition_profile.findData("fast")
    )

    job = page._selected_job()

    assert job.config_overrides["scan.frame_stride"] == 4
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
    release.set()

    loop = QEventLoop()
    poll = QTimer()
    poll.timeout.connect(lambda: loop.quit() if not page._running else None)
    poll.start(5)
    QTimer.singleShot(1000, loop.quit)
    loop.exec()
    poll.stop()

    assert page._running is False
    assert worker_threads and worker_threads[0] != main_thread
    assert page.progress_bar.value() == 100
    assert page.start_button.isEnabled()
    assert "Analysis JSON:" in page.summary.toPlainText()
    assert "Video rendering: skipped (JSON only)" in page.summary.toPlainText()
    window.close()
    app.processEvents()
