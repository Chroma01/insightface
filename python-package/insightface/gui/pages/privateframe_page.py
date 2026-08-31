"""PrivateFrame video redaction workspace."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import av
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...app.privateframe.output_paths import default_output_paths
from ...app.privateframe.pipeline import (
    analyze_streaming_pipeline,
    run_streaming_pipeline,
)
from ..core.face_engine import provider_runtime_display
from ..core.i18n import tr
from ..core.tooltips import set_button_tooltip
from ..widgets.drop_input import DropInput
from .base import BasePage


_CONFIG_DIR = Path(__file__).resolve().parents[2] / "app" / "privateframe" / "configs"
_BASE_CONFIG_NAME = "base.yaml"
_MODEL_PACKAGES = {"raccoon_s", "raccoon_l"}
_PERFORMANCE_MODES = {"normal", "fast", "ultra_fast"}
_REDACTION_METHODS = {"gaussian", "mosaic"}
_OUTPUT_MODES = {"json_and_video", "json_only"}
_BETWEEN_SCAN_MODES = {"auto", "interpolate", "visual"}
_BOX_SCALES = {1.0, 1.15, 1.30}
_VIDEO_PRESETS = {"veryfast", "medium", "slow"}
_VIDEO_CRF_VALUES = {18, 23, 28}
_RECOGNITION_MODES = {"all", "exempt", "blur_only"}
_RECOGNITION_PROFILES = {"fast", "balanced", "accurate"}
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}
_PROVIDER_NAMES = {
    "auto": "auto",
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "CPUExecutionProvider": "CPUExecutionProvider",
    "CUDAExecutionProvider": "CUDAExecutionProvider",
    "CoreMLExecutionProvider": "CoreMLExecutionProvider",
}


@dataclass(frozen=True)
class PrivateFrameJob:
    input_path: Path
    output_dir: Path
    config_path: Path
    workdir: Path
    result_path: Path
    redacted_path: Path | None
    output_mode: str
    config_overrides: dict[str, object]


def build_privateframe_job(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    model_package: str,
    performance_mode: str,
    redaction_method: str,
    runtime_provider: str,
    preserve_aac_audio: bool = True,
    output_mode: str = "json_and_video",
    custom_frame_stride: int | str | None = None,
    between_scan_frames: str = "auto",
    box_scale: float = 1.0,
    video_preset: str = "medium",
    video_crf: int = 18,
    recognition_mode: str = "all",
    recognition_gallery_dir: str | Path | None = None,
    recognition_target_persons: Sequence[str] = (),
    recognition_profile: str = "balanced",
) -> PrivateFrameJob:
    """Validate GUI selections and materialize deterministic output paths."""

    source = Path(input_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if model_package not in _MODEL_PACKAGES:
        raise ValueError(f"Unsupported PrivateFrame model package: {model_package}")
    if performance_mode not in _PERFORMANCE_MODES:
        raise ValueError(f"Unsupported performance mode: {performance_mode}")
    if redaction_method not in _REDACTION_METHODS:
        raise ValueError(f"Unsupported redaction method: {redaction_method}")
    if output_mode not in _OUTPUT_MODES:
        raise ValueError(f"Unsupported PrivateFrame output mode: {output_mode}")
    if between_scan_frames not in _BETWEEN_SCAN_MODES:
        raise ValueError(
            f"Unsupported between-scan processing mode: {between_scan_frames}"
        )
    try:
        normalized_box_scale = float(box_scale)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported redaction box scale: {box_scale}") from exc
    if normalized_box_scale not in _BOX_SCALES:
        raise ValueError(f"Unsupported redaction box scale: {box_scale}")
    if video_preset not in _VIDEO_PRESETS:
        raise ValueError(f"Unsupported video preset: {video_preset}")
    if type(video_crf) is not int or video_crf not in _VIDEO_CRF_VALUES:
        raise ValueError(f"Unsupported video CRF quality: {video_crf}")
    if recognition_mode not in _RECOGNITION_MODES:
        raise ValueError(f"Unsupported recognition policy: {recognition_mode}")
    if recognition_profile not in _RECOGNITION_PROFILES:
        raise ValueError(f"Unsupported recognition profile: {recognition_profile}")

    normalized_stride: int | None
    if custom_frame_stride is None or (
        isinstance(custom_frame_stride, str)
        and custom_frame_stride in {"", "preset", "auto"}
    ):
        normalized_stride = None
    elif type(custom_frame_stride) is int:
        normalized_stride = custom_frame_stride
        if normalized_stride not in {1, 2, 3, 4}:
            raise ValueError(
                f"Unsupported custom frame stride: {custom_frame_stride}"
            )
    elif isinstance(custom_frame_stride, str) and custom_frame_stride in {
        "1",
        "2",
        "3",
        "4",
    }:
        normalized_stride = int(custom_frame_stride)
    else:
        raise ValueError(f"Unsupported custom frame stride: {custom_frame_stride}")
    provider = _PROVIDER_NAMES.get(
        runtime_provider, _PROVIDER_NAMES.get(runtime_provider.lower())
    )
    if provider is None:
        raise ValueError(f"Unsupported PrivateFrame provider: {runtime_provider}")

    config_path = (_CONFIG_DIR / _BASE_CONFIG_NAME).resolve()
    paths = default_output_paths(source, destination)
    render_video = output_mode == "json_and_video"
    overrides: dict[str, object] = {
        "models.name": model_package,
        "scan.performance_mode": performance_mode,
        "runtime.provider": provider,
        "render.redaction.method": redaction_method,
        "render.redaction.box_scale": normalized_box_scale,
        "render.video_output.preset": video_preset,
        "render.video_output.rate_control.mode": "crf",
        "render.video_output.rate_control.quality": video_crf,
        "render.video_output.audio.redacted": (
            "aac" if render_video and preserve_aac_audio else "none"
        ),
        "recognition.mode": recognition_mode,
    }
    if normalized_stride is not None:
        overrides["scan.frame_stride"] = normalized_stride
    if between_scan_frames != "auto":
        overrides["tracking.between_scan_frames"] = between_scan_frames

    # ``all`` is deliberately a complete early exit: stale GUI gallery text is
    # not resolved, inspected, or sent to the PrivateFrame configuration.
    if recognition_mode != "all":
        gallery_text = str(recognition_gallery_dir or "").strip()
        if not gallery_text:
            raise ValueError(
                "Select a recognition gallery directory for the selected privacy policy."
            )
        unresolved_gallery = Path(gallery_text).expanduser()
        if unresolved_gallery.is_symlink():
            raise ValueError("The recognition gallery directory must not be a symlink.")
        gallery = unresolved_gallery.resolve()
        if not gallery.is_dir():
            raise ValueError("The recognition gallery directory does not exist.")
        if isinstance(recognition_target_persons, (str, bytes)):
            raise ValueError("Select at least one target person from the gallery.")
        targets = [str(person).strip() for person in recognition_target_persons]
        if not targets or any(not person for person in targets):
            raise ValueError("Select at least one target person from the gallery.")
        if len(set(targets)) != len(targets):
            raise ValueError("Target persons must not contain duplicates.")
        gallery_people = {
            child.name
            for child in gallery.iterdir()
            if child.is_dir()
            and not child.is_symlink()
            and not child.name.startswith(".")
        }
        missing = sorted(set(targets) - gallery_people)
        if missing:
            raise ValueError(
                "Selected target persons are absent from the gallery: "
                + ", ".join(missing)
            )
        overrides.update(
            {
                "recognition.gallery_dir": str(gallery),
                "recognition.target_persons": targets,
                "recognition.profile": recognition_profile,
            }
        )
    return PrivateFrameJob(
        input_path=source,
        output_dir=destination,
        config_path=config_path,
        workdir=paths.workdir,
        result_path=paths.result_json,
        redacted_path=paths.result_video if render_video else None,
        output_mode=output_mode,
        config_overrides=overrides,
    )


def run_privateframe_job(
    job: PrivateFrameJob,
    *,
    progress: Callable[[int, int, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run the in-process PrivateFrame API from a GUI worker thread."""

    if is_cancelled is not None and is_cancelled():
        raise InterruptedError("PrivateFrame operation was cancelled")
    overrides = dict(job.config_overrides)
    if job.output_mode == "json_only":
        analysis = analyze_streaming_pipeline(
            config_path=job.config_path,
            input_path=job.input_path,
            workdir=job.workdir,
            result_path=job.result_path,
            config_overrides=overrides,
            config_override_root=job.config_path.parent,
            progress=progress,
            is_cancelled=is_cancelled,
        )
        return {
            "analysis": analysis,
            "render": {},
            "gui": {
                "output_mode": job.output_mode,
                "source_audio_codec": None,
                "audio_output_mode": "none",
            },
        }
    requested_audio_mode = str(overrides["render.video_output.audio.redacted"])
    source_audio_codec = _source_audio_codec(job.input_path)
    if is_cancelled is not None and is_cancelled():
        raise InterruptedError("PrivateFrame operation was cancelled")
    if requested_audio_mode == "aac" and source_audio_codec not in {None, "aac"}:
        # The current PyAV writer can remux AAC but does not yet transcode
        # arbitrary source audio. Drop incompatible audio before the expensive
        # analysis/render pass instead of failing after the full video encode.
        overrides["render.video_output.audio.redacted"] = "none"
    result = run_streaming_pipeline(
        config_path=job.config_path,
        input_path=job.input_path,
        debug_path=None,
        redacted_path=job.redacted_path,
        workdir=job.workdir,
        result_path=job.result_path,
        config_overrides=overrides,
        config_override_root=job.config_path.parent,
        progress=progress,
        is_cancelled=is_cancelled,
    )
    result["gui"] = {
        "output_mode": job.output_mode,
        "source_audio_codec": source_audio_codec,
        "audio_output_mode": overrides["render.video_output.audio.redacted"],
    }
    return result


def _source_audio_codec(source: Path) -> str | None:
    container = av.open(str(source))
    try:
        stream = next(iter(container.streams.audio), None)
        if stream is None:
            return None
        return str(stream.codec_context.name)
    finally:
        container.close()


class PrivateFramePage(BasePage):
    def __init__(self, context, parent=None):
        super().__init__(
            context,
            "PrivateFrame Video Privacy",
            "Upload a local video, analyze faces to JSON, and optionally create "
            "a face-redacted video without uploading media.",
            parent,
        )
        self.setObjectName("privateFramePage")
        self._worker = None
        self._running = False
        self._cancel_requested = False
        self._last_job: PrivateFrameJob | None = None

        self.video_input = DropInput(
            "Input Video",
            extensions=_VIDEO_EXTENSIONS,
            dialog_filter=(
                "Videos (*.mp4 *.mov *.m4v *.mkv *.avi *.webm);;All Files (*)"
            ),
        )
        self.video_input.setObjectName("privateFrameVideoInput")
        self.video_input.title_label.setVisible(False)
        self.video_input.path_label.setMinimumHeight(60)
        self.video_input.pathsChanged.connect(
            lambda _paths: self._update_output_preview()
        )

        input_card, input_layout = self.card()
        input_layout.addWidget(self.video_input)

        self.output_dir = QLineEdit(str(Path(context.config.export_dir).expanduser()))
        self.output_dir.setObjectName("privateFrameOutputDirectory")
        self.output_dir.setPlaceholderText("Choose an output directory")
        self.output_dir.textChanged.connect(self._update_output_preview)
        self.browse_output_button = QPushButton("Browse")
        self.browse_output_button.clicked.connect(self._browse_output_directory)
        set_button_tooltip(self.browse_output_button)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir, 1)
        output_row.addWidget(self.browse_output_button)
        input_layout.addWidget(QLabel("Output Directory"))
        input_layout.addLayout(output_row)
        self.output_preview = QLabel()
        self.output_preview.setObjectName("privateFrameOutputPreview")
        self.output_preview.setWordWrap(True)
        self.output_preview.setProperty("role", "muted")
        input_layout.addWidget(self.output_preview)
        self.content.addWidget(input_card)

        options_card, options_layout = self.card()
        options_form = QFormLayout()
        self.model_package = QComboBox()
        self.model_package.setObjectName("privateFrameModelPackage")
        self.model_package.setProperty("i18nItems", True)
        self.model_package.addItem("Raccoon S (faster)", "raccoon_s")
        self.model_package.addItem("Raccoon L (more accurate)", "raccoon_l")
        self.performance_mode = QComboBox()
        self.performance_mode.setObjectName("privateFramePerformanceMode")
        self.performance_mode.setProperty("i18nItems", True)
        self.performance_mode.addItem("Normal (stride 1)", "normal")
        self.performance_mode.addItem("Fast (stride 2)", "fast")
        self.performance_mode.addItem("Ultra Fast (stride 4)", "ultra_fast")
        self.redaction_method = QComboBox()
        self.redaction_method.setObjectName("privateFrameRedactionMethod")
        self.redaction_method.setProperty("i18nItems", True)
        self.redaction_method.addItem("Gaussian blur", "gaussian")
        self.redaction_method.addItem("Mosaic", "mosaic")
        self.output_mode = QComboBox()
        self.output_mode.setObjectName("privateFrameOutputMode")
        self.output_mode.setProperty("i18nItems", True)
        self.output_mode.addItem("JSON + redacted video", "json_and_video")
        self.output_mode.addItem("JSON only (edit or render later)", "json_only")
        self.output_mode.currentIndexChanged.connect(self._output_mode_changed)
        self.recognition_policy = QComboBox()
        self.recognition_policy.setObjectName("privateFrameRecognitionPolicy")
        self.recognition_policy.setProperty("i18nItems", True)
        self.recognition_policy.addItem(
            "Blur every face (no identity recognition)", "all"
        )
        self.recognition_policy.addItem(
            "Exempt selected people; blur everyone else", "exempt"
        )
        self.recognition_policy.addItem(
            "Blur selected people only", "blur_only"
        )
        self.recognition_policy.currentIndexChanged.connect(
            self._recognition_policy_changed
        )
        provider_name, provider_tooltip = provider_runtime_display(
            str(context.config.provider)
        )
        self.provider_label = QLabel(provider_name)
        self.provider_label.setObjectName("privateFrameProvider")
        self.provider_label.setToolTip(provider_tooltip)
        model_provider = QWidget()
        model_provider_layout = QHBoxLayout(model_provider)
        model_provider_layout.setContentsMargins(0, 0, 0, 0)
        model_provider_layout.addWidget(self.model_package, 1)
        model_provider_layout.addWidget(QLabel("Provider"))
        model_provider_layout.addWidget(self.provider_label)
        options_form.addRow("Model package", model_provider)
        options_form.addRow("Performance", self.performance_mode)
        options_form.addRow("Privacy policy", self.recognition_policy)
        options_form.addRow("Redaction", self.redaction_method)
        options_form.addRow("Output mode", self.output_mode)
        self.more_options_button = QPushButton("More Options…")
        self.more_options_button.setObjectName("privateFrameMoreOptionsButton")
        self.more_options_button.clicked.connect(self._show_more_options)
        set_button_tooltip(self.more_options_button)
        options_form.addRow("Advanced settings", self.more_options_button)
        options_layout.addLayout(options_form)
        self.content.addWidget(options_card)

        self._create_more_options_dialog()

        self.start_button = QPushButton("Start Processing")
        self.start_button.setObjectName("privateFrameStartButton")
        self.start_button.clicked.connect(self.start_processing)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("privateFrameCancelButton")
        self.cancel_button.clicked.connect(self.cancel_processing)
        self.cancel_button.setEnabled(False)
        self.open_output_button = QPushButton("Open Output Folder")
        self.open_output_button.setObjectName("privateFrameOpenOutputButton")
        self.open_output_button.clicked.connect(self.open_output_folder)
        self.open_output_button.setEnabled(False)
        for button in (
            self.start_button,
            self.cancel_button,
            self.open_output_button,
        ):
            set_button_tooltip(button)
        self.content.addWidget(
            self.row(
                self.start_button,
                self.cancel_button,
                self.open_output_button,
            )
        )

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("privateFrameProgress")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(tr("Ready", context.config.ui_language))
        self.stage_label = QLabel("Ready")
        self.stage_label.setObjectName("privateFrameStage")
        self.stage_label.setWordWrap(True)
        progress_row = QHBoxLayout()
        progress_row.addWidget(self.progress_bar, 2)
        progress_row.addWidget(self.stage_label, 1)
        self.content.addLayout(progress_row)

        self.summary = QPlainTextEdit()
        self.summary.setObjectName("privateFrameSummary")
        self.summary.setReadOnly(True)
        self.summary.setPlaceholderText(
            tr(
                "Processing details and output paths will appear here.",
                context.config.ui_language,
            )
        )
        self.summary.setMaximumHeight(180)
        self.content.addWidget(self.summary)
        self.content.addWidget(
            self.notice(
                "PrivateFrame runs locally. The selected Raccoon model package may "
                "be downloaded to the default InsightFace model directory on first use."
            )
        )
        self._update_output_option_controls()
        self._update_output_preview()

    def _create_more_options_dialog(self) -> None:
        """Create the persistent non-modal dialog once so closing keeps values."""

        self.more_options_dialog = QDialog(self)
        self.more_options_dialog.setObjectName("privateFrameMoreOptionsDialog")
        self.more_options_dialog.setWindowTitle("PrivateFrame More Options")
        self.more_options_dialog.setModal(False)
        self.more_options_dialog.setMinimumWidth(560)

        layout = QVBoxLayout(self.more_options_dialog)
        form = QFormLayout()
        self.more_options_form = form

        self.custom_frame_stride = QComboBox()
        self.custom_frame_stride.setObjectName("privateFrameCustomStride")
        self.custom_frame_stride.setProperty("i18nItems", True)
        self.custom_frame_stride.addItem("Follow performance preset", None)
        for stride in (1, 2, 3, 4):
            self.custom_frame_stride.addItem(f"Every {stride} frame(s)", stride)

        self.between_scan_frames = QComboBox()
        self.between_scan_frames.setObjectName("privateFrameBetweenScanFrames")
        self.between_scan_frames.setProperty("i18nItems", True)
        self.between_scan_frames.addItem("Automatic", "auto")
        self.between_scan_frames.addItem("Interpolate", "interpolate")
        self.between_scan_frames.addItem("Visual tracking", "visual")

        self.box_scale = QComboBox()
        self.box_scale.setObjectName("privateFrameBoxScale")
        self.box_scale.setProperty("i18nItems", True)
        self.box_scale.addItem("Standard (1.00×)", 1.0)
        self.box_scale.addItem("Extra margin (1.15×)", 1.15)
        self.box_scale.addItem("Maximum margin (1.30×)", 1.30)

        self.video_preset = QComboBox()
        self.video_preset.setObjectName("privateFrameVideoPreset")
        self.video_preset.setProperty("i18nItems", True)
        self.video_preset.addItem("Very fast encoding", "veryfast")
        self.video_preset.addItem("Medium encoding", "medium")
        self.video_preset.addItem("Slow encoding", "slow")
        self.video_preset.setCurrentIndex(self.video_preset.findData("medium"))

        self.video_crf = QComboBox()
        self.video_crf.setObjectName("privateFrameVideoCrf")
        self.video_crf.setProperty("i18nItems", True)
        self.video_crf.addItem("High quality (CRF 18)", 18)
        self.video_crf.addItem("Balanced size (CRF 23)", 23)
        self.video_crf.addItem("Smaller file (CRF 28)", 28)

        self.preserve_audio = QCheckBox("Preserve AAC audio when available")
        self.preserve_audio.setObjectName("privateFramePreserveAudio")
        self.preserve_audio.setChecked(True)
        self.preserve_audio.setToolTip(
            "Existing AAC audio is remuxed. Other source audio formats are omitted automatically."
        )
        form.addRow("Custom frame stride", self.custom_frame_stride)
        form.addRow("Between scanned frames", self.between_scan_frames)
        form.addRow("Face coverage", self.box_scale)
        form.addRow("Encoding speed", self.video_preset)
        form.addRow("Video quality", self.video_crf)
        form.addRow("Audio", self.preserve_audio)

        self.gallery_dir = QLineEdit()
        self.gallery_dir.setObjectName("privateFrameGalleryDirectory")
        self.gallery_dir.setPlaceholderText(
            "Folder containing one first-level folder per person"
        )
        self.gallery_dir.editingFinished.connect(self._refresh_gallery_people)
        self.browse_gallery_button = QPushButton("Browse")
        self.browse_gallery_button.setObjectName("privateFrameBrowseGalleryButton")
        self.browse_gallery_button.clicked.connect(self._browse_gallery_directory)
        set_button_tooltip(self.browse_gallery_button)
        self.gallery_row = QWidget()
        gallery_layout = QHBoxLayout(self.gallery_row)
        gallery_layout.setContentsMargins(0, 0, 0, 0)
        gallery_layout.addWidget(self.gallery_dir, 1)
        gallery_layout.addWidget(self.browse_gallery_button)
        form.addRow("Recognition gallery", self.gallery_row)

        self.target_persons = QListWidget()
        self.target_persons.setObjectName("privateFrameTargetPersons")
        self.target_persons.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.target_persons.setMinimumHeight(110)
        form.addRow("Target people", self.target_persons)

        self.gallery_status = QLabel("Select a gallery to list people.")
        self.gallery_status.setObjectName("privateFrameGalleryStatus")
        self.gallery_status.setWordWrap(True)
        self.gallery_status.setProperty("role", "muted")
        form.addRow("", self.gallery_status)

        self.recognition_profile = QComboBox()
        self.recognition_profile.setObjectName("privateFrameRecognitionProfile")
        self.recognition_profile.setProperty("i18nItems", True)
        self.recognition_profile.addItem("Fast recognition", "fast")
        self.recognition_profile.addItem("Balanced recognition", "balanced")
        self.recognition_profile.addItem("Accurate recognition", "accurate")
        self.recognition_profile.setCurrentIndex(
            self.recognition_profile.findData("balanced")
        )
        form.addRow("Recognition profile", self.recognition_profile)
        layout.addLayout(form)

        self.selective_privacy_note = QLabel(
            "Selective policies are fail-safe: faces with uncertain identity remain blurred."
        )
        self.selective_privacy_note.setObjectName("privateFrameRecognitionNotice")
        self.selective_privacy_note.setWordWrap(True)
        self.selective_privacy_note.setProperty("role", "muted")
        layout.addWidget(self.selective_privacy_note)

        self.close_more_options_button = QPushButton("Close")
        self.close_more_options_button.setObjectName(
            "privateFrameCloseMoreOptionsButton"
        )
        self.close_more_options_button.clicked.connect(self.more_options_dialog.close)
        set_button_tooltip(self.close_more_options_button)
        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_row.addWidget(self.close_more_options_button)
        layout.addLayout(close_row)

    def _show_more_options(self) -> None:
        self.more_options_dialog.show()
        self.more_options_dialog.raise_()
        self.more_options_dialog.activateWindow()

    def _browse_gallery_directory(self) -> None:
        initial = self.gallery_dir.text().strip() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Recognition Gallery",
            str(Path(initial).expanduser()),
        )
        if folder:
            self.gallery_dir.setText(folder)
            self._refresh_gallery_people()

    def _refresh_gallery_people(self) -> None:
        """Populate targets from real first-level folders in selective mode."""

        if self.recognition_policy.currentData() == "all":
            return
        selected_before = {
            item.text() for item in self.target_persons.selectedItems()
        }
        self.target_persons.clear()
        text = self.gallery_dir.text().strip()
        language = self.context.config.ui_language
        if not text:
            self.gallery_status.setText(
                tr("Select a gallery to list people.", language)
            )
            return
        unresolved = Path(text).expanduser()
        if unresolved.is_symlink():
            self.gallery_status.setText(
                tr("Gallery directories cannot be symlinks.", language)
            )
            return
        gallery = unresolved.resolve()
        if not gallery.is_dir():
            self.gallery_status.setText(
                tr("The gallery directory does not exist.", language)
            )
            return
        people = sorted(
            child.name
            for child in gallery.iterdir()
            if child.is_dir()
            and not child.is_symlink()
            and not child.name.startswith(".")
        )
        self.target_persons.addItems(people)
        for index in range(self.target_persons.count()):
            item = self.target_persons.item(index)
            if item.text() in selected_before:
                item.setSelected(True)
        if people:
            self.gallery_status.setText(
                tr(
                    "Found {count} people. Select one or more targets.",
                    language,
                ).format(count=len(people))
            )
        else:
            self.gallery_status.setText(
                tr(
                    "No first-level person folders were found in this gallery.",
                    language,
                )
            )

    def _browse_output_directory(self) -> None:
        initial = self.output_dir.text().strip() or self.context.config.export_dir
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            str(Path(initial).expanduser()),
        )
        if folder:
            self.output_dir.setText(folder)

    def _selected_job(self) -> PrivateFrameJob:
        source = self.video_input.path()
        if not source:
            raise ValueError("Select an input video first.")
        if not Path(source).expanduser().is_file():
            raise ValueError("The selected input video does not exist.")
        output = self.output_dir.text().strip()
        if not output:
            raise ValueError("Select an output directory.")
        recognition_mode = str(self.recognition_policy.currentData())
        recognition_targets = (
            [item.text() for item in self.target_persons.selectedItems()]
            if recognition_mode != "all"
            else []
        )
        job = build_privateframe_job(
            input_path=source,
            output_dir=output,
            model_package=str(self.model_package.currentData()),
            performance_mode=str(self.performance_mode.currentData()),
            redaction_method=str(self.redaction_method.currentData()),
            runtime_provider=str(self.context.config.provider),
            preserve_aac_audio=self.preserve_audio.isChecked(),
            output_mode=str(self.output_mode.currentData()),
            custom_frame_stride=self.custom_frame_stride.currentData(),
            between_scan_frames=str(self.between_scan_frames.currentData()),
            box_scale=float(self.box_scale.currentData()),
            video_preset=str(self.video_preset.currentData()),
            video_crf=int(self.video_crf.currentData()),
            recognition_mode=recognition_mode,
            recognition_gallery_dir=(
                self.gallery_dir.text().strip()
                if recognition_mode != "all"
                else None
            ),
            recognition_target_persons=recognition_targets,
            recognition_profile=str(self.recognition_profile.currentData()),
        )
        if not job.config_path.is_file():
            raise RuntimeError(
                f"PrivateFrame configuration is missing: {job.config_path}"
            )
        return job

    def _update_output_preview(self, *_args) -> None:
        source = self.video_input.path()
        output = self.output_dir.text().strip()
        if not source or not output:
            self.output_preview.setText(
                tr(
                    "Select a video and output directory.",
                    self.context.config.ui_language,
                )
            )
            return
        paths = default_output_paths(source, output)
        language = self.context.config.ui_language
        lines = [f"{tr('Analysis JSON', language)}: {paths.result_json}"]
        if self.output_mode.currentData() == "json_and_video":
            lines.append(f"{tr('Redacted video', language)}: {paths.result_video}")
        self.output_preview.setText("\n".join(lines))

    def _output_mode_changed(self, *_args) -> None:
        self._update_output_option_controls()
        self._update_output_preview()

    def _recognition_policy_changed(self, *_args) -> None:
        self._update_output_option_controls()
        if (
            self.recognition_policy.currentData() != "all"
            and self.gallery_dir.text().strip()
        ):
            self._refresh_gallery_people()

    def _update_output_option_controls(self) -> None:
        video_enabled = (
            not self._running and self.output_mode.currentData() == "json_and_video"
        )
        self.preserve_audio.setEnabled(video_enabled)
        options_enabled = not self._running
        for widget in (
            self.custom_frame_stride,
            self.between_scan_frames,
            self.box_scale,
            self.video_preset,
            self.video_crf,
        ):
            widget.setEnabled(options_enabled)
        selective_enabled = (
            options_enabled and self.recognition_policy.currentData() != "all"
        )
        for widget in (
            self.gallery_dir,
            self.browse_gallery_button,
            self.target_persons,
            self.recognition_profile,
        ):
            widget.setEnabled(selective_enabled)
        for field in (
            self.gallery_row,
            self.target_persons,
            self.gallery_status,
            self.recognition_profile,
        ):
            self.more_options_form.setRowVisible(field, selective_enabled)
        self.selective_privacy_note.setVisible(selective_enabled)
        if self.more_options_dialog.isVisible():
            self.more_options_dialog.adjustSize()

    def start_processing(self) -> None:
        if self._running:
            return
        try:
            job = self._selected_job()
            job.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.show_error(str(exc))
            return

        existing = [
            path
            for path in (job.result_path, job.redacted_path)
            if path is not None and path.exists()
        ]
        if existing:
            answer = QMessageBox.question(
                self,
                tr("Replace Existing Output", self.context.config.ui_language),
                tr(
                    "The selected output already exists and will be replaced. Continue?",
                    self.context.config.ui_language,
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        self._last_job = job
        self._cancel_requested = False
        self._set_running(True)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("Preparing models…")
        if job.output_mode == "json_only":
            self.stage_label.setText("Preparing models and video analysis…")
        else:
            self.stage_label.setText("Preparing models, analysis, and rendering…")
        self.summary.clear()

        def task(progress=None, is_cancelled=None):
            return run_privateframe_job(
                job,
                progress=progress,
                is_cancelled=is_cancelled,
            )

        worker = self.run_task(
            "PrivateFrame video processing",
            task,
            self._processing_complete,
            show_dialog=False,
            on_progress=self._processing_progress,
            on_error=self._processing_error,
            on_finished=self._processing_finished,
        )
        if self._running:
            self._worker = worker

    def cancel_processing(self) -> None:
        if not self._running:
            return
        self._cancel_requested = True
        self.cancel_button.setEnabled(False)
        self.stage_label.setText("Cancelling after the current frame…")
        if self._worker is not None:
            self._worker.cancel()

    def _processing_progress(self, current: int, total: int, message: str) -> None:
        total = max(1, int(total))
        current = max(0, min(int(current), total))
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat("%p%")
        if message == "analysis":
            json_only = (
                self._last_job is not None and self._last_job.output_mode == "json_only"
            )
            analysis_total = total if json_only else max(1, total // 2)
            self.stage_label.setText(
                f"Analyzing video frames: {min(current, analysis_total)}/{analysis_total}"
            )
        elif message == "render":
            analysis_total = total // 2
            render_total = max(1, total - analysis_total)
            self.stage_label.setText(
                f"Rendering output video: {max(0, current - analysis_total)}/{render_total}"
            )
        elif message:
            self.stage_label.setText(message)

    def _processing_complete(self, result: dict[str, Any]) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat(tr("Completed", self.context.config.ui_language))
        self.stage_label.setText("PrivateFrame processing completed.")
        self.open_output_button.setEnabled(True)
        analysis = result.get("analysis", {})
        render = result.get("render") or {}
        gui_result = result.get("gui", {})
        resolved_provider = str(analysis.get("provider", "")).strip()
        if resolved_provider:
            _provider_name, provider_tooltip = provider_runtime_display(
                str(self.context.config.provider)
            )
            self.provider_label.setText(resolved_provider)
            self.provider_label.setToolTip(
                f"Last PrivateFrame run confirmed {resolved_provider}. "
                f"{provider_tooltip}"
            )
        source_audio = gui_result.get("source_audio_codec")
        audio_mode = gui_result.get("audio_output_mode")
        language = self.context.config.ui_language
        lines = [
            tr("PrivateFrame processing completed.", language),
            f"{tr('Analysis JSON', language)}: "
            f"{self._last_job.result_path if self._last_job else ''}",
            f"Frames: {analysis.get('frame_count', render.get('frame_count', ''))}",
            f"Accepted tracks: {analysis.get('accepted_tracks', '')}",
            f"Analysis seconds: {analysis.get('timings', {}).get('analysis_seconds', '')}",
        ]
        if self._last_job and self._last_job.output_mode == "json_and_video":
            lines.insert(
                2,
                f"{tr('Redacted video', language)}: {self._last_job.redacted_path}",
            )
            lines.append(f"Render seconds: {render.get('seconds', '')}")
            if source_audio is None:
                lines.append("Audio: no source audio track")
            elif audio_mode == "aac":
                lines.append("Audio: preserved (AAC)")
            else:
                lines.append(f"Audio: omitted (source codec: {source_audio})")
        else:
            lines.append(tr("Video rendering: skipped (JSON only)", language))
        self.summary.setPlainText("\n".join(lines))
        self.set_status("PrivateFrame processing completed.")

    def _processing_error(self, message: str) -> None:
        if self._cancel_requested or "cancel" in message.lower():
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat(
                tr("Cancelled", self.context.config.ui_language)
            )
            self.stage_label.setText("PrivateFrame processing was cancelled.")
            self.summary.setPlainText(
                "Processing cancelled. Partial work files may remain in the work directory."
            )
            self.set_status("PrivateFrame processing was cancelled.")
            return
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(tr("Failed", self.context.config.ui_language))
        self.stage_label.setText("PrivateFrame processing failed.")
        self.summary.setPlainText(message)
        self.show_error(message)

    def _processing_finished(self) -> None:
        self._worker = None
        self._set_running(False)
        self._cancel_requested = False

    def _set_running(self, running: bool) -> None:
        self._running = running
        for widget in (
            self.video_input,
            self.output_dir,
            self.browse_output_button,
            self.model_package,
            self.performance_mode,
            self.recognition_policy,
            self.redaction_method,
            self.output_mode,
            self.more_options_button,
            self.preserve_audio,
        ):
            widget.setEnabled(not running)
        self.start_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self._update_output_option_controls()
        if running:
            self.open_output_button.setEnabled(False)

    def open_output_folder(self) -> None:
        directory = (
            self._last_job.output_dir
            if self._last_job
            else Path(self.output_dir.text()).expanduser()
        )
        if not directory.is_dir():
            self.show_error("The output directory does not exist.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def refresh(self) -> None:
        provider_name, provider_tooltip = provider_runtime_display(
            str(self.context.config.provider)
        )
        self.provider_label.setText(provider_name)
        self.provider_label.setToolTip(provider_tooltip)
        self._update_output_preview()


__all__ = [
    "PrivateFrameJob",
    "PrivateFramePage",
    "build_privateframe_job",
    "run_privateframe_job",
]
